from fastapi import FastAPI, Response, Header, HTTPException, Request, Depends
import joblib
import pandas as pd
import numpy as np
import shap
import os
from datetime import datetime, timezone
from pydantic import BaseModel
from app.inference_engine import DynamicBatcher
from contextlib import asynccontextmanager
import asyncio
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from collections import defaultdict, deque

API_KEY = os.getenv("API_KEY", "dev-secret-key")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

def verify_api_key(x_api_key: str = Header(default = None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code = 401, detail = "Invalid or missing API key")

request_log = defaultdict(deque)

def rate_limit(request: Request):
    client_ip = request.client.host
    now = time.time()

    window = request_log[client_ip]

    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    if len(window) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code = 429, detail = "Rate limit exceeded")

    window.append(now)



REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "Request latency",
    ["endpoint"]
)

QUEUE_DEPTH = Gauge(
    "fraud_inference_queue_depth",
    "Current inference queue depth"
)

AVG_BATCH_SIZE = Gauge(
    "fraud_inference_avg_batch_size",
    "Average batch size"
)

REJECTED_REQUESTS = Gauge(
    "fraud_inference_rejected_requests",
    "Rejected inference requests"
)

TIMED_OUT_REQUESTS = Gauge(
    "fraud_inference_timed_out_requests",
    "Timed out inference requests"
)

def update_inference_gauges():
    metrics = batcher.metrics()

    QUEUE_DEPTH.set(metrics["queue_depth"])
    AVG_BATCH_SIZE.set(metrics["avg_batch_size"])
    REJECTED_REQUESTS.set(metrics["rejected_requests"])
    TIMED_OUT_REQUESTS.set(metrics["timed_out_requests"])

class Transaction(BaseModel):
    amt: float
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int
    category: str
    gender: str
    state: str
    merchant: str
    trans_date_trans_time: str
    dob: str

LOG_PATH = "logs/predictions.csv"

model = joblib.load("models/fraud_model.pkl")
merchant_freq = joblib.load("models/merchant_freq.pkl")
model_columns = joblib.load("models/model_columns.pkl")
threshold = joblib.load("models/threshold.pkl")

explainer = shap.TreeExplainer(model)

BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", "2"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "128"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "1000"))

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))

def log_prediction(transaction, response):
    os.makedirs("logs", exist_ok = True)

    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **transaction,
        "fraud_probability": response["fraud_probability"],
        "threshold": response["threshold"],
        "risk_level": response["risk_level"],
        "flag": response["flag"]
    }

    df = pd.DataFrame([row])

    file_exists = os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode = "a", header = (not file_exists), index = False)

def prepare_features(transactions):
    raw_df = pd.DataFrame(transactions)

    raw_df["trans_date_trans_time"] = pd.to_datetime(raw_df["trans_date_trans_time"])
    raw_df["dob"] = pd.to_datetime(raw_df["dob"])

    raw_df["trans_hour"] = raw_df["trans_date_trans_time"].dt.hour
    raw_df["trans_dayofweek"] = raw_df["trans_date_trans_time"].dt.dayofweek
    raw_df["trans_month"] = raw_df["trans_date_trans_time"].dt.month
    raw_df["age"] = (raw_df["trans_date_trans_time"] - raw_df["dob"]).dt.days / 365.25
    raw_df["is_night"] = (raw_df["trans_hour"] < 6).astype("int8")

    raw_df["distance"] = haversine(
        raw_df["lat"],
        raw_df["long"],
        raw_df["merch_lat"],
        raw_df["merch_long"],
    )

    df = raw_df.copy()

    df["amt_log"] = np.log1p(df["amt"])
    df["merchant_freq"] = df["merchant"].map(merchant_freq).fillna(0)

    df = pd.get_dummies(df, columns = ["category", "gender", "state"])

    df = df.drop(columns = ["amt", "merchant", "unix_time"], errors = "ignore")

    df = df.reindex(columns = model_columns, fill_value = 0)

    return raw_df, df

batcher = DynamicBatcher(
    model = model,
    preprocess_fn = prepare_features,
    max_batch_size = MAX_BATCH_SIZE,
    batch_timeout_ms = BATCH_TIMEOUT_MS,
    max_queue_size = MAX_QUEUE_SIZE
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await batcher.start()
    yield

app = FastAPI(lifespan = lifespan)

def build_response(score: float, df = None, include_explanations: bool = False):
    flag = int(score >= threshold)

    if score >= threshold:
        risk_level = "HIGH"
    elif score >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    response = {
        "fraud_probability": float(score),
        "threshold": float(threshold),
        "risk_level": risk_level,
        "flag": flag
    }

    if include_explanations and df is not None:
        shap_values = explainer.shap_values(df)

        shap_dict = dict(zip(df.columns, shap_values[0]))

        # get top 3 most influential features
        top_features = sorted(
            shap_dict.items(),
            key = lambda x: abs(x[1]),
            reverse = True,
        )[:3]

        response["top_reasons"] = [
            {"feature": f, "impact": float(v)} for f, v in top_features
        ]

    return response

@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.post("/predict_direct")
def predict_direct(transaction: Transaction, explain: bool = False, log: bool = False, _: None = Depends(verify_api_key), __: None = Depends(rate_limit)):
    start = time.perf_counter()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    try:
        transaction_dict = transaction.model_dump()

        raw_df, df = prepare_features([transaction_dict])

        score = model.predict_proba(df)[:, 1][0]

        response = build_response(score, df, include_explanations = explain)

        if log:
            log_prediction(raw_df.iloc[0].to_dict(), response)

        return response

    finally:
        REQUEST_LATENCY.labels(endpoint = "/predict_direct").observe(
            time.perf_counter() - start
        )

@app.post("/predict")
async def predict(transaction: Transaction, explain: bool = False, log: bool = False, _: None = Depends(verify_api_key), __: None = Depends(rate_limit)):
    start = time.perf_counter()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    try:
        transaction_dict = transaction.model_dump()

        raw_df, df = prepare_features([transaction_dict])

        try:
            score = await asyncio.wait_for(
                batcher.predict(transaction_dict),
                timeout = 5
            )

        except RuntimeError:
            raise HTTPException(
                status_code = 503,
                detail = "Inference queue is full"
            )

        except asyncio.TimeoutError:
            batcher.record_timeout()
            raise HTTPException(
                status_code = 504,
                detail = "Inference request timed out"
            )
        
        # Only preprocess this one row again if explanations/logging are requested
        if explain or log:
            raw_df, df = prepare_features([transaction_dict])
        else:
            raw_df = None
            df = None

        response = build_response(score, df, include_explanations = explain)

        if log:
            log_prediction(raw_df.iloc[0].to_dict(), response)
        
        return response

    finally:
        REQUEST_LATENCY.labels(endpoint = "/predict").observe(
            time.perf_counter() - start
        )


@app.get("/metrics/inference")
def inference_metrics():
    return batcher.metrics()

@app.get("/metrics")
def prometheus_metrics():
    update_inference_gauges()
    return Response(
        content = generate_latest(),
        media_type = CONTENT_TYPE_LATEST
    )