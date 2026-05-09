from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import shap
import os
from datetime import datetime, timezone

app = FastAPI()
LOG_PATH = "logs/predictions.csv"

model = joblib.load("models/fraud_model.pkl")
merchant_freq = joblib.load("models/merchant_freq.pkl")
model_columns = joblib.load("models/model_columns.pkl")
threshold = joblib.load("models/threshold.pkl")

explainer = shap.TreeExplainer(model)

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

@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.post("/predict")
def predict(transaction: dict):
    raw_df = pd.DataFrame([transaction])

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

    score = model.predict_proba(df)[:, 1][0]

    flag = int(score >= threshold)

    if score >= threshold:
        risk_level = "HIGH"
    elif score >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    shap_values = explainer.shap_values(df)

    # get top 3 features
    shap_dict = dict(zip(df.columns, shap_values[0]))

    top_features = sorted(
        shap_dict.items(),
        key = lambda x: abs(x[1]),
        reverse = True
    )[:3]

    response = {
        "fraud_probability": float(score),
        "threshold": float(threshold),
        "risk_level": risk_level,
        "flag": flag,
        "top_reasons": [
            {"feature": f, "impact": float(v)} for f, v in top_features
        ]
    }

    log_prediction(raw_df.iloc[0].to_dict(), response)

    return response