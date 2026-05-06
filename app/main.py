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
    df = pd.DataFrame([transaction])

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

    log_prediction(transaction, response)

    return response