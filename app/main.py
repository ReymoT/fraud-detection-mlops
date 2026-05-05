from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI()

model = joblib.load("models/fraud_model.pkl")
merchant_freq = joblib.load("models/merchant_freq.pkl")
model_columns = joblib.load("models/model_columns.pkl")
threshold = joblib.load("models/threshold.pkl")

explainer = shap.TreeExplainer(model)

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
        risk = "HIGH"
    elif score >= 0.5:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    shap_values = explainer.shap_values(df)

    # get top 3 features
    shap_dict = dict(zip(df.columns, shap_values[0]))

    top_features = sorted(
        shap_dict.items(),
        key = lambda x: abs(x[1]),
        reverse = True
    )[:3]

    return {
        "fraud_probability": float(score),
        "threshold": float(threshold),
        "risk_level": risk,
        "flag": flag,
        "top_reasons": [
            {"feature": f, "impact": float(v)} for f, v in top_features
        ]
    }