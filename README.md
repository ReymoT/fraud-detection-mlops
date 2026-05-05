# Fraud Detection MLOps System

End-to-end fraud detection system built on 1M+ transactions with a focus on real-world ML engineering.

Includes:
- Feature engineering pipeline
- XGBoost model (PR-AUC ~0.86)
- FastAPI inference service
- SHAP explanations
- MLflow experiment tracking
- Evidently AI drift monitoring
- Streamlit dashboard
- Docker + Docker Compose deployment
- GitHub Actions CI/CD

Model Performance:
- PR-AUC: ~0.86
- Precision @ top 0.5%: ~0.83
- Recall @ top 0.5%: ~0.76
- Fraud rate: ~0.5%

Architecture:
Training Pipeline → MLflow → Saved Model → FastAPI API → Docker → Dashboard + Monitoring

Setup:
### Clone repo
git clone https://github.com/ReymoT/fraud-detection-mlops.git
cd fraud-detection-mlops

### Run with Docker Compose
docker compose up --build

API: http://localhost:8000/docs  
Dashboard: http://localhost:8501

Example API Request JSON:
```
{
  "amt": 5000,
  "lat": 40.7128,
  "long": -74.0060,
  "merch_lat": 34.0522,
  "merch_long": -118.2437,
  "city_pop": 1000,
  "category": "shopping_net",
  "gender": "M",
  "state": "CA",
  "merchant": "unknown_store",
  "trans_date_trans_time": "2020-12-25 02:30:00",
  "dob": "2003-01-01"
}
```

Example API Response:
```
{
  "fraud_probability": 0.93,
  "threshold": 0.92,
  "risk_level": "HIGH",
  "flag": 1,
  "top_reasons": [
    {"feature": "distance", "impact": 0.8},
    {"feature": "amt_log", "impact": 0.6}
  ]
}
```

Interactive dashboard showing:
- Fraud score distribution
- Flagged transactions
- Risk by category
- High-risk transactions

Evidently AI used for:
- Data drift detection
- Prediction drift tracking
- Feature distribution monitoring
