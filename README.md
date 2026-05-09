# Fraud Detection MLOps System
![CI](https://github.com/ReymoT/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)

End-to-end fraud detection MLOps system built on 1M+ credit card transactions. The project includes simulated data stream ingestion, model training,
experiment tracking, explainable inference, monitoring, orchestration, and containerized deployment.


Includes:
- Feature engineering pipeline
- XGBoost model (PR-AUC ~0.86)
- FastAPI inference service
- SHAP explanations
- Apache Airflow DAG for monitoring and retraining
- MLflow experiment tracking
- Evidently AI drift monitoring
- Streamlit dashboard
- Docker + Docker Compose deployment
- GitHub Actions CI/CD with tests/coverage, Docker build, security and vulnerability checks

Model Performance:
- PR-AUC: ~0.86
- Precision @ top 0.5%: ~0.83
- Recall @ top 0.5%: ~0.76
- Fraud rate: ~0.5%


## Setup:
### Clone repo
git clone https://github.com/ReymoT/fraud-detection-mlops.git

cd fraud-detection-mlops

### Run with Docker Compose
docker compose up --build

API: http://localhost:8000/docs  
Dashboard: http://localhost:8501
Airflow: http://localhost:8080

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

## System Architecture

```
Simulated transaction stream
        ↓
FastAPI inference service
        ↓
Prediction logs
        ↓
Airflow scheduled monitoring
        ↓
Evidently drift report
        ↓
Conditional retraining
        ↓
MLflow experiment tracking
        ↓
Model promotion gate
```

## Simulated Data Stream
Run:
```
python streaming/producer.py
```
This sends transaction events to the FastAPI inference service and logs predictions to:
```
logs/predictions.csv
```

## Monitoring and Retraining

Airflow runs the DAG:
```
fraud_monitoring_and_retraining
```
The DAG:
1. Runs Evidently drift monitoring
2. Checks drift status
3. Triggers retraining if drift is detected
4. Logs candidate model metrics to MLflow
5. Promotes the model only if it improves over production metrics


## Notes
The current Airflow setup uses SQLite and SequentialExecutor for local development. A production deployment would use:

PostgreSQL metadata database

CeleryExecutor or KubernetesExecutor

Redis/RabbitMQ broker if using Celery

Cloud object storage for artifacts

Secrets manager

Container registry

Cloud deployment on ECS, SageMaker, or Kubernetes