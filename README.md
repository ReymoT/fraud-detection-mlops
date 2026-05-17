# Fraud Detection MLOps System
![CI](https://github.com/ReymoT/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)

End-to-end fraud detection MLOps system built on 1M+ credit card transactions. The project includes simulated data stream ingestion, model training,
experiment tracking, explainable inference, custom high-throughput inference engine with dynamic batching, backpressure handling and thread safety, benchmarking, monitoring, orchestration, and containerized deployment.


Includes:
- Feature engineering pipeline
- XGBoost model (PR-AUC ~0.86)
- FastAPI inference service
- Custom asynchronous inference engine with dynamic batching
- Performance benchmarking
- SHAP explanations
- Apache Airflow DAG for monitoring and retraining
- MLflow experiment tracking
- Evidently AI drift monitoring
- Streamlit dashboard
- Prometheus/Grafana for benchmarking
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

## Inference Performance Layer
The API supports two inference modes:

Direct Inference
```
client request
    ↓
FastAPI endpoint
    ↓
model.predict_proba(single request)
    ↓
response
```

Dynamic Batched Inference
```
client request
    ↓
async request queue
    ↓
dynamic batcher
    ↓
model.predict_proba(batch)
    ↓
future resolved back to request
```

The batching engine groups requests arriving within a short timeout window into a single model inference call, reducing per request overhead and improving throughput under concurrent load.

- Async request queue
- Dynamic batching
- Configurable batch size and timeout
- Futures-based response resolution
- Runtime metrics endpoint
- Concurrent benchmarking
- p50/p95/p99 latency tracking

### Runtime Stability Features

The inference runtime includes several production style safeguards:

- Bounded async request queue for backpressure handling
- HTTP 503 rejection when the inference queue is full
- Request timeout handling with HTTP 504 responses
- Future cancellation safety checks
- Runtime queue depth and batching metrics
- Configurable maximum queue size

The bounded queue prevents unbounded latency and memory growth under overload conditions.

## Benchmarking results

| Endpoint           |   Throughput |       p50 |       p95 |       p99 |
| ------------------ | -----------: | --------: | --------: | --------: |
| /predict_direct  | 157.45 req/s | 209.43 ms | 320.59 ms | 342.25 ms |
| /predict batched | 187.19 req/s | 208.83 ms | 296.49 ms | 352.47 ms |


The async dynamic batching runtime improved throughput from 157.45 req/s to 187.19 req/s under 50 concurrent clients, an increase of approximately 18.9%.

Median latency (p50) remained nearly identical, while p95 latency improved from 320.59 ms to 296.49 ms, p99 latency increased slightly due to batching queue delays, reflecting the expected throughput vs tail latency tradeoff in dynamic batching systems.

Benchmarks were performed with:
- 10,000 requests
- 50 concurrent clients
- 100 warmup requests
- 2 Uvicorn workers
- BATCH_TIMEOUT_MS=3
- MAX_BATCH_SIZE=128
- MAX_QUEUE_SIZE=500

## Metrics Endpoint
The inference engine exposes runtime metrics on the `/metrics` endpoint

Example response:
```
{
  "queue_depth": 0,
  "total_requests": 10000,
  "total_batches": 228,
  "avg_batch_size": 43.8,
  "max_batch_size": 64,
  "batch_timeout_ms": 10
}
```

Tracked metrics include:

- Current load
- Total inference requests processed
- Total batches executed
- Average batch size
- Upper limit on requests per inference call
- Batch timeout in ms

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

## Project structure
```
fraud-detection-mlops/
│
├── app/                  # FastAPI inference service and dynamic batching engine
├── src/                  # Model training, preprocessing, feature engineering
├── dashboard/            # Streamlit monitoring dashboard
├── dags/                 # Airflow DAGs and orchestration pipelines
├── monitoring/           # Drift detection and retraining logic
├── streaming/            # Kafka transaction stream simulation
├── benchmarks/           # Load testing and inference benchmarking
├── tests/                # Unit and integration tests
├── k8s/                  # Kubernetes deployments, services, and HPA configs
├── data/                 # Raw and processed datasets
├── models/               # Trained model artifacts
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-airflow.txt
├── requirements-dev.txt
└── README.md
```

### Kubernetes Autoscaling

The FastAPI inference service was deployed to Kubernetes with a Horizontal Pod Autoscaler.

During load testing, CPU utilization exceeded the 60% target and Kubernetes scaled the deployment from 2 replicas to the maximum of 6 replicas.

| Metric | Value |
|---|---:|
| Min replicas | 2 |
| Max replicas | 6 |
| CPU target | 60% |
| Observed scale-up | 2 → 6 pods |

## Monitoring and Observability

The inference service exposes Prometheus metrics and Grafana dashboards for:

- Request throughput
- p50/p99 latency
- Queue depth
- Dynamic batch size
- Rejected requests
- Timed out requests

The dashboard demonstrates how the batching engine adapts under concurrent load while maintaining bounded latency and queue growth.

## Notes
The local Airflow setup uses PostgreSQL, Redis, and CeleryExecutor for production-style distributed orchestration.

A full cloud production deployment would add:

- Cloud object storage for MLflow artifacts, model files, monitoring reports, and logs
- Secrets manager for database passwords, API keys, and service credentials
- Container registry such as Docker Hub, GHCR, ECR, or GCR
- Managed Kafka such as Confluent Cloud or AWS MSK
- Managed Airflow such as MWAA, Astronomer, or Cloud Composer
- Cloud deployment on ECS, SageMaker, or Kubernetes
