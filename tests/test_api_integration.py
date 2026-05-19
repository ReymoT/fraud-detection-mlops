import pytest
from fastapi.testclient import TestClient
from app.main import app

payload = {
    "amt": 5000,
    "lat": 40.7128,
    "long": -74.0060,
    "merch_lat": 34.0522,
    "merch_long": -118.2437,
    "city_pop": 1000,
    "category": "shopping_net",
    "gender": "M",
    "state": "CA",
    "merchant": "unknown_rare_store",
    "trans_date_trans_time": "2020-12-25 02:30:00",
    "dob": "2003-01-01",
}

API_KEY = os.getenv("API_KEY", "dev-secret-key")
HEADERS = {"x-api-key": API_KEY}

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def assert_prediction_response(data):
    assert "fraud_probability" in data
    assert "threshold" in data
    assert "risk_level" in data
    assert "flag" in data

    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["threshold"], float)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["flag"] in [0, 1]


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_direct_endpoint_returns_expected_fields(client):
    response = client.post("/predict_direct", json = payload, headers = HEADERS)

    assert response.status_code == 200

    data = response.json()
    assert_prediction_response(data)


def test_predict_direct_endpoint_with_explanations(client):
    response = client.post("/predict_direct?explain=true", json = payload, headers = HEADERS)

    assert response.status_code == 200

    data = response.json()
    assert_prediction_response(data)

    assert "top_reasons" in data
    assert isinstance(data["top_reasons"], list)
    assert len(data["top_reasons"]) > 0
    assert "feature" in data["top_reasons"][0]
    assert "impact" in data["top_reasons"][0]


def test_inference_metrics_endpoint(client):
    response = client.get("/metrics/inference")

    assert response.status_code == 200

    data = response.json()

    assert "queue_depth" in data
    assert "total_requests" in data
    assert "total_batches" in data
    assert "avg_batch_size" in data
    assert "max_batch_size" in data
    assert "batch_timeout_ms" in data