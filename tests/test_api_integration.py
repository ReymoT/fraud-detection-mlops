from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

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


def assert_prediction_response(data):
    assert "fraud_probability" in data
    assert "threshold" in data
    assert "risk_level" in data
    assert "flag" in data

    assert isinstance(data["fraud_probability"], float)
    assert data["flag"] in [0, 1]


def test_predict_endpoint_returns_expected_fields():
    response = client.post("/predict", json = payload)

    assert response.status_code == 200

    data = response.json()
    assert_prediction_response(data)


def test_predict_direct_endpoint_returns_expected_fields():
    response = client.post("/predict_direct", json = payload)

    assert response.status_code == 200

    data = response.json()
    assert_prediction_response(data)


def test_batched_predict_endpoint_returns_expected_fields():
    response = client.post("/predict", json = payload)

    assert response.status_code == 200

    data = response.json()
    assert_prediction_response(data)


def test_inference_metrics_endpoint():
    response = client.get("/metrics/inference")

    assert response.status_code == 200

    data = response.json()

    assert "queue_depth" in data
    assert "total_requests" in data
    assert "total_batches" in data
    assert "avg_batch_size" in data
    assert "max_batch_size" in data
    assert "batch_timeout_ms" in data

def test_predict_direct_endpoint_with_explanations():
    response = client.post("/predict_direct?explain=true", json = payload)

    assert response.status_code == 200

    data = response.json()
    assert_prediction_response(data)

def test_batched_predict_endpoint_with_explanations():
    response = client.post("/predict?explain=true", json = payload)

    assert response.status_code == 200

    data = response.json()
    assert_prediction_response(data)