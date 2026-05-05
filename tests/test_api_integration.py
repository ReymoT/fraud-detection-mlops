from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint_returns_expected_fields():
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
        "dob": "2003-01-01"
    }

    response = client.post("/predict", json = payload)

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "threshold" in data
    assert "risk_level" in data
    assert "flag" in data
    assert "top_reasons" in data

    assert isinstance(data["fraud_probability"], float)
    assert data["flag"] in [0, 1]