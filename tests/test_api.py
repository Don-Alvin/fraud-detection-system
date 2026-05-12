from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_predict_valid_transaction():
    payload = {
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "is_fraud" in response.json()
    assert "fraud_probability" in response.json()

def test_predict_high_risk_transaction():
    payload = {
        "TransactionAmt": 1200.0,
        "ProductCD": "C",
        "card4": "amex",
        "card6": "credit",
        "P_emaildomain": "yahoo.com"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["fraud_probability"] > 0.5