# Fraud Detection API

REST API for real-time fraud detection using XGBoost.

## Live Endpoint

**Base URL:** https://fraud-detection-system-1xqf.onrender.com

**Interactive Docs:** https://fraud-detection-system-1xqf.onrender.com/docs

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run API

```bash
# Development
uvicorn api.main:app --reload

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### POST /predict
Predict fraud for a transaction.

**Request:**
```json
{
  "TransactionAmt": 250.50,
  "ProductCD": "C",
  "card4": "visa",
  "P_emaildomain": "outlook.es"
}
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.5505,
  "confidence_level": 0.101,
  "risk_level": "MEDIUM",
  "recommendation": "review - Additional verification required.",
  "timestamp": "2026-05-11T12:06:52.271644"
}
```

### GET /health
Health check.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0",
  "timestamp": "2026-05-11T12:10:51.172393"
}
```

### GET /model-info
Model metadata and performance.

## Performance

- **ROC-AUC** 0.951
- **Recall** 0.849
- **Precision** 24% 

## License

MIT