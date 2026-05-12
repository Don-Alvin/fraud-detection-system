"""
FastAPI application for the API endpoints.
"""

import pickle
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging

from api.models import TransactionRequest, PredictionResponse, HealthCheckResponse, ModelInfoResponse
from api.config import (
    MODEL_PATH,
    ENCODERS_PATH,
    FEATURES_PATH,
    METADATA_PATH,
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    FRAUD_THRESHOLD
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Global variables for model and encoders
model = None
label_encoders = None
feature_names = None
metadata = None

@app.on_event("startup")
async def load_model():
    """
    Load the machine learning model, label encoders, feature names, and metadata at startup.
    """

    global model, label_encoders, feature_names, metadata
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")

        # Load label encoders
        with open(ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        logger.info("Label encoders loaded successfully.")

        # Load feature names
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        logger.info("Feature names loaded successfully.")

        # Load metadata
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            print(f"Model metadata: {metadata}")
        logger.info("Model metadata loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Error loading model.")
    
@app.get("/", tags=['General'])
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {
        "message": "Fraud detection API is running",
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model_info"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, tags=['General'])
async def health_check():
    """
    Health check endpoint to verify if the API is healthy and the model is loaded.
    """
    status = "healthy" if model is not None else "unhealthy"
    return HealthCheckResponse(
        status=status,
        model_loaded=model is not None,
        version=API_VERSION,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model_info", response_model=ModelInfoResponse, tags=['General'])
async def model_info():
    """
    Endpoint to get information about the loaded model.
    """
    if metadata is None:
        raise HTTPException(status_code=500, detail="Model metadata not available.")
    
    return ModelInfoResponse(
        model_name=metadata['model_name'],
        version=metadata['version'],
        date_created=metadata['training_date'],
        performance_metrics=metadata['metrics'],
        features_count=len(metadata['features']),
        training_data=metadata['training_data']
    )

@app.post("/predict", response_model=PredictionResponse, tags=['Prediction'])
async def predict(transaction: TransactionRequest):
    """
    Endpoint to make a fraud prediction based on transaction data.
    """
    if model is None:
        logger.warning("Prediction requested but model is not loaded using demo prediction mode.")
    
    try:
        # Convert request to dict
        transaction_dict = transaction.dict()
        
        
        # Simple preprocessing (minimal feature set for testing and demonstration)
        logger.warning("Using minimal feature set for prediction. This is for testing and demonstration purposes only.")

        # For demo we will just return a dummy prediction based on the transaction amount
        transaction_amount = transaction_dict['TransactionAmt']
        product_cd = transaction_dict['ProductCD']
        email = transaction_dict.get('P_emaildomain', 'gmail.com')

        # Dummy logic for prediction (for demonstration only)
        fraud_probability = min(1.0, transaction_amount / 1000) 

        # ProductCD-based adjustment, product c is riskier
        if product_cd == 'C':
            fraud_probability += 0.1
        
        # Email domain-based adjustment, certain domains are riskier (outlook.es is riskiest)
        if email == 'outlook.es':
            fraud_probability += 0.2
        
        fraud_probability = min(fraud_probability, 1.0)

        # Determine prediction
        is_fraud = fraud_probability >= FRAUD_THRESHOLD

        # Calculate confidence level
        confidence_level = abs(fraud_probability - 0.5) * 2

        # Determine risk level
        if fraud_probability >= 0.8:
            risk_level = "CRITICAL"
            recommendation = "reject"
        elif fraud_probability >= 0.6:
            risk_level = "HIGH"
            recommendation = "review - Manual review recommended due to high fraud probability."
        elif fraud_probability >= 0.5:
            risk_level = "MEDIUM"
            recommendation = "review - Additional verification required."
        else:
            risk_level = "LOW"
            recommendation = "approve"
        
        # Log the prediction details
        logger.info(f"Transaction amount: {transaction_amount}, ProductCD: {product_cd}, Email: {email}, Fraud probability: {fraud_probability:.4f}, Risk level: {risk_level}, Recommendation: {recommendation}")

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            confidence_level=round(confidence_level, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
