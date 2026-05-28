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
from api.feature_engineering import reconstruct_features, load_lookup_tables, card_lookup, product_lookup, global_stats
import api.feature_engineering as fe

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
    global model, label_encoders, feature_names, metadata
    """
    Load the machine learning model, label encoders, feature names, and metadata at startup.
    """

    global model, label_encoders, feature_names, metadata
    try:
        # Load lookup tables
        lookups_loaded = load_lookup_tables()
        if lookups_loaded:
            logger.info('Lookup tables loaded successfully.')
        else:
            logger.warning("Lookup tables failed to load")
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
        training_data=metadata.get('training_samples', metadata.get('training_data', 590540))
    )

@app.post("/predict", response_model=PredictionResponse, tags=['Prediction'])
async def predict(transaction: TransactionRequest):
    """
    Endpoint to make a fraud prediction based on transaction data.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        logger.info(f"Prediction request recieved: {transaction.dict()}")
        logger.info(f"Lookup tables status - card: {card_lookup is not None}",
                    f"product: {product_lookup is not None}, "
                    f"global_stats: {global_stats is not None}")
        transaction_dict = transaction.dict()
        features_df = reconstruct_features(transaction_dict)

        # Predict using model
        prediction_result = model.predict_proba(features_df)
        fraud_probability = float(prediction_result[0, 1])
        is_fraud = fraud_probability >= FRAUD_THRESHOLD
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
        

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_probability, 4),
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
    
