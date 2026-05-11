from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class TransactionRequest(BaseModel):
    """
    Model for incoming transaction requests.
    """

    TransactionAmt: float = Field(..., description="The amount of the transaction.", ge=0, example=123.45)
    ProductCD: str = Field(..., description="Product code of the transaction.", example="W")

    # Card features
    card1: Optional[float] = Field(None, description="First card feature (e.g., card type).", example=1234)
    card2: Optional[float] = Field(None, description="Second card feature (e.g., card category).", example=5678)
    card3: Optional[float] = Field(None, description="Third card feature (e.g., card issuer).", example=910)
    card4: Optional[str] = Field('visa', description="Fourth card feature (e.g., card brand).", example="visa")
    card5: Optional[float] = Field(None, description="Fifth card feature (e.g., card country).", example=123)
    card6: Optional[str] = Field('debit', description="Card category", example="debit")

    # Address features
    addr1: Optional[float] = Field(None, description="First address feature", example=123)
    addr2: Optional[float] = Field(None, description="Second address feature", example=456)

    # Email domain features
    P_emaildomain: Optional[str] = Field('gmail.com', description="Payer email domain", example="gmail.com")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain", example="yahoo.com")

    # Distance features
    dist1: Optional[float] = Field(None, description="Distance feature 1")
    dist2: Optional[float] = Field(None, description="Distance feature 2")

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 123.45,
                "ProductCD": "W",
                "card1": 1234,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
            }
        }
    
class PredictionResponse(BaseModel):
    """
    Model for prediction responses.
    """
    is_fraud: bool = Field(..., description="Fraud prediction result (True if fraudulent, False otherwise).")
    fraud_probability: float = Field(..., description="Probability of the transaction being fraudulent.", ge=0, le=1)
    confidence_level: float = Field(..., description="Prediction confidence (0 - low, 1 - high).", ge=0, le=1)
    risk_level: str = Field(..., description="Risk level based on the fraud probability (e.g., 'LOW', 'MEDIUM', 'HIGH').")
    recommendation: str = Field(..., description="Action recommendation based on the prediction (e.g., 'approve', 'review', 'reject').")
    timestamp: datetime = Field(..., description="Timestamp of the prediction.")

    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": True,
                "fraud_probability": 0.85,
                "confidence_level": 0.9,
                "risk_level": "HIGH",
                "recommendation": "review",
                "timestamp": "2026-06-01T12:00:00Z"
            }
        }

class HealthCheckResponse(BaseModel):
    """
    Model for health check responses.
    """

    status: str = Field(..., description="Health status of the API (e.g., 'healthy', 'unhealthy').")
    model_loaded: bool = Field(..., description="Indicates if the model is loaded and ready for predictions.")
    version: str = Field(..., description="Version of the API.")
    timestamp: datetime = Field(..., description="Timestamp of the health check.")

class ModelInfoResponse(BaseModel):
    """
    Model for model information responses.
    """

    model_name: str = Field(..., description="Name of the loaded model.")
    version: str = Field(..., description="Version of the loaded model.")
    date_created: datetime = Field(..., description="Date when the model was created.")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics of the model (e.g., accuracy, precision, recall).")
    features_count: int = Field(..., description="Number of features used by the model.")
    training_samples: int = Field(..., description="Number of samples used to train the model.")

