import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Model files
MODEL_PATH = MODELS_DIR / "fraud_detector_v1.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders_v1.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names_v1.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata_v1.json"

# API settings
API_TITLE = 'Fraud Detection API'
API_VERSION = '1.0'
API_DESCRIPTION = 'An API for detecting fraudulent transactions using a pre-trained machine learning model.'    

# Prediction settings
FRAUD_THRESHOLD = 0.5  # Threshold for classifying a transaction as fraudulent
HIGH_CONFIDENCE_THRESHOLD = 0.8  # Threshold for high confidence predictions

# Feature defaults (if any)
FEATURE_DEFAULTS = {
    'transaction_amount': 100.0,
    'ProductCD': 'W',
    'card4': 'visa',
    'card6': 'debit',
    'P_emaildomain': 'gmail.com'
}