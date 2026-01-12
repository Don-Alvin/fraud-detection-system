# Fraud Detection System
> An end to end machine learning system for detecting cfedit card fraud in real-time

## Introduction
- Credit card fraud is a big threat to institution across the globe. According to reports, for every dollar lost to fraud, institutins spend another two or three dollars on related costs.
- Machine learning and artificial interlligence can transform the way we detect and handle fraud. With these technologies we can identify patterns and learn form each transaction eventually becoming more accurate and catch fraud.
- In this project I intend to leverage machine learning tehniques to detect fraud in over 590k transactions. The dataset used in this project is provided by Vesta Corporation and features real world data.

## Project Overview
In this project I build a production-ready fraud detection system that:
 - Processes 590k+ transaction with 434 features.
 - Handles extreme class imbalance
 - Provides real-time predictions
 - Features a monitoring dashboard

## Project Structure

fraud-detection-system/
- ├── data/                   # Data storage
- ├── research/               # Jupyter notebooks for exploration
- ├── src/                    # Production code
- ├── api/                    # FastAPI application
- ├── dashboard/              # Streamlit monitoring dashboard
- ├── tests/                  # Unit and integration tests
- └── deployment/             # Docker and deployment configs

## Dataset
**Source**: [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)

### Data Characteristics
 - Training transactions (`train_transactions.csv`): 590,540
 - Test transactions(`test_transactions.csv`):506,691
 - Features: 434 (Got from merging transactions and identity data)
 - Fraud rate: 3.5%
 - Time period: ~6 months of data

## Tech Stack
**Machine Learning**
- XGBoost
- LightGBM
- TensorFlow/PyTorch
- Imbalanced-learn

**Deployment**
- FastApi
- Looker Studio
- Docker
- Github Actions