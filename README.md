# Fraud Detection System
> An end to end machine learning system for detecting cfedit card fraud in real-time

## Live Demo

| Service | Link | Status |
|---------|------|--------|
| **Streamlit Dashboard** | [fraud-detection-dashboard.streamlit.app](https://fraud-detection-system-pe2cwyxqfjjtqth4ub8muq.streamlit.app/) | Live |
| **API Documentation** | [fraud-detection-api.onrender.com/docs](https://fraud-detection-system-1xqf.onrender.com/docs) | Live |

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

## Key Business Insights from EDA

### 1. Missing Data = Fraud Signal

| Feature | Fraud Rate (Present) | Fraud Rate (Missing) | Difference |
|---------|---------------------|---------------------|-------------|
| `addr1` (address) | 2.46% | 11.78% | **+9.32%** |
| `addr2` (address) | 2.46% | 11.78% | **+9.32%** |
| `R_emaildomain` | 2.08% | 8.18% | **+6.10%** |

**Insight:** Fraudsters often skip optional fields. Missing data isn't random - it's a behavioral signal.

### 2. The "Witching Hour"


**Insight:** Night transactions (12-6 AM) are **22% riskier** than daytime.

### 3. Product C is a Fraud Magnet

| Product | Fraud Rate | Risk vs Baseline |
|---------|------------|------------------|
| C | 11.69% | **5.7x higher** |
| S | 5.90% | 1.7x higher |
| H | 4.77% | 1.4x higher |
| R | 3.78% | 1.1x higher |
| W | 2.04% | Baseline (safest) |

**Insight:** Product C (likely digital goods like gift cards) is fraudsters' top target.

### 4. Velocity = Suspicious Activity

| Pattern | Fraud Rate | vs Baseline |
|---------|------------|-------------|
| Single transaction | 3.5% | Baseline |
| 2-5 transactions in 1 hour | 4.4% | **+26%** |
| <5 minutes since last transaction | 5.66% | **+62%** |

**Insight:** Fraudsters test stolen cards with rapid, small transactions.

---

## 🛠️ Feature Engineering

**Total Features: 524 (434 original + 90 engineered)**

| Feature Group | Count | Key Examples | Impact |
|---------------|-------|--------------|--------|
| **Missing Value Flags** | 44 | `addr1_is_missing`, `M6_is_missing` | +2.1% AUC |
| **Temporal Features** | 6 | `transaction_hour`, `is_night`, `time_period` | +1.8% AUC |
| **Velocity Features** | 10 | `txn_count_1h`, `time_since_last_txn_hrs` | +1.9% AUC |
| **Card Aggregations** | 8 | `card_amt_mean`, `card_amt_std`, `card_txn_count` | +1.0% AUC |
| **Product Aggregations** | 6 | `product_fraud_rate`, `product_amt_mean` | +0.5% AUC |
| **Email Domain Features** | 4 | `P_email_fraud_rate`, `R_email_fraud_rate` | +0.5% AUC |
| **Address Features** | 4 | `addr1_fraud_rate`, `addr2_fraud_rate` | +0.5% AUC |
| **Risk Indicators** | 5 | `is_moderate_velocity`, `is_rapid_succession` | +0.5% AUC |

### Feature Engineering Philosophy

1. **Preserve missing data as signal** - Created binary flags for features with 25-75% missing
2. **Capture behavior patterns** - Velocity, time, and frequency features
3. **Establish baselines** - Aggregations provide "normal" for each entity
4. **Combine weak signals** - Risk flags aggregate multiple indicators

---

## Model Performance

### Model Configuration (XGBoost)

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'scale_pos_weight': 27.58,  # Handles class imbalance
    'tree_method': 'hist'        # Faster training
}
```

### Performance Metrics

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| ROC-AUC | 0.929 | 0.951 | ▲ **+2.2%** |
| Recall | 80.3% | 84.9% | ▲ **+4.6%** |
| Precision | 22.7% | 24.2% | ▲ **+1.5%** |
| F1-Score | 0.354 | 0.377 | ▲ **+6.5%** |
| False Positive Rate | ~2% | 1.04% | ▼ **-48%** |

## Tech Stack
**Machine Learning**
- XGBoost

**Deployment**
- FastApi
- Streamlit
- Docker
- Github Actions