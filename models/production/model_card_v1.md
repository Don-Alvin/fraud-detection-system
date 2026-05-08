## Model Card - Fraud Detection System v1.0

### Model Details
- **Name:** Fraud Detection XGBoost v1.0
- **Type:** XGboost Gradient Boosting Classifier
- **Purpose:** Real time e-commerce transactions fraud detection
- **Version:** 1.0
- **Date created:**: 8th May 2026

### Training Data
- **Source:** IEEE-CIS Frau Detection Dataset (Kaggle)
- **Provider:** Vesta Corporation

### Performance Metrics
| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.951 |
| **Recall** | 0.849 |
|**Precision** | 24% |
|**F1-Score** | 0.34 |

### Model Architecture
- **Algorithm:** XGBoost (Gradient Boosted Decision Trees)
- **Trees:** 100 estimators
- **Max Depth:** 6
- **Learning Rate:** 0.1
- **Class Imbalance:** Handled via scale_pos_weight (27.58)

### Contact
- **Developer:** Don Alvin
- **Email:** alvindon41@gmail.com
- **GitHub:** https://github.com/Don-Alvin/fraud-detection-system


**NB:** This is the first model after performing initial feature engineering. We will work on improving the model as we continue to perform further feature engineering.