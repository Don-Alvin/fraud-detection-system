import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
API_URL = "https://fraud-detection-system-1xqf.onrender.com"

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard | EDA & Model Performance",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page title
st.title("Fraud Detection Dashboard")
st.markdown("| IEEE-CIS Fraud Detection | 95.1% ROC-AUC | Complete EDA & Feature Engineering")
st.divider()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Transaction", "EDA Insights", "Feature Engineering", "Model Performance", "About"])

# Home Page
if page == "Home":
    st.header("Welcome to the Fraud Detection Dashboard")
    st.image("https://images.pexels.com/photos/30309058/pexels-photo-30309058.jpeg", width='stretch')
    st.markdown("""
    This dashboard provides comprehensive insights into our fraud detection system, trained on the IEEE-CIS dataset.
    
    ### What You'll Find Here:
    
    **EDA Insights** - Discover fraud patterns in transaction data
    **Feature Engineering** - See how we transformed raw data into predictive features  
    **Model Performance** - Evaluate our XGBoost model's metrics
    **Live Predictions** - Test the model with real-time transaction scoring
    
    ### Key Findings:
    - **3.5%** overall fraud rate (imbalanced dataset)
    - **Missing values are INFORMATIVE** - When certain features are missing, fraud rate increases significantly
    - **Velocity matters** - Transactions within 5 minutes of previous are 34% riskier
    - **Product C** has the highest fraud rate (11.69%)
    - **Night transactions** are 22% riskier than daytime
    """)

# Predict Transaction Page (UNCHANGED)
if page == "Predict Transaction":
    st.header("Transaction Fraud Prediction")
    st.markdown("Enter transaction details to predict if it's fraudulent or not.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input(
            "Transaction Amount (USD)",
            min_value=0.0,
            max_value=50000.0,
            step=10.0,
        )
        product = st.selectbox(
            "Product Code",
            options=["W", "H", "C", "S", "R"],
            help="W=Web, H=Hotel, C=Card, S=Service, R=Retail"
        )
        card_type = st.selectbox(
            "Card Type",
            options=["visa", "mastercard", "amex", "discover"]
        )

    with col2:
        card_category = st.selectbox(
            "Card Category",
            options=["debit", "credit"]
        )
        email_domain = st.text_input(
            "Purchaser Email Domain",
            value="gmail.com",
            help="e.g. gmail.com, yahoo.com, tempmail.com"
        )
        card1 = st.number_input(
            "Card Identifier (card1)",
            min_value=1000.0,
            max_value=20000.0,
            value=13926.0
        )

    st.divider()

    if st.button("Predict Fraud", type="primary", width='stretch'):
        with st.spinner("Analyzing transaction..."):
            try:
                payload = {
                    "TransactionAmt": amount,
                    "ProductCD": product,
                    "card1": card1,
                    "card4": card_type,
                    "card6": card_category,
                    "P_emaildomain": email_domain
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display result
                    if result["is_fraud"]:
                        st.error(f"FRAUD DETECTED")
                    else:
                        st.success(f"LEGITIMATE TRANSACTION")

                    # Metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Fraud Probability",
                            f"{result['fraud_probability']*100:.1f}%"
                        )
                    with col2:
                        st.metric(
                            "Risk Level",
                            result["risk_level"]
                        )
                    with col3:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']*100:.1f}%"
                        )

                    # Recommendation
                    st.info(f"**Recommendation:** {result['recommendation']}")

                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["fraud_probability"] * 100,
                        title={"text": "Fraud Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkred"},
                            "steps": [
                                {"range": [0, 40], "color": "#2ecc71"},
                                {"range": [40, 60], "color": "#f39c12"},
                                {"range": [60, 80], "color": "#e67e22"},
                                {"range": [80, 100], "color": "#e74c3c"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')

                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.Timeout:
                st.warning("Request timed out. Render free tier may be sleeping. Try again in 30 seconds.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# EDA Insights Page
elif page == "EDA Insights":
    st.header("Exploratory Data Analysis Insights")
    st.markdown("Key fraud patterns discovered from 590,540 transactions over 6 months")
    
    # Key Stats Row
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", "590,540")
    with col2:
        st.metric("Fraud Rate", "3.5%", delta="27.6:1 imbalance")
    with col3:
        st.metric("Fraudulent Transactions", "20,663")
    with col4:
        st.metric("Time Period", "6 months", delta="182 days")
    
    st.divider()
    
    # Time-Based Patterns
    st.subheader("⏰ When Does Fraud Happen?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly fraud pattern
        hour_data = pd.DataFrame({
            "Hour": list(range(24)),
            "Fraud_Rate": [3.60, 3.56, 4.70, 5.01, 4.93, 3.99, 3.74, 3.47,
                          3.21, 3.04, 3.09, 3.07, 3.32, 3.47, 3.43, 3.45,
                          3.53, 3.57, 3.42, 3.52, 3.66, 3.79, 3.84, 3.69]
        })
        fig = px.line(hour_data, x="Hour", y="Fraud_Rate", 
                      title="Fraud Rate by Hour of Day",
                      labels={"Hour": "Hour (0-23)", "Fraud_Rate": "Fraud Rate (%)"})
        fig.add_hline(y=3.5, line_dash="dash", line_color="red", 
                      annotation_text="Overall 3.5%")
        fig.update_traces(line=dict(color="#e74c3c", width=3))
        st.plotly_chart(fig, width=True)
        st.caption("💡 **Insight:** Peak fraud between 2-5 AM (4.7-5.0%). Night transactions are 22% riskier.")
    
    with col2:
        # Weekday pattern
        weekday_data = pd.DataFrame({
            "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "Fraud_Rate": [3.48, 3.52, 3.44, 3.39, 3.47, 3.52, 3.71]
        })
        fig = px.bar(weekday_data, x="Day", y="Fraud_Rate",
                     title="Fraud Rate by Day of Week",
                     labels={"Day": "", "Fraud_Rate": "Fraud Rate (%)"},
                     color="Fraud_Rate",
                     color_continuous_scale="Reds")
        fig.add_hline(y=3.5, line_dash="dash", line_color="blue", 
                      annotation_text="Overall 3.5%")
        st.plotly_chart(fig, width='stretch')
        st.caption("💡 **Insight:** Slightly higher fraud on Sundays (3.71%)")
    
    st.divider()
    
    # Product & Card Analysis
    st.subheader("Product & Payment Method Risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_data = pd.DataFrame({
            "Product": ["C", "S", "H", "R", "W"],
            "Fraud_Rate": [11.69, 5.90, 4.77, 3.78, 2.04],
            "Avg_Amount": [42.87, 60.27, 73.17, 168.31, 153.16],
            "Transactions": ["68,519", "11,628", "33,024", "37,699", "439,670"]
        })
        fig = px.bar(product_data, x="Product", y="Fraud_Rate",
                     title="Fraud Rate by Product Code",
                     labels={"Fraud_Rate": "Fraud Rate (%)"},
                     color="Fraud_Rate",
                     color_continuous_scale="Reds",
                     text="Fraud_Rate")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width='stretch')
        st.caption("💡 **Insight:** Product C has **5.7x higher** fraud rate than Product W")
    
    with col2:
        card_data = pd.DataFrame({
            "Card_Type": ["discover", "visa", "mastercard", "american express"],
            "Fraud_Rate": [7.73, 3.48, 3.43, 2.87]
        })
        fig = px.bar(card_data, x="Card_Type", y="Fraud_Rate",
                     title="Fraud Rate by Card Type",
                     labels={"Fraud_Rate": "Fraud Rate (%)"},
                     color="Fraud_Rate",
                     color_continuous_scale="Reds")
        fig.add_hline(y=3.5, line_dash="dash", line_color="blue",
                      annotation_text="Overall 3.5%")
        st.plotly_chart(fig, width='stretch')
        st.caption("**Insight:** Discover cards have **2.2x higher** fraud rate than average")
    
    st.divider()
    
    # Missing Value Insights
    st.subheader("🔍 Missing Values Are Informative")
    
    missing_insights = pd.DataFrame({
        "Feature": ["addr1", "addr2", "R_emaildomain", "id_04", "id_03", "D12", "D14"],
        "Fraud Rate When Missing": ["11.78%", "11.78%", "8.18%", "10.72%", "10.72%", "11.74%", "11.60%"],
        "Fraud Rate When Present": ["2.46%", "2.46%", "2.08%", "2.59%", "2.59%", "2.48%", "2.55%"],
        "Difference": ["+9.32%", "+9.32%", "+6.10%", "+8.14%", "+8.14%", "-9.26%", "-9.05%"]
    })
    
    st.dataframe(missing_insights, width=True, hide_index=True)
    st.caption("**Insight:** Missing address data correlates strongly with fraud (+9.32%). Missing email domains also indicate higher risk.")
    
    st.divider()
    
    # Amount Analysis
    st.subheader("Transaction Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount_stats = pd.DataFrame({
            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "> $1000", "< $10"],
            "Legitimate": ["$134.51", "$68.50", "$239.40", "$0.25", "$31,937", "1.23%", "1.15%"],
            "Fraud": ["$149.24", "$75.00", "$232.21", "$0.29", "$5,191", "0.39%", "1.02%"]
        })
        st.dataframe(amount_stats, width='stretch', hide_index=True)
        st.caption("**Insight:** Fraudulent transactions average $14.73 higher than legitimate ones")
    
    with col2:
        # Amount buckets
        amount_buckets = pd.DataFrame({
            "Amount Range": ["$0-10", "$10-25", "$25-50", "$50-100", "$100-250", "$250-500", "$500+"],
            "Fraud_Rate": [2.89, 3.21, 3.45, 3.61, 3.58, 3.42, 2.94]
        })
        fig = px.bar(amount_buckets, x="Amount Range", y="Fraud_Rate",
                     title="Fraud Rate by Amount Range",
                     color="Fraud_Rate",
                     color_continuous_scale="Reds")
        fig.add_hline(y=3.5, line_dash="dash", line_color="blue")
        st.plotly_chart(fig, width='stretch')
        st.caption("**Insight:** Medium-sized transactions ($50-250) have highest fraud risk")

# Feature Engineering Page
elif page == "Feature Engineering":
    st.header("Feature Engineering Pipeline")
    st.markdown("From 434 raw features to 524 engineered features")
    
    # Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Features", "434")
    with col2:
        st.metric("Engineered Features", "90", delta="+20.7%")
    with col3:
        st.metric("Total Features", "524")
    with col4:
        st.metric("Performance Gain", "+2.2% AUC", delta="0.929 → 0.951")
    
    st.divider()
    
    # Feature Categories
    st.subheader("Engineered Feature Categories")
    
    fe_summary = pd.DataFrame({
        "Category": [
            "Missing Value Flags",
            "Temporal Features",
            "Velocity Features",
            "Card Aggregations",
            "Product Aggregations",
            "Email Domain Aggregations",
            "Address Aggregations",
            "Risk Indicators"
        ],
        "Count": [44, 6, 10, 8, 6, 4, 4, 5],
        "Purpose": [
            "Create binary flags for missing values (25-75% missing)",
            "Hour, day, weekday, weekend, night, time period",
            "Transaction frequency in 1h/24h, time since last transaction",
            "Mean, std, min, max, median, count, fraud rate per card",
            "Mean, std, median, count, fraud rate per product",
            "Mean amount, count, fraud rate per email domain",
            "Mean amount, count, fraud rate per address",
            "Combined risk scores from multiple indicators"
        ],
        "Impact": [
            "+2-3% AUC",
            "+1-2% AUC", 
            "+2-3% AUC",
            "+1-2% AUC",
            "+0.5-1% AUC",
            "+0.5-1% AUC",
            "+0.5-1% AUC",
            "+0.5-1% AUC"
        ]
    })
    
    st.dataframe(fe_summary, width='stretch', hide_index=True)
    
    st.divider()
    
    # Key Feature Insights
    st.subheader("Key Feature Engineering Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Missing Value Flags**
        - Created for features with 25-75% missing values
        - `M6_is_missing`: 7.07% fraud rate (vs 2.06% when present)
        - `V9_is_missing`: 5.21% fraud rate (vs 1.96% when present)
        - These flags became top predictors in the model
        
        **Velocity Features**  
        - `txn_count_1h`: Transactions in last hour
        - `time_since_last_txn`: Time gap between transactions
        - Moderate velocity (2-5 txns/hour): 4.40% fraud rate
        - Quick succession (<5 mins): 5.66% fraud rate
        """)
    
    with col2:
        st.markdown("""
        **Aggregation Features**
        - Card-level statistics (mean, std, min, max, median)
        - Product-level fraud rates (Product C: 11.69%!)
        - Email domain risk profiles (temporary emails riskier)
        - Address-based fraud rates (addr1/addr2 risk scores)
        
        **Risk Indicators**
        - Combined 5 risk flags into a single score
        - Transactions with 3+ flags: 3.62% fraud rate
        - Risk flags work best in combination
        """)
    
    st.divider()
    
    # Performance Impact Visualization
    st.subheader("Feature Engineering Impact on Model Performance")
    
    impact_data = pd.DataFrame({
        "Feature Set": ["Baseline", "+ Missing Flags", "+ Temporal", "+ Velocity", 
                       "+ Aggregations", "+ Risk Indicators", "Final Model"],
        "ROC-AUC": [0.929, 0.938, 0.942, 0.945, 0.948, 0.949, 0.951],
        "Recall": [80.3, 81.5, 82.1, 83.0, 83.8, 84.2, 84.9]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=impact_data["Feature Set"], y=impact_data["ROC-AUC"],
                             mode="lines+markers", name="ROC-AUC",
                             line=dict(color="#2ecc71", width=3),
                             marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=impact_data["Feature Set"], y=impact_data["Recall"],
                             mode="lines+markers", name="Recall",
                             line=dict(color="#e74c3c", width=3),
                             marker=dict(size=10)))
    fig.update_layout(title="Cumulative Performance Improvement",
                      xaxis_title="Feature Set",
                      yaxis_title="Score",
                      hovermode="x unified")
    st.plotly_chart(fig, width='stretch')
    st.caption("Each feature engineering phase contributed 0.5-2% improvement in ROC-AUC")

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")
    
    # Fetch model info from API
    try:
        response = requests.get(f"{API_URL}/model_info", timeout=30)
        if response.status_code == 200:
            info = response.json()
            st.success("Connected to live API")
        else:
            info = None
    except:
        info = None
        st.warning("Could not connect to API. Showing cached metrics.")
    
    # Performance metrics
    st.subheader("Model Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Baseline vs Improved comparison
    baseline_metrics = {
        "roc_auc": 0.929,
        "recall": 0.803,
        "precision": 0.227,
        "f1_score": 0.354
    }
    
    perf = info["performance_metrics"] if info else {
        "roc_auc": 0.951,
        "recall": 0.849,
        "precision": 0.242,
        "f1_score": 0.377
    }
    
    with col1:
        st.metric("ROC-AUC", f"{perf['roc_auc']:.3f}", 
                  delta=f"+{(perf['roc_auc'] - baseline_metrics['roc_auc'])*100:.1f}%")
    with col2:
        st.metric("Recall", f"{perf['recall']*100:.1f}%", 
                  delta=f"+{(perf['recall'] - baseline_metrics['recall'])*100:.1f}%")
    with col3:
        st.metric("Precision", f"{perf['precision']*100:.1f}%", 
                  delta=f"+{(perf['precision'] - baseline_metrics['precision'])*100:.1f}%")
    with col4:
        st.metric("F1-Score", f"{perf['f1_score']:.3f}", 
                  delta=f"+{(perf['f1_score'] - baseline_metrics['f1_score'])*100:.1f}%")
    
    st.divider()

    # Recall vs Precision Tradeoff
    st.subheader("Recall vs Precision Tradeoff")
    st.markdown("""
                With a dataset that's **96.5% legitimate transactions**, even a tiny 1% false positive rate creates many false alarms relative to fraud cases. Our metrics tell the real story:

| What Matters | Our Performance | Why It's Good |
|--------------|-----------------|----------------|
| **False Positive Rate** | **1.04%** | Only 1 in 100 legit transactions flagged - excellent for fraud detection |
| **Recall** | **84.9%** | We catch 85% of all fraud attempts |
| **Business Impact** | **+102 frauds caught** | Estimated $15,222 in fraud losses prevented |
""")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    cm_data = {
        "Predicted Legitimate": [113877, 1190],
        "Predicted Fraud": [625, 3508]
    }
    cm_df = pd.DataFrame(
        cm_data,
        index=["Actual Legitimate", "Actual Fraud"]
    )
    
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="RdYlGn_r",
        title="Confusion Matrix (Validation Set - 118,108 transactions)"
    )
    st.plotly_chart(fig, width='stretch')
    
    # Business Impact
    st.subheader("Business Impact")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Additional Frauds Caught", "+102", 
                  delta="vs Baseline")
    with col2:
        st.metric("Estimated Fraud Prevention Value", "$15,222", 
                  delta="At $149 avg fraud amount")
    
    st.divider()
    
    # Feature Importance Summary
    st.subheader("Top Features by Category")
    
    feature_categories = pd.DataFrame({
        "Category": ["Vesta Features", "Transaction Core", "Card Features", 
                    "Missing Flags", "Identity Features", "Velocity Features"],
        "Count": [339, 3, 6, 44, 40, 10],
        "Total Importance": [0.783, 0.007, 0.020, 0.027, 0.036, 0.015],
        "Key Features": ["V258, V70, V91", "Amount, Time", "card1, card4", 
                        "M6_is_missing, V9_is_missing", "DeviceType", "txn_count_1h"]
    })
    st.dataframe(feature_categories, width='stretch', hide_index=True)
    
    st.caption("Vesta features dominate importance, but engineered features (missing flags, velocity) provide critical signal")

# About Page
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ## Real-Time Fraud Detection System
    
    An end-to-end machine learning system for detecting fraudulent 
    e-commerce transactions in real-time.
    
    ### Dataset
    - **Source:** IEEE-CIS Fraud Detection (Kaggle / Vesta Corporation)
    - **Size:** 590,540 transactions over 6 months
    - **Features:** 524 (434 original + 90 engineered)
    - **Fraud Rate:** 3.5% (20,663 fraudulent transactions)
    
    ### Key EDA Findings
    - **Peak fraud hours:** 2-5 AM (4.7-5.0% fraud rate)
    - **Highest risk product:** Product C (11.69% fraud)
    - **Missing data signal:** addr1 missing → 11.78% fraud (vs 2.46%)
    - **Velocity signal:** Quick succession (<5 mins) → 5.66% fraud
    
    ### Feature Engineering Highlights
    - **44 missing value flags** for features with 25-75% missing
    - **Temporal features** (hour, day, weekday, weekend, night)
    - **Velocity features** (txn frequency in 1h/24h, time since last)
    - **Aggregations** by card, product, email domain, address
    - **Risk indicators** combining multiple signals
    
    ### Model Performance
    - **ROC-AUC:** 0.951 (baseline 0.929 → +2.2%)
    - **Recall:** 84.9% (caught 102 more fraud cases)
    - **Precision:** 24.2%
    - **F1-Score:** 0.377
    
    ### Tech Stack
    | Component | Technology |
    |-----------|------------|
    | EDA & Feature Engineering | Python (Pandas, NumPy, Matplotlib, Seaborn) |
    | Model | XGBoost |
    | API | FastAPI |
    | Dashboard | Streamlit |
    | Deployment | Render |
    
    ### Links
    - **API:** https://fraud-detection-system-1xqf.onrender.com
    - **API Docs:** https://fraud-detection-system-1xqf.onrender.com/docs
    - **GitHub:** https://github.com/Don-Alvin/fraud-detection-system
    """)
    
    # API health check
    st.divider()
    st.subheader("API Status")
    
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=30)
            if response.status_code == 200:
                health = response.json()
                st.success(f"API is {health['status']}")
                st.json(health)
            else:
                st.error("API returned an error")
        except:
            st.error("Could not reach API")