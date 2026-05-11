import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_URL = "https://fraud-detection-system-1xqf.onrender.com/docs"

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page title
st.title("Fraud Detection Dashboard")
st.markdown("| IEEE-CIS Dataset| Fraud Detection System | 95.1 ROC-AUC")
st.divider()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Transaction", "Model Performance", "About"])

# Home Page
if page == "Home":
    st.header("Welcome to the Fraud Detection Dashboard")
    st.markdown("""
    This dashboard provides insights into the performance of our fraud detection system, which is trained on the IEEE-CIS dataset. 
    Explore the data overview, model performance metrics, and access the API documentation for more details.
    """)
    st.image("https://images.pexels.com/photos/7948054/pexels-photo-7948054.jpeg", width=True)

# Predict Transaction Page
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

    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
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
                        st.error(f"🚨 FRAUD DETECTED")
                    else:
                        st.success(f"✅ LEGITIMATE TRANSACTION")

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
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.Timeout:
                st.warning("Request timed out. Render free tier may be sleeping. Try again in 30 seconds.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")

    # Fetch model info from API
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=30)
        if response.status_code == 200:
            info = response.json()
            st.success("✅ Connected to live API")
        else:
            info = None
    except:
        info = None
        st.warning("⚠️ Could not connect to API. Showing cached metrics.")

    # Performance metrics
    st.subheader("Model Metrics")

    col1, col2, col3, col4 = st.columns(4)

    perf = info["performance"] if info else {
        "roc_auc": 0.959,
        "recall": 0.823,
        "precision": 0.588,
        "f1_score": 0.685
    }

    with col1:
        st.metric("ROC-AUC", f"{perf['roc_auc']:.3f}", delta="vs 0.500 random")
    with col2:
        st.metric("Recall", f"{perf['recall']*100:.1f}%", delta="+78.8% vs naive")
    with col3:
        st.metric("Precision", f"{perf['precision']*100:.1f}%")
    with col4:
        st.metric("F1-Score", f"{perf['f1_score']:.3f}")

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm_data = {
        "Predicted Legitimate": [111245, 876],
        "Predicted Fraud": [2730, 3257]
    }
    cm_df = pd.DataFrame(
        cm_data,
        index=["Actual Legitimate", "Actual Fraud"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="RdYlGn_r",
        title="Confusion Matrix (Validation Set)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Fraud patterns from EDA
    st.subheader("Fraud Patterns Discovered (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        # Product fraud rates
        product_data = pd.DataFrame({
            "Product": ["C", "H", "W", "R", "S"],
            "Fraud Rate (%)": [4.25, 3.11, 2.81, 1.29, 1.02]
        })
        fig = px.bar(
            product_data,
            x="Product",
            y="Fraud Rate (%)",
            title="Fraud Rate by Product",
            color="Fraud Rate (%)",
            color_continuous_scale="Reds"
        )
        fig.add_hline(y=3.5, line_dash="dash", annotation_text="Overall avg")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Card type fraud rates
        card_data = pd.DataFrame({
            "Card Type": ["discover", "amex", "mastercard", "visa"],
            "Fraud Rate (%)": [6.68, 3.88, 3.50, 3.42]
        })
        fig = px.bar(
            card_data,
            x="Card Type",
            y="Fraud Rate (%)",
            title="Fraud Rate by Card Type",
            color="Fraud Rate (%)",
            color_continuous_scale="Reds"
        )
        fig.add_hline(y=3.5, line_dash="dash", annotation_text="Overall avg")
        st.plotly_chart(fig, use_container_width=True)

    # Feature engineering summary
    st.divider()
    st.subheader("Feature Engineering Summary")

    fe_data = pd.DataFrame({
        "Category": [
            "Original Features",
            "Missing Value Flags",
            "Temporal Features",
            "Aggregation Features"
        ],
        "Count": [434, 156, 15, 26],
        "Impact": [
            "Baseline",
            "+2-3% ROC-AUC",
            "+1-2% ROC-AUC",
            "+1-2% ROC-AUC"
        ]
    })
    st.dataframe(fe_data, use_container_width=True, hide_index=True)


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
    - **Features:** 634 (434 original + 200 engineered)
    - **Fraud Rate:** 3.5% (20,663 fraudulent transactions)

    ### Key Findings
    - Missing data correlates with fraud (+47% for some features)
    - Velocity spikes indicate fraudulent behavior
    - Temporary email domains have 5× higher fraud rate
    - Night transactions are 22% riskier than daytime

    ### Tech Stack
    | Component | Technology |
    |-----------|------------|
    | Model | XGBoost |
    | API | FastAPI |
    | Dashboard | Streamlit |
    | Deployment | Render |
    | Language | Python 3.10 |

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
                st.success(f"✅ API is {health['status']}")
                st.json(health)
            else:
                st.error("API returned an error")
        except:
            st.error("Could not reach API")