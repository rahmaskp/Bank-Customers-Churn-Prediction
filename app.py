import streamlit as st
import pandas as pd
import joblib

# =====================
# Load Model
# =====================
best_model = joblib.load("best_gb_model.joblib")

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# =====================
# Custom CSS
# =====================
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .title {
        font-size:28px !important;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size:18px !important;
        color: #7f8c8d;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# Title
# =====================
st.markdown('<p class="title">📊 Customer Churn Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fill in the customer details below to predict whether the customer will churn or stay loyal.</p>', unsafe_allow_html=True)

# =====================
# Input Form
# =====================
with st.form("churn_form"):
    st.markdown('<div class="card"><h4>📝 Customer Information</h4>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        CreditScore = st.number_input("Credit Score", min_value=300, max_value=1000, value=600)
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=5)
        Balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
        NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col2:
        HasCrCard_str = st.selectbox("Has Credit Card?", ["No", "Yes"])
        HasCrCard = 1 if HasCrCard_str == "Yes" else 0

        IsActiveMember_str = st.selectbox("Is Active Member?", ["No", "Yes"])
        IsActiveMember = 1 if IsActiveMember_str == "Yes" else 0

        EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    submitted = st.form_submit_button("🔮 Predict")
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# Prediction Result
# =====================
if submitted:
    input_data = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        "Gender": Gender,
        "Geography": Geography
    }])

    prediction = best_model.predict(input_data)[0]
    proba = best_model.predict_proba(input_data)[0][1]  # churn probability

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="card" style="background-color:#ffecec;">
                <h3 style="color:#e74c3c;">⚠️ Prediction: CHURN</h3>
                <p style="font-size:20px;">Churn probability: <b>{proba:.2%}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="card" style="background-color:#ecf9f1;">
                <h3 style="color:#27ae60;">✅ Prediction: LOYAL</h3>
                <p style="font-size:20px;">Churn probability: <b>{proba:.2%}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            f"""
            <div class="card">
            <h4>👤 Customer Profile</h4>
            <ul>
                <li><b>Credit Score:</b> {CreditScore}</li>
                <li><b>Age:</b> {Age}</li>
                <li><b>Tenure:</b> {Tenure} years</li>
                <li><b>Balance:</b> {Balance:,.0f}</li>
                <li><b>Products:</b> {NumOfProducts}</li>
                <li><b>Has Credit Card:</b> {HasCrCard_str}</li>
                <li><b>Active Member:</b> {IsActiveMember_str}</li>
                <li><b>Salary:</b> {EstimatedSalary:,.0f}</li>
                <li><b>Gender:</b> {Gender}</li>
                <li><b>Geography:</b> {Geography}</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
