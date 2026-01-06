import streamlit as st
import pickle
import pandas as pd

# Load your trained churn model
model = pickle.load(open("churn_model.pkl", "rb"))

st.title("📊 Telecom Churn Prediction App")
st.write("Welcome, Eunice! Your app is running locally 🎉")

# Tabs for single vs batch prediction
tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction"])

# --- Single Prediction ---
with tab1:
    st.header("Single Customer Prediction")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0, value=50)
    pricing_issue = st.selectbox("Churn Reason: Pricing", ["Yes", "No"])

    if st.button("Predict Single Customer"):
        # Format input like training data
        data = {
            "AgeGroup_Numeric": [age],
            "Gender_Numeric": [1 if gender == "Male" else 0],
            "MonthlyCharges_Numeric": [monthly_charges],
            "ChurnReason_Pricing": [1 if pricing_issue == "Yes" else 0]
        }
        df = pd.DataFrame(data)
        prediction = model.predict(df)[0]
        st.success("Customer will churn" if prediction == 1 else "Customer will stay")

# --- Batch Prediction ---
with tab2:
    st.header("Batch Prediction (Upload CSV)")
    uploaded_file = st.file_uploader("Upload customer data CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        predictions = model.predict(batch_df)
        batch_df["Prediction"] = ["Churn" if p == 1 else "Stay" for p in predictions]
        st.write(batch_df)
import joblib
model = joblib.load("churn_model.pkl")
