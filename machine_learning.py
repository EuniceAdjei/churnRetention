# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 2. Load dataset
df = pd.read_csv("GhanaTelecomData_cleaned.csv")

# 3. Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Define features (X) and target (y)
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data loaded and split successfully!")
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# 6. Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n✅ Logistic Regression Model Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# 7. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n✅ Random Forest Model Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# 8. Feature importance
importances = rf_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n🔝 Top 10 Features Driving Churn:")
print(feature_importance_df.head(10))

# 9. Save model AND training columns together
joblib.dump((rf_model, list(X.columns)), "churn_model.pkl")
print("\n✅ Model and training columns saved successfully as churn_model.pkl")
import pandas as pd
import joblib

# Load model and training columns
rf_model, training_columns = joblib.load("churn_model.pkl")

def predict_churn(raw_input_dict):
    input_df = pd.DataFrame([raw_input_dict])
    input_df = input_df.reindex(columns=training_columns, fill_value=0)
    prediction = rf_model.predict(input_df)[0]
    return "Churn" if prediction == 1 else "Not Churn"

# Example test input
raw_input = {
    "AgeGroup_Numeric": 25,
    "Gender_Numeric": 1,  # Male
    "ChurnReason_Pricing": 1,
    "ChurnReason_NetworkCoverage": 0,
    "ChurnReason_CustomerService": 1,
    "DurationWithCompany_6-12 months": 1,
    "TelecomCompany_MTN": 1
}

result = predict_churn(raw_input)
print("Prediction:", result)
import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
rf_model, training_columns = joblib.load("churn_model.pkl")

def predict_churn(raw_input_dict):
    input_df = pd.DataFrame([raw_input_dict])
    input_df = input_df.reindex(columns=training_columns, fill_value=0)
    prediction = rf_model.predict(input_df)[0]
    return "Churn" if prediction == 1 else "Not Churn"

# Streamlit UI
st.title("📊 Telecom Churn Prediction App")

st.write("Enter customer details below to predict churn:")

# Input fields
age = st.number_input("Age (numeric group)", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=1000, value=120)
duration = st.selectbox("Duration With Company", ["Less than 6 months", "6-12 months", "3-4 years", "5 or more years"])
pricing_issue = st.checkbox("Churn Reason: Pricing")
coverage_issue = st.checkbox("Churn Reason: Network Coverage")
customer_service_issue = st.checkbox("Churn Reason: Customer Service")
telecom_company = st.selectbox("Telecom Company", ["MTN", "Vodafone", "AirtelTigo"])

# Convert inputs into dictionary
raw_input = {
    "AgeGroup_Numeric": age,
    "Gender_Numeric": 1 if gender == "Male" else 0,
    "MonthlyCharges_Numeric": monthly_charges,
    "ChurnReason_Pricing": 1 if pricing_issue else 0,
    "ChurnReason_NetworkCoverage": 1 if coverage_issue else 0,
    "ChurnReason_CustomerService": 1 if customer_service_issue else 0,
    f"DurationWithCompany_{duration}": 1,
    f"TelecomCompany_{telecom_company}": 1
}

# Prediction button
if st.button("Predict Churn"):
    result = predict_churn(raw_input)
    st.success(f"Prediction: {result}")
    import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
rf_model, training_columns = joblib.load("churn_model.pkl")

def predict_churn(raw_input_dict):
    input_df = pd.DataFrame([raw_input_dict])
    input_df = input_df.reindex(columns=training_columns, fill_value=0)
    prediction = rf_model.predict(input_df)[0]
    return "Churn" if prediction == 1 else "Not Churn"

def batch_predict(df):
    # Reindex uploaded DataFrame to match training columns
    df = df.reindex(columns=training_columns, fill_value=0)
    predictions = rf_model.predict(df)
    df["Prediction"] = ["Churn" if p == 1 else "Not Churn" for p in predictions]
    return df

# Streamlit UI
st.title("📊 Telecom Churn Prediction App")

st.write("Choose between single customer prediction or batch CSV upload.")

# Tabs for single vs batch
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("🔹 Single Customer Prediction")
    age = st.number_input("Age (numeric group)", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=1000, value=120)
    duration = st.selectbox("Duration With Company", ["Less than 6 months", "6-12 months", "3-4 years", "5 or more years"])
    pricing_issue = st.checkbox("Churn Reason: Pricing")
    coverage_issue = st.checkbox("Churn Reason: Network Coverage")
    customer_service_issue = st.checkbox("Churn Reason: Customer Service")
    telecom_company = st.selectbox("Telecom Company", ["MTN", "Vodafone", "AirtelTigo"])

    raw_input = {
        "AgeGroup_Numeric": age,
        "Gender_Numeric": 1 if gender == "Male" else 0,
        "MonthlyCharges_Numeric": monthly_charges,
        "ChurnReason_Pricing": 1 if pricing_issue else 0,
        "ChurnReason_NetworkCoverage": 1 if coverage_issue else 0,
        "ChurnReason_CustomerService": 1 if customer_service_issue else 0,
        f"DurationWithCompany_{duration}": 1,
        f"TelecomCompany_{telecom_company}": 1
    }

    if st.button("Predict Churn"):
        result = predict_churn(raw_input)
        st.success(f"Prediction: {result}")

with tab2:
    st.subheader("🔹 Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded Data Preview:", df.head())
        
        if st.button("Run Batch Prediction"):
            results_df = batch_predict(df)
            st.write("✅ Predictions Complete")
            st.dataframe(results_df)
            
            # Optionally allow download
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
streamlit run app.py

