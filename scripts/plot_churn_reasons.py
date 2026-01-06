import os
import pandas as pd
import plotly.express as px

os.makedirs('plots', exist_ok=True)

# Load cleaned dataset
fn = 'GhanaTelecomData_final.csv'
if not os.path.exists(fn):
    fn = 'GhanaTelecomData_cleaned.csv'

if not os.path.exists(fn):
    raise FileNotFoundError(f"Neither GhanaTelecomData_final.csv nor GhanaTelecomData_cleaned.csv found in working dir: {os.getcwd()}")

df = pd.read_csv(fn)

# Count churn reasons
reason_counts = {
    "Network Coverage": int(df["ChurnReason_NetworkCoverage"].sum()) if "ChurnReason_NetworkCoverage" in df.columns else 0,
    "Customer Service": int(df["ChurnReason_CustomerService"].sum()) if "ChurnReason_CustomerService" in df.columns else 0,
    "Pricing": int(df["ChurnReason_Pricing"].sum()) if "ChurnReason_Pricing" in df.columns else 0
}

reason_df = pd.DataFrame(list(reason_counts.items()), columns=["Reason", "Count"])

# Plot pie chart
fig = px.pie(reason_df, names="Reason", values="Count", title="Churn Reasons Distribution")
html_path = os.path.join('plots', 'churn_reasons_distribution.html')
fig.write_html(html_path)
print(f"Saved pie chart to {html_path}")

# Attempt to save static image (use kaleido engine)
try:
    png_path = os.path.join('plots', 'churn_reasons_distribution.png')
    fig.write_image(png_path, format='png', engine='kaleido')
    print(f"Also saved static PNG to {png_path}")
except Exception as e:
    print("PNG not saved (kaleido may not be available):", e)
