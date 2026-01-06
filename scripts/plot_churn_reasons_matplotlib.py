import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)

fn = 'GhanaTelecomData_final.csv'
if not os.path.exists(fn):
    fn = 'GhanaTelecomData_cleaned.csv'

if not os.path.exists(fn):
    raise FileNotFoundError(f"Neither GhanaTelecomData_final.csv nor GhanaTelecomData_cleaned.csv found in {os.getcwd()}")

 df = pd.read_csv(fn)

reason_counts = {
    'Network Coverage': int(df['ChurnReason_NetworkCoverage'].sum()) if 'ChurnReason_NetworkCoverage' in df.columns else 0,
    'Customer Service': int(df['ChurnReason_CustomerService'].sum()) if 'ChurnReason_CustomerService' in df.columns else 0,
    'Pricing': int(df['ChurnReason_Pricing'].sum()) if 'ChurnReason_Pricing' in df.columns else 0
}

labels = list(reason_counts.keys())
sizes = list(reason_counts.values())

fig, ax = plt.subplots(figsize=(6,6))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
plt.title('Churn Reasons Distribution')
png_path = os.path.join('plots','churn_reasons_distribution_matplotlib.png')
plt.savefig(png_path, bbox_inches='tight')
plt.close()
print(f"Saved matplotlib PNG to {png_path}")
