import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("GhanaTelecomData.csv")

# Preview the first 5 rows
print(df.head())

# Drop unnecessary columns (keep numeric derived columns)
drop_cols = [
    "AgeGroup",          # we’ll use AgeGroup_Numeric instead
    "Gender",            # we’ll use Gender_Numeric instead
    "MonthlyCharges",    # we’ll use MonthlyCharges_Numeric instead
    "ChurnLikelihood"    # derived score, not raw data
]
for c in drop_cols:
    if c not in df.columns:
        # already absent is fine
        drop_cols.remove(c)
        break
if len(drop_cols) > 0:
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(existing, axis=1)

# Basic checks
print("Remaining columns:", df.columns.tolist())
print("Dataset shape:", df.shape)
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after cleaning:", df.duplicated().sum())
print("Dataset shape after cleaning:", df.shape)

# Fix common typo in column name if present
if 'Gender_NuUmeric' in df.columns:
    df = df.rename(columns={'Gender_NuUmeric': 'Gender_Numeric'})

# Identify numeric columns automatically
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# create plots dir
os.makedirs('plots', exist_ok=True)

def cap_outliers_iqr(series: pd.Series):
    """Cap numeric outliers using the IQR method. Returns (capped_series, n_outliers, lower, upper)."""
    s = series.dropna().astype(float)
    if s.empty:
        return series, 0, None, None
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    n_out = int(mask.sum())
    if n_out == 0:
        return series, 0, lower, upper
    capped = series.copy().astype(float)
    capped.loc[capped < lower] = lower
    capped.loc[capped > upper] = upper
    return capped, n_out, lower, upper

# Loop over numeric columns, show before/after boxplots and cap outliers
for col in numeric_cols:
    try:
        if df[col].dropna().empty:
            print(f"Skipping {col} — no numeric data")
            continue
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Before - Outlier Check: {col}")
        plt.savefig(f"plots/{col}_before.png")
        plt.close()

        capped_series, n_out, lower, upper = cap_outliers_iqr(df[col])
        if n_out > 0:
            df[col] = capped_series
            print(f"Capped {n_out} outliers in {col} to range [{lower}, {upper}]")
        else:
            print(f"No outliers detected in {col}")

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"After - Outlier Check: {col}")
        plt.savefig(f"plots/{col}_after.png")
        plt.close()
    except Exception as e:
        print(f"Skipping column {col} due to error: {e}")

# Optionally save cleaned dataset
try:
    df.to_csv("GhanaTelecomData_cleaned.csv", index=False)
    print("Saved cleaned data to GhanaTelecomData_cleaned.csv")
except Exception:
    pass

# --- Additional cleaning steps for modeling readiness ---
# 1) Handle missing values: median for numeric, mode for categorical
# 2) Ensure correct dtypes
# 3) One-hot encode categorical variables (drop_first=True)
# 4) Save final dataset

# Recompute numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print('\nMissing values before imputation:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# Impute numeric columns with median
for col in numeric_cols:
    if df[col].isnull().any():
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"Imputed numeric {col} with median={med}")

# Impute categorical with mode (or 'Missing' if mode not available)
for col in cat_cols:
    if df[col].isnull().any():
        try:
            mode = df[col].mode().iloc[0]
        except Exception:
            mode = 'Missing'
        df[col] = df[col].fillna(mode)
        print(f"Imputed categorical {col} with mode={mode}")

print('\nMissing values after imputation:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# Convert boolean-like columns to int
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# One-hot encode categorical columns for modeling
if cat_cols:
    print(f"One-hot encoding categorical columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print('\nFinal dataset shape (ready for modeling):', df.shape)

try:
    df.to_csv("GhanaTelecomData_final.csv", index=False)
    print("Saved final cleaned data to GhanaTelecomData_final.csv")
except Exception as e:
    print("Failed to save final CSV:", e)
import plotly.express as px
import pandas as pd

# Count churn reasons
reason_counts = {
    "Network Coverage": df["ChurnReason_NetworkCoverage"].sum(),
    "Customer Service": df["ChurnReason_CustomerService"].sum(),
    "Pricing": df["ChurnReason_Pricing"].sum()
}

reason_df = pd.DataFrame(list(reason_counts.items()), columns=["Reason", "Count"])

# Plot pie chart
fig = px.pie(reason_df, names="Reason", values="Count", title="Churn Reasons Distribution")
fig.show()

