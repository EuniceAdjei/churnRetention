import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load your churn dataset
df = pd.read_csv("telecom_churn.csv")  # replace with your actual file

# Example visualization: churn vs non-churn counts
fig = px.histogram(df, x="Churn", title="Customer Churn Distribution")

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Churn Retention Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)
