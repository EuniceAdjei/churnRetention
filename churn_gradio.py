import gradio as gr
import pandas as pd

def churn_prediction(file):
    df = pd.read_csv(file.name)
    # For now, just return the first 5 rows
    return df.head()

demo = gr.Interface(
    fn=churn_prediction,
    inputs=gr.File(label="Upload CSV"),
    outputs=gr.Dataframe()
)

demo.launch()
