import pandas as pd
from prophet import Prophet
import plotly.express as px
import os
import tempfile
from io import StringIO

def process_file(contents):
    """Convert uploaded file content to DataFrame."""
    stringio = StringIO(contents.decode())
    df = pd.read_csv(stringio)

    if "date" not in df.columns or "temperature" not in df.columns:
        raise ValueError("File must contain 'date' and 'temperature' columns.")

    df["date"] = pd.to_datetime(df["date"])
    return df

def make_prediction(df):
    """Make weather forecast and generate visualization."""
    df = df.rename(columns={"date": "ds", "temperature": "y"})

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=7) 
    forecast = model.predict(future)

    forecast = forecast.rename(columns={"ds": "date", "yhat": "temperature"})
    combined = pd.concat([df.rename(columns={"ds": "date", "y": "temperature"}), forecast], ignore_index=True)

    plot_dir = tempfile.gettempdir()
    plot_path = os.path.join(plot_dir, "forecast_plot.html")
    fig = px.line(
        combined,
        x="date",
        y="temperature",
        title="Weather Forecast (Historical + Prediction)",
        labels={"date": "Date", "temperature": "Temperature"},
    )
    fig.update_traces(mode="lines+markers")  
    fig.write_html(plot_path)

    return plot_path, combined[["date", "temperature"]]
