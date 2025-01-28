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

    # Check for required columns
    if "date" not in df.columns or "temperature" not in df.columns:
        raise ValueError("File must contain 'date' and 'temperature' columns.")

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    return df

def make_prediction(df):
    """Make weather forecast and generate visualization."""
    # Rename columns for Prophet
    df = df.rename(columns={"date": "ds", "temperature": "y"})

    # Train Prophet model
    model = Prophet()
    model.fit(df)

    # Generate future dates
    future = model.make_future_dataframe(periods=7)  # Predict next 7 days
    forecast = model.predict(future)

    # Combine historical and forecasted data
    forecast = forecast.rename(columns={"ds": "date", "yhat": "temperature"})
    combined = pd.concat([df.rename(columns={"ds": "date", "y": "temperature"}), forecast], ignore_index=True)

    # Save the visualization
    plot_dir = tempfile.gettempdir()
    plot_path = os.path.join(plot_dir, "forecast_plot.html")
    fig = px.line(
        combined,
        x="date",
        y="temperature",
        title="Weather Forecast (Historical + Prediction)",
        labels={"date": "Date", "temperature": "Temperature"},
    )
    fig.update_traces(mode="lines+markers")  # Add markers for clarity
    fig.write_html(plot_path)

    return plot_path, combined[["date", "temperature"]]
