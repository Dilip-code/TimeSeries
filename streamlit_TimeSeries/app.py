import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("Time Series Forecasting App")

uploaded_file = st.file_uploader("Upload a CSV file with 'ds' (date) and 'y' (value) columns", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    if 'ds' in data.columns and 'y' in data.columns:
        st.write("Dataset Overview:")
        st.write(data.describe())

        st.write("Time Series Data Visualization:")
        fig, ax = plt.subplots()
        ax.plot(data['ds'], data['y'], label="Original Data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        st.write("Prophet Model Configuration:")
        periods = st.slider("Select number of periods to forecast (e.g., months)", min_value=1, max_value=36, value=12)
        freq = st.selectbox("Select frequency for the forecast", ['D', 'W', 'M'], index=2)  # Default to 'M'

        model = Prophet()
        model.fit(data)

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        st.write("Forecasted Data Preview:")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        st.write("Forecast Visualization:")
        fig_forecast = model.plot(forecast)
        st.pyplot(fig_forecast)

        st.write("Forecast Components:")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)
    else:
        st.error("The dataset must have 'ds' (date) and 'y' (value) columns.")
else:
    st.info("Please upload a CSV file to get started.")

