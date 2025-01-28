import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet 

def prophet_forecast(df, forecast_periods):
    df['ds'] = pd.to_datetime(df['ds'])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)

def main():
    st.title('Time Series Forecasting with Prophet')

    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview", data.head())

        if 'ds' not in data.columns or 'y' not in data.columns:
            st.error("The CSV file must contain 'ds' (date) and 'y' (value) columns.")
        else:
            periods = st.sidebar.slider("Select number of forecast periods", min_value=1, max_value=365, value=6)

            forecast_df = prophet_forecast(data, periods)
            st.write("### Forecasted Values", forecast_df)

            plt.figure(figsize=(10, 6))
            plt.plot(forecast_df['ds'], forecast_df['yhat'], label="Forecasted", color='blue')
            plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='blue', alpha=0.2)
            plt.title("Prophet Time Series Forecast")
            plt.xlabel("Date")
            plt.ylabel("Forecasted Value")
            plt.legend()

            st.pyplot(plt)

if __name__ == "__main__":
    main()
