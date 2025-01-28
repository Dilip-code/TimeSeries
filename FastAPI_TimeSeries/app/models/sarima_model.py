from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
def sarima_forecast(data, periods):
    # Ensure 'ds' column is in datetime format
    data['ds'] = pd.to_datetime(data['ds'])
    
    # Fit SARIMA model (adjust the order as needed)
    model = SARIMAX(data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit()

    # Generate forecast for the specified periods
    forecast = result.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()

    # Convert forecast result into a DataFrame and rename columns
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={"index": "ds", "mean": "yhat", 
                                "mean_ci_lower": "yhat_lower", "mean_ci_upper": "yhat_upper"}, inplace=True)

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
