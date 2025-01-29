from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
def sarima_forecast(data, periods):
    data['ds'] = pd.to_datetime(data['ds'])
    
    model = SARIMAX(data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit()

    forecast = result.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()

    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={"index": "ds", "mean": "yhat", 
                                "mean_ci_lower": "yhat_lower", "mean_ci_upper": "yhat_upper"}, inplace=True)

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
