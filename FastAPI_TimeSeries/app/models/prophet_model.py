from prophet import Prophet

def prophet_forecast(data, periods):
    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
