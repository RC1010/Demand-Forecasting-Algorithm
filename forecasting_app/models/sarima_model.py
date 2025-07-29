import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def calculate_sarima_forecast(data, order=(1,1,1), seasonal_order=(1,1,1,7), steps=7):
    train = data['Quantity Ordered']
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast
