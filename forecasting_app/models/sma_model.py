import pandas as pd

def calculate_sma_forecast(data, window_size=7):
    data = data.copy()
    data['SMA_Prediction'] = data['Quantity Ordered'].rolling(window=window_size).mean().shift(1)
    return data
