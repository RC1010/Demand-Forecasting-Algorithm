import pandas as pd
import numpy as np
import xgboost as xgb

def apply_xgboost_model(data, forecast_days=7):
    df = data.copy()
    df['lag_1'] = df['Quantity Ordered'].shift(1)
    df['dayofweek'] = df['Order Date'].dt.dayofweek
    df = df.dropna()

    X = df[['lag_1', 'dayofweek']]
    y = df['Quantity Ordered']

    model = xgb.XGBRegressor()
    model.fit(X, y)

    future_preds = []
    future_dates = []

    last_quantity = df.iloc[-1]['Quantity Ordered']
    last_date = df.iloc[-1]['Order Date']

    for _ in range(forecast_days):
        next_date = last_date + pd.Timedelta(days=1)
        next_dayofweek = next_date.dayofweek

        input_data = pd.DataFrame({
            'lag_1': [last_quantity],
            'dayofweek': [next_dayofweek]
        })

        pred = model.predict(input_data)[0]
        future_preds.append(pred)
        future_dates.append(next_date)

        # Update for next loop
        last_quantity = pred
        last_date = next_date

    forecast_df = pd.DataFrame({
        'Order Date': future_dates,
        'Forecast': future_preds
    })

    # Extract last known actual values for comparison
    y_true = df['Quantity Ordered'].iloc[-forecast_days:]
    y_pred = forecast_df['Forecast'].iloc[:forecast_days]

    return forecast_df, y_true.reset_index(drop=True), y_pred.reset_index(drop=True)
