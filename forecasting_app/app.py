import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ------------------------- App Config -------------------------
st.set_page_config(page_title="Forecasting Dashboard", layout="wide")
st.title("📊 Forecasting Dashboard (SMA | SARIMA | XGBoost)")

# ------------------------- Metrics Function -------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

def compute_metrics(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "MSE": mean_squared_error(true, pred),
        "RMSE": math.sqrt(mean_squared_error(true, pred)),
        "MAPE": mean_absolute_percentage_error(true, pred)
    }

# ------------------------- SMA Forecast -------------------------
def calculate_sma_forecast(data, window_size=4):
    data = data.copy()
    data = data.sort_values('Order Date')
    data['SMA'] = data['Quantity ordered new'].rolling(window=window_size).mean()
    data = data.dropna()
    metrics = compute_metrics(data['Quantity ordered new'][window_size:], data['SMA'][window_size:])
    return data[['Order Date', 'Quantity ordered new', 'SMA']], metrics

# ------------------------- SARIMA Forecast -------------------------
def calculate_sarima_forecast(data, order=(1,1,1), steps=7):
    data = data.copy()
    data = data.dropna(subset=["Quantity ordered new"])
    data.set_index("Order Date", inplace=True)

    # Differencing for stationarity
    differenced = data['Quantity ordered new'].diff().dropna()
    model = SARIMAX(differenced, order=order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    forecast_diff = model_fit.forecast(steps=steps)
    last_value = data['Quantity ordered new'].iloc[-1]
    forecast = np.r_[last_value, forecast_diff].cumsum()[1:]

    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps)

    forecast_df = pd.DataFrame({
        "Order Date": forecast_dates,
        "Forecast": forecast
    })

    recent_actual = data['Quantity ordered new'][-steps:].values
    true_values = recent_actual if len(recent_actual) == steps else [0]*steps

    metrics = compute_metrics(true_values, forecast[:len(true_values)])
    return forecast_df, metrics

# ------------------------- XGBoost Forecast -------------------------
def apply_xgboost_model(data, steps=7):
    df = data.copy()
    df = df.dropna(subset=['Quantity ordered new'])
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df.sort_values('Order Date', inplace=True)

    # Feature Engineering with enhanced lag/time features
    df['dayofweek'] = df['Order Date'].dt.dayofweek
    df['month'] = df['Order Date'].dt.month
    df['day'] = df['Order Date'].dt.day
    df['lag1'] = df['Quantity ordered new'].shift(1)
    df['lag2'] = df['Quantity ordered new'].shift(2)
    df['rolling_mean'] = df['Quantity ordered new'].rolling(window=3).mean()
    df.dropna(inplace=True)

    features = ['dayofweek', 'month', 'day', 'lag1', 'lag2', 'rolling_mean']
    X = df[features]
    y = df['Quantity ordered new']

    if len(X) < steps + 1:
        return pd.DataFrame(), {"MAE": None, "MSE": None, "RMSE": None, "MAPE": None}

    X_train, y_train = X.iloc[:-steps], y.iloc[:-steps]
    X_test, y_test = X.iloc[-steps:], y.iloc[-steps:]

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, objective='reg:squarederror')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    forecast_df = pd.DataFrame({
        'Order Date': df['Order Date'].iloc[-steps:].values,
        'Forecast': y_pred,
        'Actual': y_test.values
    })

    metrics = compute_metrics(y_test, y_pred)
    return forecast_df, metrics

# ------------------------- File Upload -------------------------
uploaded_file = st.file_uploader("📄 Upload Excel or CSV file", type=["xlsx", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'Order Date' not in df.columns or 'Quantity ordered new' not in df.columns:
            st.error("❌ Required columns: 'Order Date' and 'Quantity ordered new'")
        else:
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df = df.sort_values('Order Date')

            st.subheader("📋 Preview Data")
            st.dataframe(df.tail())

            # SMA Forecast
            st.header("🟦 SMA Forecast")
            sma_result, sma_metrics = calculate_sma_forecast(df)
            st.json({"📌 SMA Metrics": sma_metrics})
            st.line_chart(sma_result.set_index("Order Date")[["Quantity ordered new", "SMA"]])

            # SARIMA Forecast
            st.header("🟧 SARIMA Forecast")
            try:
                sarima_result, sarima_metrics = calculate_sarima_forecast(df)
                st.json({"📌 SARIMA Metrics": sarima_metrics})
                st.line_chart(sarima_result.set_index("Order Date")["Forecast"])
            except Exception as e:
                st.error(f"❌ SARIMA Error: {e}")

            # XGBoost Forecast
            st.header("🟥 XGBoost Forecast")
            xgb_result, xgb_metrics = apply_xgboost_model(df)
            st.json({"📌 XGBoost Metrics": xgb_metrics})
            if not xgb_result.empty:
                st.line_chart(xgb_result.set_index("Order Date")[["Forecast", "Actual"]])
            else:
                st.warning("⚠️ Not enough data for XGBoost forecast.")

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")