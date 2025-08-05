# Demand-Forecasting-Algorithm

# 📊 Forecasting Dashboard (SMA | SARIMA | XGBoost)

An interactive web app built with **Streamlit** that performs time series forecasting using:

- 🟦 Simple Moving Average (SMA)
- 🟧 SARIMA (Seasonal Auto-Regressive Integrated Moving Average)
- 🟥 XGBoost Regression

This dashboard allows users to upload their time series sales/order data and visualize forecasts and model performance metrics.

---

## 🚀 Features

- 📄 Upload `.csv` or `.xlsx` files
- 📈 Forecasts next 7 days of data using 3 different models
- 📊 Visualizes actual vs forecast values
- 📌 Displays performance metrics: MAE, MSE, RMSE, MAPE
- 🔍 Automatically extracts time and lag-based features for XGBoost

---

## 📂 Sample Input Format

The uploaded file must include the following columns:

| Order Date | Quantity ordered new |
|------------|----------------------|
| 2025-01-01 | 100                  |
| 2025-01-02 | 120                  |
| ...        | ...                  |

---

## 🧠 Forecasting Methods

### 1. 🟦 Simple Moving Average (SMA)
Forecasts by averaging the past N values. (Window size = 4 by default)

### 2. 🟧 SARIMA
Time series forecasting using seasonal ARIMA models with automatic differencing.

### 3. 🟥 XGBoost
Machine learning-based regression using time-based and lag-based feature engineering.

---

## 📊 Metrics

Each model evaluates its predictions using:

- **MAE** – Mean Absolute Error
- **MSE** – Mean Squared Error
- **RMSE** – Root Mean Squared Error
- **MAPE** – Mean Absolute Percentage Error

---

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/forecasting-dashboard.git
   cd forecasting-dashboard
2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies:

pip install -r requirements.txt

4. Run the app:

streamlit run app.py

📦 Dependencies
- streamlit
- pandas
- numpy
- matplotlib
- xgboost
- statsmodels
- scikit-learn
- openpyxl (for Excel file support)
  
You can install all dependencies using:

pip install -r requirements.txt

📁 Project Structure
forecasting-dashboard/
├── app.py                  # Main Streamlit app
├── requirements.txt
└── README.md

✨ Screenshots
<img width="1910" height="584" alt="Screenshot 2025-08-05 214351" src="https://github.com/user-attachments/assets/38fff48f-c006-433c-9ed5-e66113b30f7e" />

SMA
<img width="1851" height="596" alt="Screenshot 2025-08-05 214412" src="https://github.com/user-attachments/assets/cfcb9d26-7885-42d4-ae86-5a8be292e415" />

SARIMA
<img width="1900" height="626" alt="Screenshot 2025-08-05 214428" src="https://github.com/user-attachments/assets/66cb3c86-7690-4ba8-9f72-07d1bbbe6990" />

XGBOOST
<img width="1861" height="581" alt="Screenshot 2025-08-05 214440" src="https://github.com/user-attachments/assets/7a28ce85-6751-4e99-8aa1-b1c0703f52f8" />


🙌 Acknowledgements
- Streamlit
- XGBoost
- statsmodels
- scikit-learn


---

Let me know if:
- You want this converted to a real markdown file (`README.md`) download
- You want to organize the code into folders (like `models/`) and I can update the README accordingly
- You want to publish it on GitHub and need help writing the `requirements.txt` too
