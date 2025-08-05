import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# Load trained model
model = joblib.load('xgb_energy_forecast_model.joblib')

# Title
st.title("30-Day Energy Consumption Forecast")
st.markdown("Forecast the next 30 days of energy (MW) using XGBoost model.")

# Sidebar input
st.sidebar.header("Input Latest Known Data")
lag_1 = st.sidebar.number_input("Lag 1 (Yesterday's MW)", value=50000.0)
lag_2 = st.sidebar.number_input("Lag 2 (Day Before Yesterday's MW)", value=49500.0)

# Generate forecast
if st.button("Forecast Next 30 Days"):
    history = pd.Series([lag_2, lag_1])
    future_dates = pd.date_range(start=pd.Timestamp.today() + timedelta(days=1), periods=30, freq='D')
    future_preds = []

    for date in future_dates:
        rolling_mean_3 = history[-3:].mean() if len(history) >= 3 else history.mean()
        dayofweek = date.dayofweek
        month = date.month

        input_data = pd.DataFrame([[history[-1], history[-2], rolling_mean_3, dayofweek, month]],
                                  columns=['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month'])

        pred = model.predict(input_data)[0]
        future_preds.append(pred)
        history.loc[date] = pred

    # Plot
    st.subheader("Forecast Plot")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(future_dates, future_preds, marker='o', linestyle='-', color='red')
    ax.set_title("Forecasted Energy Consumption for Next 30 Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (MW)")
    ax.grid(True)
    st.pyplot(fig)

    # Show table
    st.subheader("Forecast Data")
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted MW": future_preds})
    st.dataframe(forecast_df)
