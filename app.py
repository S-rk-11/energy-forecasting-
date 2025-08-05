import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Load model
model = joblib.load("xgb_energy_forecast_model.joblib")

# App Title
st.title("Hourly Energy Consumption Forecast")
st.markdown("Forecast the next hour's PJM Energy Consumption (MW)")

# Input section
st.header("Enter Current Data")

# Input values
lag_1 = st.number_input("Lag 1 (previous hour consumption)", min_value=0.0)
lag_2 = st.number_input("Lag 2 (2 hours ago consumption)", min_value=0.0)
rolling_mean_3 = st.number_input("3-Hour Rolling Mean (before current hour)", min_value=0.0)
dayofweek = st.selectbox("Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.selectbox("Month", list(range(1,13)), format_func=lambda x: datetime.date(1900, x, 1).strftime('%B'))

# Predict button
if st.button("Predict Energy Consumption"):
    input_data = pd.DataFrame([[lag_1, lag_2, rolling_mean_3, dayofweek, month]],
                              columns=['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month'])
    prediction = model.predict(input_data)[0]
    st.success(f"Forecasted PJMW_MW for Next Hour: {prediction:.2f} MW")
