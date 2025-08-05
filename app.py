import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

# Title
st.title("Daily Energy Consumption Forecast")

# Load the saved model
model = joblib.load("daily_energy_forecast_model.joblib")

# User input - last known value (can be from real data or user input)
last_value = st.number_input("Enter last known energy consumption value (MW):", value=40000.0)

# Forecast next 30 days
forecast = []
current_input = last_value

for i in range(30):
    # You may need to shape the input depending on your model
    input_features = np.array([[current_input]])
    prediction = model.predict(input_features)[0]
    forecast.append(prediction)
    current_input = prediction

# Prepare dates
forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=30)
df_forecast = pd.DataFrame({'Date': forecast_dates, 'Predicted Consumption (MW)': forecast})

# Line chart
st.line_chart(df_forecast.set_index('Date'))

# Show table
st.dataframe(df_forecast)

# Download CSV
csv = df_forecast.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast as CSV", csv, "30_day_forecast.csv", "text/csv")
