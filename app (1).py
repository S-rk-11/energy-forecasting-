import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="PJM Energy Forecast", layout="centered")
st.title("🔌 PJM Hourly Energy Forecast")
st.markdown("""
This app forecasts PJM hourly energy consumption using an XGBoost model.
Select how many future days you want to forecast.
""")

# ----------------------
# Load Trained Model
# ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_energy_forecast_model.joblib")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ----------------------
# Load Historical Data
# ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("PJMW_hourly.csv", parse_dates=['Datetime'])
        df.set_index('Datetime', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

data = load_data()

# ----------------------
# Feature Engineering
# ----------------------
def create_features(df):
    df['lag_1'] = df['PJMW_MW'].shift(1)
    df['lag_2'] = df['PJMW_MW'].shift(2)
    df['rolling_mean_3'] = df['PJMW_MW'].rolling(window=3).mean().shift(1)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df[['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']]

# ----------------------
# User Input
# ----------------------
future_days = st.slider("Select number of future days to forecast:", min_value=1, max_value=30, value=7)

# ----------------------
# Prepare Forecast Data
# ----------------------
df = data.copy()
forecast_steps = future_days * 24
predictions = []

for i in range(forecast_steps):
    last_row = df.iloc[-1:].copy()
    features = create_features(df).iloc[[-1]]

    if features.isnull().any().any():
        st.warning("❌ Not enough past data to compute all features. Try again later.")
        st.stop()

    pred = model.predict(features)[0]
    next_timestamp = last_row.index[0] + timedelta(hours=1)
    df.loc[next_timestamp] = [pred]  # Add forecast to df for next step

    predictions.append((next_timestamp, pred))

# ----------------------
# Forecast Result
# ----------------------
forecast_df = pd.DataFrame(predictions, columns=["Datetime", "Forecast_MW"]).set_index("Datetime")

# Combine with Past Data for Plot
recent_df = data[['PJMW_MW']].rename(columns={'PJMW_MW': 'Actual_MW'}).tail(7*24)
plot_df = pd.concat([recent_df, forecast_df], axis=0)

# ----------------------
# Plot
# ----------------------
st.subheader("📈 Forecasted Energy Consumption")
fig, ax = plt.subplots(figsize=(10, 4))
plot_df.plot(ax=ax, linewidth=2)
plt.xlabel("Datetime")
plt.ylabel("MW Consumption")
plt.grid(True)
st.pyplot(fig)

# ----------------------
# Download
# ----------------------
st.download_button("📥 Download Forecast CSV", data=forecast_df.to_csv(), file_name="forecast.csv")
