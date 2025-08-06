import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="PJM Daily Energy Forecast", layout="centered")
st.title("\U0001F50C PJM Daily Energy Forecast")
st.markdown("""
This professional web application forecasts PJM **daily** energy consumption using a pre-trained **XGBoost** model.

Upload the most recent dataset, select the forecast horizon (up to 30 days), and visualize or download the predicted results.
""")

# ----------------------
# Load Trained Model
# ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_energy_forecast_model.joblib")
    except Exception as e:
        st.error(f"\u274C Error loading model: {e}")
        st.stop()

model = load_model()

# ----------------------
# Load and Resample Data
# ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"])
        df.set_index("Datetime", inplace=True)
        return df.resample("D").mean()  # Convert to daily average
    except Exception as e:
        st.error(f"\u274C Error loading dataset: {e}")
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
    return df

# ----------------------
# User Input: Forecast Days
# ----------------------
future_days = st.slider("Select the number of future days to forecast:", 1, 30, 7)

# ----------------------
# Forecast Logic
# ----------------------
df = data.copy()
df = create_features(df)
df.dropna(inplace=True)

predictions = []
last_known = df.copy()

for _ in range(future_days):
    next_date = last_known.index[-1] + timedelta(days=1)

    next_row = pd.DataFrame(index=[next_date])
    next_row['lag_1'] = last_known['PJMW_MW'].iloc[-1]
    next_row['lag_2'] = last_known['PJMW_MW'].iloc[-2]
    next_row['rolling_mean_3'] = last_known['PJMW_MW'].iloc[-3:].mean()
    next_row['dayofweek'] = next_date.dayofweek
    next_row['month'] = next_date.month

    X_pred = next_row[['lag_1', 'lag_2', 'rolling_mean_3', 'dayofweek', 'month']]
    pred = model.predict(X_pred)[0]

    next_row['PJMW_MW'] = pred
    last_known = pd.concat([last_known, next_row])
    predictions.append((next_date, pred))

# ----------------------
# Results Preparation
# ----------------------
forecast_df = pd.DataFrame(predictions, columns=["Datetime", "Forecast_MW"]).set_index("Datetime")
recent_actual = df[["PJMW_MW"]].rename(columns={"PJMW_MW": "Actual_MW"}).tail(30)
plot_df = pd.concat([recent_actual, forecast_df], axis=0)

# ----------------------
# Visualization
# ----------------------
st.subheader("\U0001F4C8 Forecasted Daily Energy Consumption")
fig, ax = plt.subplots(figsize=(10, 5))
plot_df.plot(ax=ax, linewidth=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Energy Consumption (MW)")
plt.title("PJM Daily Energy Forecast: Actual vs Predicted")
plt.grid(True)
st.pyplot(fig)

# ----------------------
# Download Forecast
# ----------------------
st.download_button(
    label="\U0001F4E5 Download Forecast CSV",
    data=forecast_df.reset_index().to_csv(index=False),
    file_name="pjm_30_day_forecast.csv",
    mime="text/csv"
)
