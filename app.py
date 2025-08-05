import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# Title
st.title("Daily Energy Consumption Forecast (Next 30 Days)")

# Load Model and Scaler
model = joblib.load("daily_energy_forecast_model.joblib")

try:
    scaler = joblib.load("scaler.joblib")  # Only if you used it
except:
    scaler = None

# Load your latest data
uploaded_file = st.file_uploader("Upload latest daily energy consumption data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Datetime"])
    df = df.set_index("Datetime").resample("D").mean()

    # Show column names to verify
    st.write("Columns in uploaded file:", df.columns.tolist())

    # Rename to standard 'MW' column for forecast
    df.columns = ["MW"]

    # Show data preview
    st.subheader("Latest Uploaded Data (Daily)")
    st.dataframe(df.tail(10))

    # Generate features (lag-based)
    df_forecast = df.copy()
    for i in range(1, 8):
        df_forecast[f"lag_{i}"] = df_forecast["MW"].shift(i)
    df_forecast.dropna(inplace=True)

    # Forecast next 30 days
    future_preds = []
    last_known = df_forecast.iloc[-1][[f"lag_{i}" for i in range(1, 8)]].values

    for _ in range(30):
        input_features = last_known.reshape(1, -1)
        if scaler:
            input_features = scaler.transform(input_features)
        next_pred = model.predict(input_features)[0]
        future_preds.append(next_pred)

        # Update last_known lags
        last_known = np.roll(last_known, 1)
        last_known[0] = next_pred

    # Prepare results
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted_MW": future_preds
    })

    st.subheader("Forecast: Next 30 Days")
    st.dataframe(forecast_df)

    # Line Chart
    st.line_chart(forecast_df.set_index("Date"))

    # Download option
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button("Download Forecast CSV", data=csv, file_name="30_day_forecast.csv", mime="text/csv")
