
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime

st.title("🔋 Energy Management App")

uploaded_file = st.file_uploader("Upload your half-hourly data Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
        st.stop()
    
    # Ensure date and time are in correct format
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df[['Timestamp', 'Value']]
    df = df.sort_values('Timestamp')

    # Add calendar info
    df['Weekday'] = df['Timestamp'].dt.day_name()
    df['IsWeekend'] = df['Timestamp'].dt.weekday >= 5
    df['HalfHour'] = df['Timestamp'].dt.hour * 2 + df['Timestamp'].dt.minute // 30
    df['Week'] = df['Timestamp'].dt.to_period("W").apply(lambda r: r.start_time)

    # Forecasting using linear regression
    df['Ordinal'] = df['Timestamp'].map(datetime.toordinal)
    X = df[['Ordinal']]
    y = df['Value']
    model = LinearRegression().fit(X, y)
    df['Forecast'] = model.predict(X)
    rmse = mean_squared_error(y, df['Forecast'], squared=False)

    # Anomaly detection using IQR
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['Anomaly'] = (df['Value'] < lower_bound) | (df['Value'] > upper_bound)

    # Identify quadrant position
    df['Quadrant'] = pd.cut(df['Value'], bins=[-np.inf, Q1, Q3, np.inf], labels=['Lower', 'Middle', 'Upper'])

    # Plot energy usage with forecast and anomalies
    st.subheader("📈 Energy Usage, Forecast & Anomalies")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Timestamp'], df['Value'], label='Actual')
    ax.plot(df['Timestamp'], df['Forecast'], label='Forecast', linestyle='--')
    ax.scatter(df.loc[df['Anomaly'], 'Timestamp'], df.loc[df['Anomaly'], 'Value'], color='red', label='Anomalies')
    ax.scatter(df.loc[df['Quadrant'] == 'Upper', 'Timestamp'], df.loc[df['Quadrant'] == 'Upper', 'Value'], 
               color='orange', label='Upper Quadrant', marker='^')
    ax.scatter(df.loc[df['Quadrant'] == 'Lower', 'Timestamp'], df.loc[df['Quadrant'] == 'Lower', 'Value'], 
               color='blue', label='Lower Quadrant', marker='v')
    ax.axhline(y=lower_bound, color='blue', linestyle=':', linewidth=1, label='Lower Threshold')
    ax.axhline(y=upper_bound, color='orange', linestyle=':', linewidth=1, label='Upper Threshold')
    ax.set_title(f"Energy Usage Forecast (RMSE: {rmse:.2f})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)
