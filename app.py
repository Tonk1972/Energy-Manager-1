import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Energy Management App", layout="wide")
st.title("📊 Energy Management: Anomaly Detection, Benchmarking & Forecasting")

uploaded_file = st.file_uploader("Upload Excel File with 'Date', 'Time', and 'Value' columns", type=["xlsx", "xls", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    try:
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M')
    except Exception as e:
        st.error(f"Date format error: {e}")
    else:
        df = df[['Timestamp', 'Value']]
        df = df.sort_values('Timestamp')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df.dropna(subset=['Value'], inplace=True)

        # Weekend/Weekday categorization
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

        # Anomaly detection using z-score
        df['ZScore'] = zscore(df['Value'].fillna(method='ffill'))
        df['Anomaly'] = abs(df['ZScore']) > 3

        # Weekly grouping and time slotting
        df['Week'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
        df['Weekday'] = df['Timestamp'].dt.weekday
        df['HalfHour'] = df['Timestamp'].dt.hour * 2 + df['Timestamp'].dt.minute // 30

        # Forecasting with linear regression
        df['OrdinalTime'] = df['Timestamp'].map(pd.Timestamp.toordinal)
        X = df[['OrdinalTime']]
        y = df['Value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression().fit(X_train, y_train)
        df['Forecast'] = model.predict(X)

        # RMSE
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Plotting
        st.subheader("📈 Energy Usage, Forecast & Anomalies")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Timestamp'], df['Value'], label='Actual')
        ax.plot(df['Timestamp'], df['Forecast'], label='Forecast', linestyle='--')
        ax.scatter(df.loc[df['Anomaly'], 'Timestamp'], df.loc[df['Anomaly'], 'Value'], color='red', label='Anomalies')
        ax.set_title(f"Energy Usage Forecast (RMSE: {rmse:.2f})")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 Weekday vs Weekend Comparison")
        weekday_avg = df.groupby('IsWeekend')['Value'].mean()
        st.bar_chart(weekday_avg.rename({False: "Weekday", True: "Weekend"}))

        st.subheader("📆 Weekly Half-Hourly Trends")
        pivot = df.pivot_table(index=['Weekday', 'HalfHour'], columns='Week', values='Value')
        fig2 = plt.figure(figsize=(12, 5))
        for week in pivot.columns:
            plt.plot(pivot.index, pivot[week], label=str(week.date()))
        plt.title("Weekly Half-Hourly Comparison")
        plt.xlabel("Time of Week (Weekday + HalfHour Slot)")
        plt.ylabel("Value")
        plt.legend()
        st.pyplot(fig2)

        st.subheader("📥 Download Anomalies Report")
        anomaly_df = df[df['Anomaly']][['Timestamp', 'Value', 'ZScore']]
        st.dataframe(anomaly_df)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            anomaly_df.to_excel(writer, index=False, sheet_name='Anomalies')
        st.download_button("Download Anomalies as Excel", data=output.getvalue(), file_name="anomalies_report.xlsx")
