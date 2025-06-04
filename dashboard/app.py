
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.forecasting import (
    load_and_clean_data,
    prepare_monthly_log_data,
    train_prophet_model,
    make_forecast,
    merge_forecasts,
)

from src.ai_recommendation import generate_recommendation

st.set_page_config(page_title="📈 Business Forecast Dashboard", layout="wide")

# === Sidebar ===
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

forecast_period = st.sidebar.slider("Forecast Months", min_value=1, max_value=120, value=6)

if uploaded_file:
    # === Load & Process Data ===
    df = load_and_clean_data(uploaded_file)
    monthly_df = prepare_monthly_log_data(df)

    # === Train Models ===
    st.sidebar.success("✅ Training models...")
    revenue_model = train_prophet_model(monthly_df, 'Log_Revenue')
    expense_model = train_prophet_model(monthly_df, 'Log_Expense')

    # === Make Forecasts ===
    revenue_forecast = make_forecast(revenue_model, forecast_period)
    expense_forecast = make_forecast(expense_model, forecast_period)

    # === Merge Forecasts ===
    merged_df, threshold = merge_forecasts(revenue_forecast, expense_forecast)

    # === Show Latest Forecast ===
    latest_row = merged_df.iloc[-1]
    predicted_profit = latest_row['Predicted_Profit']
    date = latest_row['ds']

    st.title("📊 Business Profit Forecast")

    st.subheader(f"📅 Forecast for: {pd.to_datetime(date).strftime('%B %Y')}")
    st.metric("Predicted Profit", f"${predicted_profit:,.2f}")

    if predicted_profit < 0:
        st.error("🔔 Alert: Projected Loss Detected")
        with st.spinner("🤖 Generating AI recommendations to reduce loss..."):
            advice = generate_recommendation(predicted_profit)
        st.markdown("### 💡 AI Recommendations")
        st.info(advice)
    else:
        st.success("✅ Business is on track. No immediate action needed.")

    # === Forecast Chart ===
    st.markdown("### 📉 Forecast Overview")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(merged_df['ds'], merged_df['Predicted_Revenue'], label='Revenue', color='green')
    ax.plot(merged_df['ds'], merged_df['Predicted_Expense'], label='Expense', color='red')
    ax.plot(merged_df['ds'], merged_df['Predicted_Profit'], label='Profit', color='blue')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount ($)")
    st.pyplot(fig)

    # === Show Full Forecast Table ===
    st.markdown("### 📋 Forecast Table")
    st.dataframe(merged_df[['ds', 'Predicted_Revenue', 'Predicted_Expense', 'Predicted_Profit', 'Status']].tail(forecast_period))

else:
    st.warning("📤 Please upload a CSV file to begin.")
