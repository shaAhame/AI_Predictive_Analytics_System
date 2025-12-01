# Step 7: Full pipeline runner with model saving


import os
import pickle

# Import required functions
from src.forecasting import (
    load_and_clean_data,
    prepare_monthly_log_data,
    train_prophet_model,
    make_forecast,
    merge_forecasts
)

def run_pipeline(filepath, future_months=6):
    df = load_and_clean_data(filepath)
    monthly_df = prepare_monthly_log_data(df)

    rev_model = train_prophet_model(monthly_df, 'Log_Revenue')
    exp_model = train_prophet_model(monthly_df, 'Log_Expense')

    forecast_rev = make_forecast(rev_model, periods=future_months)
    forecast_exp = make_forecast(exp_model, periods=future_months)

    result_df, threshold = merge_forecasts(forecast_rev, forecast_exp)

    # Save models
    os.makedirs("pipelines", exist_ok=True)
    with open("pipelines/revenue_forecast_model.pkl", "wb") as f:
        pickle.dump(rev_model, f)
    with open("pipelines/expense_forecast_model.pkl", "wb") as f:
        pickle.dump(exp_model, f)

    return result_df, threshold

# --- RUN THE PIPELINE ---
if __name__ == "__main__":
    filepath = "data/SuperstoresSales.csv"  # <-- Change this to your real CSV file path
    result, threshold = run_pipeline(filepath)
    print("Pipeline executed successfully.")
    print("Threshold for anomaly detection:", threshold)
    print(result.tail())  # Show last few forecasted records
