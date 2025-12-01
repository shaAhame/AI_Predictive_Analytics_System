#profitability_pipeline

import pickle
from src.forecasting import (
    load_and_clean_data,
    prepare_monthly_log_data,
    train_prophet_model,
    make_forecast,
)
from src.profitability_checker import check_profitability

def main():
    # Step 1: Load and clean your dataset (update file path accordingly)
    df = load_and_clean_data("data/SuperstoresSales.csv")
    
    # Step 2: Prepare the data (group by month, remove outliers, log-transform)
    monthly_df = prepare_monthly_log_data(df)
    
    # Step 3: Train Prophet models on revenue and expense
    rev_model = train_prophet_model(monthly_df, 'Log_Revenue')
    exp_model = train_prophet_model(monthly_df, 'Log_Expense')
    
    # Step 4: Make forecasts for the next 6 months (or any period you want)
    forecast_rev = make_forecast(rev_model, periods=6)
    forecast_exp = make_forecast(exp_model, periods=6)
    
    # Step 5: Check profitability from the forecasts
    status, projected_profit = check_profitability(forecast_rev, forecast_exp)
    
    # Step 6: Print the results
    print("Forecast Status:", status)
    print("Projected Profit:", round(projected_profit, 2))

if __name__ == "__main__":
    main()


# Save the pipeline (function) to a .pkl file
with open("pipelines/profitability_pipeline.pkl", "wb") as f:
    pickle.dump(check_profitability, f)

print("✅ Profitability pipeline saved to pipelines/profitability_pipeline.pkl")
