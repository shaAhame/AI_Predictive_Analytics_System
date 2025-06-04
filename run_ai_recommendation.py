import os
import pickle
from src.forecasting import (
    load_and_clean_data,
    prepare_monthly_log_data,
    train_prophet_model,
    make_forecast,
)
from src.profitability_checker import check_profitability
from src.ai_recommendation import generate_recommendation

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
    
    # Step 6: Generate recommendation message
    recommendation_msg = generate_recommendation(projected_profit)
    
    # Step 7: Print the results
    print("Forecast Status:", status)
    print("Projected Profit:", round(projected_profit, 2))
    print("\nBusiness Recommendation:")
    print(recommendation_msg)

    # Step 8: Save results as pickle file
    output_data = {
        "status": status,
        "projected_profit": projected_profit,
        "recommendation": recommendation_msg,
        "forecast_revenue": forecast_rev,
        "forecast_expense": forecast_exp
    }
    
    os.makedirs("pipelines", exist_ok=True)
    with open("pipelines/forecast_results.pkl", "wb") as f:
        pickle.dump(output_data, f)
    print("\nResults saved to pipelines/forecast_results.pkl")

if __name__ == "__main__":
    main()
