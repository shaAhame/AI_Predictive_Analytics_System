#profitability_checker

import numpy as np

def check_profitability(forecast_revenue, forecast_expense):
    """
    Compare the last predicted revenue and expense to assess future profitability.

    Args:
        forecast_revenue (DataFrame): Prophet forecast for log-transformed revenue.
        forecast_expense (DataFrame): Prophet forecast for log-transformed expense.

    Returns:
        str: Profitability status.
        float: Forecasted profit value.
    """
    last_log_rev = forecast_revenue['yhat'].iloc[-1]
    last_log_exp = forecast_expense['yhat'].iloc[-1]

    last_rev = np.expm1(last_log_rev)
    last_exp = np.expm1(last_log_exp)

    profit = last_rev - last_exp

    if profit > 0:
        return "✅ Future OK", round(profit, 2)
    else:
        return "🔴 Loss Predicted", round(profit, 2)
