import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Revenue'] = df['Sales']
    df['Expense'] = df['Sales'] - df['Profit']
    return df

#feature_engineering

def prepare_monthly_log_data(df):
    monthly_df = df.groupby(pd.Grouper(key='Ship Date', freq='MS')).agg({
        'Revenue': 'sum',
        'Expense': 'sum'
    }).reset_index()

    # Remove outliers
    revenue_95 = monthly_df['Revenue'].quantile(0.95)
    expense_95 = monthly_df['Expense'].quantile(0.95)
    monthly_df = monthly_df[(monthly_df['Revenue'] < revenue_95) & (monthly_df['Expense'] < expense_95)]

    # Log transformation
    monthly_df['Log_Revenue'] = np.log1p(monthly_df['Revenue'])
    monthly_df['Log_Expense'] = np.log1p(monthly_df['Expense'])
    
    return monthly_df

# Model building


def train_prophet_model(df, column):
    prophet_df = df[['Ship Date', column]].rename(columns={'Ship Date': 'ds', column: 'y'})
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.1)
    model.fit(prophet_df)
    return model

def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    return forecast


#Model Evaluvation

def evaluate(actual_log, predicted_log):
    actual = np.expm1(actual_log)
    predicted = np.expm1(predicted_log)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return round(mae, 2), round(rmse, 2), round(mape, 2)


def merge_forecasts(rev_forecast, exp_forecast):
    df = pd.merge(
        rev_forecast[['ds', 'yhat']].rename(columns={'yhat': 'Log_Predicted_Revenue'}),
        exp_forecast[['ds', 'yhat']].rename(columns={'yhat': 'Log_Predicted_Expense'}),
        on='ds'
    )
    df['Predicted_Revenue'] = np.expm1(df['Log_Predicted_Revenue'])
    df['Predicted_Expense'] = np.expm1(df['Log_Predicted_Expense'])
    df['Predicted_Profit'] = df['Predicted_Revenue'] - df['Predicted_Expense']

    threshold = df['Predicted_Profit'].mean() - 2 * df['Predicted_Profit'].std()
    df['Status'] = df['Predicted_Profit'].apply(
        lambda x: "🔴 Anomaly (Loss Spike)" if x < threshold else ("✅ Business is doing fine" if x > 0 else "⚠️ Review costs or boost sales")
    )
    return df, threshold





