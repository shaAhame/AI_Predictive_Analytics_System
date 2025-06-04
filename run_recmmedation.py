import pickle
from src.ai_recommendation import generate_recommendation

# Save the pipeline 
with open("pipelines/recommendation_pipeline.pkl", "wb") as f:
    pickle.dump(generate_recommendation, f)

# Load predicted profit from Prophet or any model
with open("outputs/prophet_forecast.pkl", "rb") as f:
    forecast_data = pickle.load(f)

# Assume last predicted profit (customize this part as needed)
predicted_profit = forecast_data['yhat'].iloc[-1] - forecast_data.get('cost', 0)

# Generate recommendation
recommendation = generate_recommendation(predicted_profit)

# Save recommendation to file
with open("outputs/recommendations.txt", "w") as f:
    f.write(recommendation)
