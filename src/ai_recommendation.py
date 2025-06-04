import os
import requests

def generate_recommendation(predicted_profit: float) -> str:
    if predicted_profit >= 0:
        return "✅ Business is on track. No immediate action is needed."

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        return "⚠️ Missing OpenRouter API key. Please set the 'OPENROUTER_API_KEY' environment variable."

    prompt = f"""
    The business is projected to make a loss of ${abs(predicted_profit):,.2f}.
    Suggest 3 specific and impactful ways to reduce loss and improve profitability
    in a retail business.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful business advisor AI."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()

        # Safe extraction
        content = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", None)
        )

        if content:
            return content.strip()
        else:
            return "⚠️ AI response format error: Missing content."

    except requests.exceptions.RequestException as e:
        return f"❌ Network/API error: {str(e)}"

