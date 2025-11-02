import os
import json
from openai import OpenAI

# -------------------------------------------------
# Generate Humanized Explanation via GPT
# -------------------------------------------------

def generate_human_explanation(result):
    """Send the backend JSON result to GPT-5 for a human-readable summary."""
    system_message = (
        "You are a helpful financial assistant working for a digital bank. "
        "Your goal is to explain loan decisions in clear, human terms. "
        "Avoid technical jargon like 'features' or 'weights'. "
        "Use polite, confident, and professional language suitable for a customer. It should be in active voice and point form. "
        "Keep explanations concise (10–20 words) and focused on the most influential factors."
    )

    # ✅ Clean and summarize input before sending to GPT
    try:
        explanation = result.get("explanation", {})
        top_features = sorted(
            explanation.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        summary_data = {
            "loan_decision": result.get("loan_decision"),
            "probability": result.get("probability"),
            "top_features": top_features,
        }
    except Exception as e:
        print(f"⚠️ Failed to summarize features for GPT: {e}")
        summary_data = result

    user_prompt = f"""
    Here is the loan decision result from the model:
    {json.dumps(summary_data, indent=2)}

    Please rewrite this decision as a clear, human explanation for the customer.
    Include:
    1. The decision outcome and confidence.
    2. The main financial factors that influenced the decision.
    3. A short and polite conclusion encouraging responsible borrowing.
    """

    try:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=350,
            timeout=30,
        )

        message = response.choices[0].message.content.strip()

        # ✅ Fallback in case GPT returns empty
        if not message:
            print("⚠️ GPT returned an empty message. Using fallback text.")
            decision = result.get("loan_decision", "a decision")
            prob = result.get("probability", 0)
            message = (
                f"The system made a {decision.lower()} with confidence {prob*100:.1f}%. "
                "Your overall financial profile suggests stable income and reasonable asset strength. "
                "Please continue maintaining healthy credit habits."
            )

        return message

    except Exception as e:
        print(f"⚠️ GPT explanation failed: {e}")

        # ✅ Fallback response to prevent null in API response
        decision = result.get("loan_decision", "a decision")
        prob = result.get("probability", 0)
        return (
            f"The system made a {decision.lower()} with confidence {prob*100:.1f}%. "
            "Your financial details were analyzed carefully, and the result reflects your current standing."
        )
