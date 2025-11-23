import os
import json
from openai import OpenAI


# ======================================================
# OOAD COMPONENT 1 — Summarizer (Single Responsibility)
# ======================================================
class ExplanationSummarizer:
    """Extracts and summarizes the top SHAP features for the GPT prompt."""

    def summarize(self, result: dict) -> dict:
        explanation = result.get("explanation", {})

        try:
            # Sort by absolute SHAP impact (largest first)
            top_features = sorted(
                explanation.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
        except Exception:
            top_features = []

        return {
            "loan_decision": result.get("loan_decision"),
            "probability": result.get("probability"),
            "top_features": top_features,
        }


# ======================================================
# OOAD COMPONENT 2 — GPT Client (OpenAI Dependency)
# ======================================================
class GPTExplanationClient:
    """Handles communication with GPT (OpenAI API)."""

    def __init__(self):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.system_message = (
            "You are a helpful financial assistant working for a digital bank. "
            "Your goal is to explain loan decisions in clear, human terms. "
            "Avoid technical jargon like 'features' or 'weights'. "
            "Use polite, confident, and professional language suitable for a customer. "
            "Use active voice and bullet points. "
            "Keep explanations concise (10–20 words per bullet). "
            "Focus only on the most influential financial factors."
        )

    def generate(self, summary_data: dict) -> str:
        """Send the structured summary to GPT for a human explanation."""

        prompt = f"""
        Here is the loan decision information:
        {json.dumps(summary_data, indent=2)}

        Please write a clear, human-friendly explanation including:
        1. The decision outcome & confidence level.
        2. The main factors influencing the decision.
        3. A polite closing remark about responsible borrowing.
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
            timeout=30,
        )

        msg = response.choices[0].message.content.strip()
        return msg


# ======================================================
# OOAD COMPONENT 3 — High-Level Service
# ======================================================
class ExplanationService:
    """Coordinates summarization + GPT explanation with fallback."""

    def __init__(self):
        self.summarizer = ExplanationSummarizer()
        self.gpt_client = GPTExplanationClient()

    def explain(self, result: dict) -> str:
        """Main entry: create human-friendly explanation."""

        try:
            summary = self.summarizer.summarize(result)
        except Exception:
            summary = result

        try:
            message = self.gpt_client.generate(summary)
            if message:
                return message
        except Exception as e:
            print(f"⚠️ GPT explanation failed: {e}")

        # Fallback
        decision = result.get("loan_decision", "a loan decision")
        prob = result.get("probability", 0)
        return (
            f"The system made a {decision.lower()} with confidence {prob*100:.1f}%. "
            "Your financial profile was carefully assessed."
        )


# ======================================================
# PUBLIC API (unchanged)
# ======================================================
def generate_human_explanation(result: dict) -> str:
    """
    Public function used by your loan pipeline.
    Uses an OO-based internal architecture.
    """
    service = ExplanationService()
    return service.explain(result)
