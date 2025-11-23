import pandas as pd


class LoanPipeline:
    def __init__(self, engineer, predictor, shap, explainer):
        self.engineer = engineer
        self.predictor = predictor
        self.shap = shap
        self.explainer = explainer  # GPT generator

    def run(self, input_json, variant="xai"):
        df = pd.DataFrame([input_json])

        # Step 1 — engineered features
        df = self.engineer.transform(df)

        # Step 2 — prediction
        label, proba, scaled = self.predictor.predict(df)

        # Step 3 — SHAP explanation if XAI
        explanation = None
        if variant == "xai":
            explanation = self.shap.explain(scaled)

        # Step 4 — GPT human-friendly explanation
        human_message = (
            self.explainer(
                {
                    "loan_decision": label,
                    "probability": proba,
                    "explanation": explanation,
                }
            )
            if variant == "xai"
            else f"Your loan application has been {label.lower()}."
        )

        return {
            "prediction": label,
            "probability": round(proba, 3),
            "explanation": explanation,
            "human_message": human_message,
        }
