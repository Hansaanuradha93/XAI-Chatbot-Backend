from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import shap

# --- Initialize FastAPI app ---
app = FastAPI(title="TrustAI Loan Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js local dev
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load your model, scaler, and explainer ---
MODEL_PATH = "./models/model.pkl"
SCALER_PATH = "./models/preprocess.pkl"
EXPLAINER_PATH = "./models/explainer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    print("✅ Model, Scaler, and Explainer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    model = scaler = explainer = None


# --- Input schema for loan applications ---
class LoanApplication(BaseModel):
    education: int = Field(..., description="0 = Graduate, 1 = Not Graduate")
    self_employed: int = Field(..., description="1 = Yes, 0 = No")
    income_annum: float = Field(..., ge=0, description="Annual income (>=0)")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount (>=0)")
    loan_term: int = Field(..., ge=1, le=12, description="Repayment duration (1–12 months)")
    cibil_score: float = Field(..., ge=300, le=900, description="Credit score (300–900)")


@app.get("/")
def home():
    return {"message": "🚀 TrustAI Loan Prediction API is running!"}


@app.post("/predict")
def predict_loan(data: LoanApplication):
    """Predict loan approval and provide SHAP-based explanations."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model files not loaded properly.")

    # Convert input into DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Scale numeric features using saved scaler
    scaled = scaler.transform(input_data)

    # Predict loan status (1 = approved, 0 = rejected)
    prediction = int(model.predict(scaled)[0])
    label = "Approved" if prediction == 1 else "Rejected"

    # Generate SHAP explanations
    feature_importance = {}  # <-- define early so it's always available

    try:
        shap_values = explainer.shap_values(scaled)

        # ✅ Handle different SHAP output structures
        if isinstance(shap_values, list):  # For TreeExplainer (list per class)
            shap_values = shap_values[1]  # positive class (Approved)

        shap_array = np.array(shap_values)

        # Normalize shape
        if shap_array.ndim == 3:
            shap_array = shap_array[:, 0, :]
        elif shap_array.ndim == 1:
            shap_array = shap_array.reshape(1, -1)

        # Convert to JSON-safe dict
        feature_importance = {
            str(col): round(float(val), 4)
            for col, val in zip(input_data.columns, shap_array[0])
            }

    except Exception as e:
        feature_importance = {"error": f"SHAP explanation failed: {e}"}

    # Response payload
    return {
        "prediction": label,
        "explanation": feature_importance,
    }


# --- Run with: uvicorn app:app --reload ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
