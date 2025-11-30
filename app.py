import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from core.config import APP_TITLE

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
# 1️⃣ Basic App Config
# -------------------------------------------------

# Load environment variables from .env file
load_dotenv()
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xai-chatbot.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model and Scaler (Loan Prediction)
from ml_models.loader import load_models

model, scaler, explainer = load_models()

# Loan Prediction Schema
from ml_models.schemas import LoanApplication

# Generate Humanized Explanation via GPT
from services.explanation_service import generate_human_explanation

# Supabase Client Setup
from core.supabase_client import supabase

import traceback

from ml_models.feature_engineering import FeatureEngineer
from ml_models.predictor import LoanPredictor
from ml_models.shap_explainer import ShapExplainer
from ml_models.loan_pipeline import LoanPipeline

engineer = FeatureEngineer()

feature_order = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "has_residential_assets_value",
    "has_commercial_assets_value",
    "has_luxury_assets_value",
    "has_bank_asset_value",
    "income_annum_log",
    "loan_amount_log",
    "residential_assets_value_log",
    "commercial_assets_value_log",
    "luxury_assets_value_log",
    "bank_asset_value_log",
    "debt_to_income_ratio",
    "total_asset_value",
    "loan_to_asset_ratio",
    "cibil_category_encoded",
]

predictor = LoanPredictor(model, scaler, feature_order)
shap = ShapExplainer(explainer, feature_order)
pipeline = LoanPipeline(engineer, predictor, shap, generate_human_explanation)


@app.post("/loan_form_test")
def loan_form_test(request: Request, data: LoanApplication):
    variant = request.query_params.get("variant", "xai")
    return pipeline.run(data.dict(), variant)
    """
    Predict loan decision using the experimental logistic model.
    Receives 11 raw user inputs, derives all engineered features,
    scales data, predicts, and returns SHAP explanation.
    Includes detailed logs for debugging.
    """
    print("📩 [loan_form_test] Request received.")
    print(f"🔹 Raw input: {data.dict()}")

    # --- Model existence check ---
    if model is None or scaler is None or explainer is None:
        print("❌ Experimental model files not loaded properly.")
        raise HTTPException(status_code=500, detail="Experimental model not loaded.")

    # --- Mode selection ---
    variant = request.query_params.get("variant", "xai").lower()
    if variant not in ("xai", "baseline"):
        variant = "xai"
    print(f"⚙️ Mode selected: {variant}")

    # --- DataFrame conversion ---
    try:
        input_data = pd.DataFrame([data.dict()])
        print("✅ Converted input to DataFrame.")
    except Exception as e:
        print(f"❌ Failed converting to DataFrame: {e}")
        raise HTTPException(status_code=500, detail=f"Data conversion error: {e}")

    # --- Derive engineered features ---
    try:
        print("🧮 Deriving engineered features...")
        input_data["has_residential_assets_value"] = (
            input_data["residential_assets_value"] > 0
        ).astype(int)
        input_data["has_commercial_assets_value"] = (
            input_data["commercial_assets_value"] > 0
        ).astype(int)
        input_data["has_luxury_assets_value"] = (
            input_data["luxury_assets_value"] > 0
        ).astype(int)
        input_data["has_bank_asset_value"] = (
            input_data["bank_asset_value"] > 0
        ).astype(int)

        for col in [
            "income_annum",
            "loan_amount",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]:
            input_data[f"{col}_log"] = np.log1p(input_data[col])

        input_data["debt_to_income_ratio"] = np.where(
            input_data["income_annum"] > 0,
            input_data["loan_amount"] / input_data["income_annum"],
            0,
        )

        input_data["total_asset_value"] = (
            input_data["residential_assets_value"]
            + input_data["commercial_assets_value"]
            + input_data["luxury_assets_value"]
            + input_data["bank_asset_value"]
        )

        input_data["loan_to_asset_ratio"] = np.where(
            input_data["total_asset_value"] > 0,
            input_data["loan_amount"] / input_data["total_asset_value"],
            0,
        )

        input_data["cibil_category_encoded"] = pd.cut(
            input_data["cibil_score"],
            bins=[300, 600, 750, 900],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype(int)
        print("✅ Feature engineering complete.")
    except Exception as e:
        print("❌ Feature engineering failed:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

    # --- Column alignment ---
    feature_order = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "has_residential_assets_value",
        "has_commercial_assets_value",
        "has_luxury_assets_value",
        "has_bank_asset_value",
        "income_annum_log",
        "loan_amount_log",
        "residential_assets_value_log",
        "commercial_assets_value_log",
        "luxury_assets_value_log",
        "bank_asset_value_log",
        "debt_to_income_ratio",
        "total_asset_value",
        "loan_to_asset_ratio",
        "cibil_category_encoded",
    ]

    for col in feature_order:
        if col not in input_data.columns:
            print(f"⚠️ Missing column added as 0: {col}")
            input_data[col] = 0

    input_data = input_data[feature_order]
    print(f"✅ Final feature set columns: {list(input_data.columns)}")

    # --- Prediction ---
    try:
        print("🔮 Running model prediction...")
        scaled = scaler.transform(input_data)
        prediction = int(model.predict(scaled)[0])
        proba = round(float(model.predict_proba(scaled)[0][1]), 3)
        label = "Approved" if prediction == 1 else "Rejected"
        print(f"✅ Prediction complete → {label} (prob={proba})")
    except Exception as e:
        print("❌ Model prediction failed:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # --- SHAP Explanation ---
    feature_importance = None
    if variant == "xai":
        try:
            print("📊 Generating SHAP explanation...")
            shap_values = explainer.shap_values(scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
            shap_array = np.array(shap_values)
            if shap_array.ndim == 3:
                shap_array = shap_array[:, 0, :]
            elif shap_array.ndim == 1:
                shap_array = shap_array.reshape(1, -1)

            feature_importance = {
                str(col): round(float(val), 4)
                for col, val in zip(feature_order, shap_array[0])
            }
            print("✅ SHAP explanation generated successfully.")
        except Exception as e:
            print("❌ SHAP explanation failed:", traceback.format_exc())
            feature_importance = {"error": f"SHAP explanation failed: {e}"}

    # --- Human explanation generation ---
    try:
        print("🗣 Generating human explanation...")
        if variant == "xai":
            result = {
                "loan_decision": label,
                "probability": proba,
                "explanation": feature_importance,
            }
            human_message = generate_human_explanation(result)
        else:
            human_message = f"Your loan application has been {label.lower()}."
        print("✅ Human explanation generated.")
    except Exception as e:
        print("⚠️ Human explanation generation failed:", traceback.format_exc())
        human_message = f"Your loan application has been {label.lower()}."

    # --- Final response ---
    print("🚀 Returning final response to client.")
    return {
        "prediction": label,
        "probability": proba,
        "explanation": feature_importance,
        "human_message": human_message,
    }

    variant = request.query_params.get("variant", "xai")
    return pipeline.run(data.dict(), variant)
    """
    Predict loan decision using the experimental logistic model.
    Receives 11 raw user inputs, derives all engineered features,
    scales data, predicts, and returns SHAP explanation.
    Includes detailed logs for debugging.
    """
    print("📩 [loan_form_test] Request received.")
    print(f"🔹 Raw input: {data.dict()}")

    # --- Model existence check ---
    if model is None or scaler is None or explainer is None:
        print("❌ Experimental model files not loaded properly.")
        raise HTTPException(status_code=500, detail="Experimental model not loaded.")

    # --- Mode selection ---
    variant = request.query_params.get("variant", "xai").lower()
    if variant not in ("xai", "baseline"):
        variant = "xai"
    print(f"⚙️ Mode selected: {variant}")

    # --- DataFrame conversion ---
    try:
        input_data = pd.DataFrame([data.dict()])
        print("✅ Converted input to DataFrame.")
    except Exception as e:
        print(f"❌ Failed converting to DataFrame: {e}")
        raise HTTPException(status_code=500, detail=f"Data conversion error: {e}")

    # --- Derive engineered features ---
    try:
        print("🧮 Deriving engineered features...")
        input_data["has_residential_assets_value"] = (
            input_data["residential_assets_value"] > 0
        ).astype(int)
        input_data["has_commercial_assets_value"] = (
            input_data["commercial_assets_value"] > 0
        ).astype(int)
        input_data["has_luxury_assets_value"] = (
            input_data["luxury_assets_value"] > 0
        ).astype(int)
        input_data["has_bank_asset_value"] = (
            input_data["bank_asset_value"] > 0
        ).astype(int)

        for col in [
            "income_annum",
            "loan_amount",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]:
            input_data[f"{col}_log"] = np.log1p(input_data[col])

        input_data["debt_to_income_ratio"] = np.where(
            input_data["income_annum"] > 0,
            input_data["loan_amount"] / input_data["income_annum"],
            0,
        )

        input_data["total_asset_value"] = (
            input_data["residential_assets_value"]
            + input_data["commercial_assets_value"]
            + input_data["luxury_assets_value"]
            + input_data["bank_asset_value"]
        )

        input_data["loan_to_asset_ratio"] = np.where(
            input_data["total_asset_value"] > 0,
            input_data["loan_amount"] / input_data["total_asset_value"],
            0,
        )

        input_data["cibil_category_encoded"] = pd.cut(
            input_data["cibil_score"],
            bins=[300, 600, 750, 900],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype(int)
        print("✅ Feature engineering complete.")
    except Exception as e:
        print("❌ Feature engineering failed:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

    # --- Column alignment ---
    feature_order = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "has_residential_assets_value",
        "has_commercial_assets_value",
        "has_luxury_assets_value",
        "has_bank_asset_value",
        "income_annum_log",
        "loan_amount_log",
        "residential_assets_value_log",
        "commercial_assets_value_log",
        "luxury_assets_value_log",
        "bank_asset_value_log",
        "debt_to_income_ratio",
        "total_asset_value",
        "loan_to_asset_ratio",
        "cibil_category_encoded",
    ]

    for col in feature_order:
        if col not in input_data.columns:
            print(f"⚠️ Missing column added as 0: {col}")
            input_data[col] = 0

    input_data = input_data[feature_order]
    print(f"✅ Final feature set columns: {list(input_data.columns)}")

    # --- Prediction ---
    try:
        print("🔮 Running model prediction...")
        scaled = scaler.transform(input_data)
        prediction = int(model.predict(scaled)[0])
        proba = round(float(model.predict_proba(scaled)[0][1]), 3)
        label = "Approved" if prediction == 1 else "Rejected"
        print(f"✅ Prediction complete → {label} (prob={proba})")
    except Exception as e:
        print("❌ Model prediction failed:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # --- SHAP Explanation ---
    feature_importance = None
    if variant == "xai":
        try:
            print("📊 Generating SHAP explanation...")
            shap_values = explainer.shap_values(scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
            shap_array = np.array(shap_values)
            if shap_array.ndim == 3:
                shap_array = shap_array[:, 0, :]
            elif shap_array.ndim == 1:
                shap_array = shap_array.reshape(1, -1)

            feature_importance = {
                str(col): round(float(val), 4)
                for col, val in zip(feature_order, shap_array[0])
            }
            print("✅ SHAP explanation generated successfully.")
        except Exception as e:
            print("❌ SHAP explanation failed:", traceback.format_exc())
            feature_importance = {"error": f"SHAP explanation failed: {e}"}

    # --- Human explanation generation ---
    try:
        print("🗣 Generating human explanation...")
        if variant == "xai":
            result = {
                "loan_decision": label,
                "probability": proba,
                "explanation": feature_importance,
            }
            human_message = generate_human_explanation(result)
        else:
            human_message = f"Your loan application has been {label.lower()}."
        print("✅ Human explanation generated.")
    except Exception as e:
        print("⚠️ Human explanation generation failed:", traceback.format_exc())
        human_message = f"Your loan application has been {label.lower()}."

    # --- Final response ---
    print("🚀 Returning final response to client.")
    return {
        "prediction": label,
        "probability": proba,
        "explanation": feature_importance,
        "human_message": human_message,
    }


# Health Check Endpoint
@app.get("/")
def root():
    return {"message": "TrustAI backend is running!"}


# --- Run with: uvicorn app:app --reload ---
if __name__ == "__main__":
    import uvicorn, os

    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=5000)
