import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
# 1️⃣ Basic App Config
# -------------------------------------------------

# Load environment variables from .env file
load_dotenv()
app = FastAPI(title="TrustAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model and Scaler (Loan Prediction)
from ml_models.loader import load_models
model, scaler, explainer, model_exp, scaler_exp, explainer_exp = load_models()

# Loan Prediction Schema
from ml_models.schemas import LoanApplication, FAQQuery

# Generate Humanized Explanation via GPT
from services.explanation_service import generate_human_explanation

@app.post("/loan_form_test")
def loan_form_test(request: Request, data: LoanApplication):
    """
    Predict loan decision using the experimental logistic model.
    Receives 11 raw user inputs, derives all engineered features,
    scales data, predicts, and returns SHAP explanation.
    """
    # Ensure experimental models exist
    if model_exp is None or scaler_exp is None or explainer_exp is None:
        raise HTTPException(status_code=500, detail="Experimental model not loaded.")

    # Convert input to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # -------------- Derive Additional Features ------------------

    # Binary asset indicators
    input_data["has_residential_assets_value"] = (input_data["residential_assets_value"] > 0).astype(int)
    input_data["has_commercial_assets_value"] = (input_data["commercial_assets_value"] > 0).astype(int)
    input_data["has_luxury_assets_value"] = (input_data["luxury_assets_value"] > 0).astype(int)
    input_data["has_bank_asset_value"] = (input_data["bank_asset_value"] > 0).astype(int)

    # Log-transformed versions (avoid log(0))
    for col in [
        "income_annum",
        "loan_amount",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]:
        input_data[f"{col}_log"] = np.log1p(input_data[col])

    # Financial ratios
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

    # CIBIL category encoding
    input_data["cibil_category_encoded"] = pd.cut(
        input_data["cibil_score"],
        bins=[300, 600, 750, 900],
        labels=[0, 1, 2],
        include_lowest=True,
    ).astype(int)

    # -------------- Align Column Order ------------------
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

    # Safety: add missing cols as 0
    for col in feature_order:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_order]

    # -------------- Scale & Predict ------------------
    try:
        scaled = scaler_exp.transform(input_data)
        prediction = int(model_exp.predict(scaled)[0])
        proba = round(float(model_exp.predict_proba(scaled)[0][1]), 3)
        label = "Approved" if prediction == 1 else "Rejected"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # -------------- SHAP Explanation ------------------
    try:
        shap_values = explainer_exp.shap_values(scaled)
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
    except Exception as e:
        feature_importance = {"error": f"SHAP explanation failed: {e}"}

    # --- Generate human explanation ---
    result = {
        "loan_decision": label,
        "probability": round(proba, 3),
        "explanation": feature_importance,
    }
    human_message = generate_human_explanation(result)

    # -------------- Return Structured Response ------------------
    return {
        "loan_decision": label,
        "probability": round(proba, 3),
        "explanation": feature_importance,
        "human_message": human_message,
    }

@app.post("/predict")
def predict_loan(request: Request, data: LoanApplication):
    """Predict loan approval and provide SHAP-based explanations."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model files not loaded properly.")

    # Determine mode (default 'xai' if not provided)
    variant = request.query_params.get("variant", "xai").lower()
    if variant not in ("xai", "baseline"):
        variant = "xai"

    # Convert input into DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Scale numeric features using saved scaler
    scaled = scaler.transform(input_data)

    # Predict loan status (1 = approved, 0 = rejected)
    prediction = int(model.predict(scaled)[0])
    label = "Approved" if prediction == 1 else "Rejected"

    # Generate SHAP explanations (only if XAI mode)
    feature_importance = {}  # <-- define early so it's always available

    try:
        if variant == "xai":
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
        else:
            # baseline: no SHAP explanation generated
            feature_importance = None
    except Exception as e:
        feature_importance = {"error": f"SHAP explanation failed: {e}"} if variant == "xai" else None

    # --- Log prediction + explanation to Supabase ---
    if supabase:
        try:
            # build the bot message depending on mode
            bot_message = f"Loan Decision: {label}"
            if variant == "xai" and feature_importance:
                expl_text = "\n".join([f"{k}: {v}" for k, v in feature_importance.items()])
                bot_message += f"\n\nExplanation:\n{expl_text}"

            supabase.table("chat_history").insert([
                {
                    "user_email": getattr(data, "user_email", "anonymous"),
                    "message": bot_message,
                    "sender": "bot",
                    "context": "loan",
                    "prediction": label,
                    "explanation_json": feature_importance if variant == "xai" else None,
                    "variant": variant,
                }
            ]).execute()
        except Exception as e:
            print(f"⚠️ Supabase chat_history insert failed: {e}")

    # ✅ Always return valid response
    return {
        "prediction": label,
        "explanation": feature_importance if variant == "xai" else None,
        "variant": variant,
    }

# -------------------------------------------------
# 4️⃣ Supabase Client Setup
# -------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase connected.")
else:
    print("⚠️ Supabase credentials not set. Logging disabled.")

# -------------------------------------------------
# 5️⃣ FAQ Semantic Search with GPT-5 Fallback
# -------------------------------------------------
FAQ_CSV = "data/faq/faq_cleaned.csv"
EMBED_PATH = "data/faq/faq_embeddings.npy"

if os.path.exists(FAQ_CSV) and os.path.exists(EMBED_PATH):
    print("📘 Loading preprocessed FAQ data...")
    faq_df = pd.read_csv(FAQ_CSV)
    faq_questions = faq_df["question"].astype(str).tolist()
    faq_answers = faq_df["answer"].astype(str).tolist()
    faq_classes = faq_df["class"].astype(str).tolist() if "class" in faq_df.columns else ["General"] * len(faq_df)
    faq_embeddings = np.load(EMBED_PATH)
    faq_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"✅ Loaded {len(faq_questions)} FAQ entries.")
else:
    faq_df, faq_questions, faq_answers, faq_classes, faq_embeddings, faq_model = None, [], [], [], None, None
    print("⚠️ FAQ data not found. Run scripts/prepare_faq.py first.")

SIM_THRESHOLD = 0.60

@app.post("/faq_answer")
def faq_answer(payload: FAQQuery):
    if faq_df is None or faq_embeddings is None:
        return {"answer": "FAQ data not loaded. Please run scripts/prepare_faq.py.", "source": "error"}

    query = payload.query.strip()
    if not query:
        return {"answer": None, "match": None, "class": None, "similarity": 0.0}

    query_emb = faq_model.encode([query], normalize_embeddings=True)
    sims = util.cos_sim(query_emb, faq_embeddings).cpu().numpy()[0]
    top_idx = int(np.argmax(sims))
    best_score = float(sims[top_idx])

    result = {}
    variant = "faq"
    matched_q = None
    src = "BankFAQs"

    if best_score >= SIM_THRESHOLD:
        result = {
            "answer": faq_answers[top_idx],
            "match": faq_questions[top_idx],
            "class": faq_classes[top_idx],
            "similarity": round(best_score, 3),
            "source": src
        }
        matched_q = faq_questions[top_idx]
    else:
        # GPT fallback
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        variant = "gpt-fallback"

        if not OPENAI_API_KEY:
            result = {
                "answer": "I couldn’t find this in our FAQ database. Please consult our support team.",
                "source": "local-fallback"
            }
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "You are a helpful financial assistant."},
                        {"role": "user", "content": f"Question: {query}"}
                    ]
                )

                gpt_answer = resp.choices[0].message.content
                result = {
                    "answer": gpt_answer,
                    "match": None,
                    "class": None,
                    "similarity": round(best_score, 3),
                    "source": "gpt-5-fallback"
                }
            except Exception as e:
                result = {"answer": None, "error": str(e)}

    # --- 🧾 Log interaction to Supabase ---
    if supabase:
        try:
            user_email = payload.user_email or "anonymous"

            supabase.table("faq_logs").insert({
                "user_email": user_email,
                "query": query,
                "matched_question": matched_q,
                "similarity": round(best_score, 3),
                "answer_source": result.get("source", "unknown"),
                "variant": variant,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            print(f"⚠️ Supabase log insert failed: {e}")

    return result

# -------------------------------------------------
# 6️⃣ Health Check Endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "TrustAI backend is running!"}

# --- Run with: uvicorn app:app --reload ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
