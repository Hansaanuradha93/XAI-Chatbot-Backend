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

# -------------------------------------------------
# 2️⃣ Load ML Model and Scaler (Loan Prediction)
# -------------------------------------------------
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


# -------------------------------------------------
# 3️⃣ Loan Prediction Schema
# -------------------------------------------------
class LoanApplication(BaseModel):
    education: int = Field(..., description="0 = Graduate, 1 = Not Graduate")
    self_employed: int = Field(..., description="1 = Yes, 0 = No")
    income_annum: float = Field(..., ge=0, description="Annual income (>=0)")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount (>=0)")
    loan_term: int = Field(..., ge=1, le=12, description="Repayment duration (1–12 months)")
    cibil_score: float = Field(..., ge=300, le=900, description="Credit score (300–900)")

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

    # Generate SHAP explanations
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
            # baseline: no SHAP work, no explanation
            feature_importance = None
    except Exception as e:
        feature_importance = {"error": f"SHAP explanation failed: {e}"} if variant == "xai" else None

    # --- Log prediction + explanation to Supabase ---
    if supabase:
        try:
            supabase.table("chat_history").insert([
                {
                    "user_email": getattr(data, "user_email", "anonymous"),
                    "message": f"💡 Loan Decision: {label}",
                    "sender": "bot",
                    "context": "loan",
                    "prediction": label,
                    "explanation_json": feature_importance if variant == "xai" else None,
                    "variant": variant,
                }
            ]).execute()
        except Exception as e:
            print(f"⚠️ Supabase chat_history insert failed: {e}")

    # Response payload
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

class FAQQuery(BaseModel):
    query: str
    user_email: Optional[str] = None

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
            # Safely extract user email or default to "anonymous"
            user_email = None
            if hasattr(payload, "user_email") and payload.user_email:
                user_email = payload.user_email
            else:
                user_email = "anonymous"

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
