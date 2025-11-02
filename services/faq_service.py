import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from openai import OpenAI
from core.supabase_client import supabase

# FAQ Semantic Search with GPT-5 Fallback
from core.config import FAQ_CSV, EMBED_PATH, SIM_THRESHOLD, GPT_MODEL_FAQ_FALLBACK

faq_df, faq_questions, faq_answers, faq_classes, faq_embeddings, faq_model = None, [], [], [], None, None

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
    print("⚠️ FAQ data not found. Run scripts/prepare_faq.py first.")

def answer_faq(query: str, user_email: str = "anonymous"):
    """Answer FAQ questions using semantic similarity and GPT fallback."""
    if faq_df is None or faq_embeddings is None:
        return {"answer": "FAQ data not loaded. Please run scripts/prepare_faq.py.", "source": "error"}

    query = query.strip()
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
            "source": src,
        }
        matched_q = faq_questions[top_idx]
    else:
        # GPT fallback
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        variant = "gpt-fallback"

        if not OPENAI_API_KEY:
            result = {
                "answer": "I couldn’t find this in our FAQ database. Please consult our support team.",
                "source": "local-fallback",
            }
        else:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=GPT_MODEL_FAQ_FALLBACK,
                    messages=[
                        {"role": "system", "content": "You are a helpful financial assistant."},
                        {"role": "user", "content": f"Question: {query}"},
                    ],
                )
                gpt_answer = resp.choices[0].message.content
                result = {
                    "answer": gpt_answer,
                    "match": None,
                    "class": None,
                    "similarity": round(best_score, 3),
                    "source": "gpt-5-fallback",
                }
            except Exception as e:
                result = {"answer": None, "error": str(e)}

    # --- 🧾 Log interaction to Supabase ---
    if supabase:
        try:
            supabase.table("faq_logs").insert({
                "user_email": user_email,
                "query": query,
                "matched_question": matched_q,
                "similarity": round(best_score, 3),
                "answer_source": result.get("source", "unknown"),
                "variant": variant,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            print(f"⚠️ Supabase log insert failed: {e}")

    return result
