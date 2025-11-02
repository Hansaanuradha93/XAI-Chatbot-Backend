import os
from dotenv import load_dotenv

# -------------------------------------------------
# 🧩 Global Configuration for TrustAI Backend
# -------------------------------------------------

load_dotenv()

# --- Model Paths ---
MODEL_PATH = "./artifacts/main/model.pkl"
SCALER_PATH = "./artifacts/main/preprocess.pkl"
EXPLAINER_PATH = "./artifacts/main/explainer.pkl"

# Experimental Models
MODEL_PATH_EXPERIMENTAL = "./artifacts/experimental/model.pkl"
SCALER_PATH_EXPERIMENTAL = "./artifacts/experimental/preprocess.pkl"
EXPLAINER_PATH_EXPERIMENTAL = "./artifacts/experimental/explainer.pkl"

# --- FAQ Assets ---
FAQ_CSV = "data/faq/faq_cleaned.csv"
EMBED_PATH = "data/faq/faq_embeddings.npy"

# --- Similarity Threshold ---
SIM_THRESHOLD = 0.60

# --- OpenAI Model Names ---
GPT_MODEL_HUMANIZE = "gpt-4-turbo"
GPT_MODEL_FAQ_FALLBACK = "gpt-5"

# --- Application Metadata ---
APP_TITLE = "TrustAI Backend"
APP_HOST = "127.0.0.1"
APP_PORT = 8000
