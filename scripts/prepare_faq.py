"""
scripts/prepare_faq.py
Purpose:
  - Clean the raw FAQ CSV (bankfaqs.csv)
  - Standardize column names
  - Generate embeddings using SentenceTransformer
  - Save the cleaned CSV + embeddings to disk
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

RAW_PATH = Path("bankfaqs.csv")
OUT_DIR = Path("data/faq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("📘 Loading dataset...")
df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip().str.lower()

# Basic validation
if not {"question", "answer"}.issubset(df.columns):
    raise ValueError("CSV must contain 'Question' and 'Answer' columns.")

# Drop duplicates and clean text
df = df.dropna(subset=["question", "answer"])
df = df.drop_duplicates(subset=["question"])
df["question"] = df["question"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()
df["class"] = df["class"].fillna("General")

# Save cleaned CSV
clean_path = OUT_DIR / "faq_cleaned.csv"
df.to_csv(clean_path, index=False)
print(f"✅ Cleaned data saved to {clean_path}")

# --- Generate embeddings ---
print("🔹 Encoding FAQ questions (this may take a minute)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df["question"].tolist(), normalize_embeddings=True)
embeddings_path = OUT_DIR / "faq_embeddings.npy"

# Save embeddings to disk
np.save(embeddings_path, embeddings)
print(f"✅ Embeddings saved to {embeddings_path}")
print(f"Total questions encoded: {len(df)}")
