import joblib
import os
import traceback

def safe_load(path, label):
    print(f"🔍 Attempting to load {label} → {path}")
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        return None
    try:
        obj = joblib.load(path)
        print(f"✅ Loaded {label} successfully ({type(obj).__name__})")
        return obj
    except Exception as e:
        print(f"❌ Failed to load {label}: {e}")
        print(traceback.format_exc())
        return None

def load_models():
    print("📦 [loader] Initializing model loading sequence...")
    base_path = "./artifacts/main"
    exp_path = "./artifacts/experimental"

    # Show directory contents
    for p in [base_path, exp_path]:
        if os.path.exists(p):
            print(f"📁 Contents of {p}: {os.listdir(p)}")
        else:
            print(f"⚠️ Directory not found: {p}")

    model = safe_load(os.path.join(base_path, "model.pkl"), "main model")
    scaler = safe_load(os.path.join(base_path, "preprocess.pkl"), "main scaler")
    explainer = safe_load(os.path.join(base_path, "explainer.pkl"), "main explainer")

    model_exp = safe_load(os.path.join(exp_path, "model.pkl"), "experimental model")
    scaler_exp = safe_load(os.path.join(exp_path, "preprocess.pkl"), "experimental scaler")
    explainer_exp = safe_load(os.path.join(exp_path, "explainer.pkl"), "experimental explainer")

    print("✅ [loader] Model loading complete.")
    return model, scaler, explainer, model_exp, scaler_exp, explainer_exp
