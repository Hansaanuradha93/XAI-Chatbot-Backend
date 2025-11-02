import joblib
from core.config import (
    MODEL_PATH,
    SCALER_PATH,
    EXPLAINER_PATH,
    MODEL_PATH_EXPERIMENTAL,
    SCALER_PATH_EXPERIMENTAL,
    EXPLAINER_PATH_EXPERIMENTAL,
)

def load_models():
    """
    Load both the main (production) model set and the experimental model set.
    Returns:
        model, scaler, explainer,
        model_exp, scaler_exp, explainer_exp
    """
    model = scaler = explainer = None
    model_exp = scaler_exp = explainer_exp = None

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        explainer = joblib.load(EXPLAINER_PATH)
        print("✅ Model, Scaler, and Explainer loaded successfully.")

        model_exp = joblib.load(MODEL_PATH_EXPERIMENTAL)
        scaler_exp = joblib.load(SCALER_PATH_EXPERIMENTAL)
        explainer_exp = joblib.load(EXPLAINER_PATH_EXPERIMENTAL)
        print("🧪 Experimental model set loaded successfully")
    except Exception as e:
        # We keep the same behavior as current app.py: swallow and just print.
        print(f"❌ Error loading model files: {e}")

    return model, scaler, explainer, model_exp, scaler_exp, explainer_exp
