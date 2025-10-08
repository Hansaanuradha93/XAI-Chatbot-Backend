# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, shap, numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
explainer = shap.Explainer(model)

class LoanInput(BaseModel):
    features: dict

@app.get("/health")
def health(): return {"ok": True}

@app.post("/predict")
def predict(inp: LoanInput):
    X = np.array([list(inp.features.values())])
    pred = model.predict(X)[0]
    return {"result": int(pred)}

@app.post("/predict_explain")
def predict_explain(inp: LoanInput):
    X = np.array([list(inp.features.values())])
    shap_vals = explainer(X)
    feature_names = list(inp.features.keys())
    top = sorted(zip(feature_names, shap_vals.values[0]),
                 key=lambda x: abs(x[1]), reverse=True)[:3]
    explanation = [{"feature": f, "impact": float(v)} for f, v in top]
    return {"result": int(model.predict(X)[0]),
            "explanation": explanation}
