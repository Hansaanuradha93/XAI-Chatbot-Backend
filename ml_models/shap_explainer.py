import numpy as np


class ShapExplainer:
    def __init__(self, explainer, feature_order):
        self.explainer = explainer
        self.feature_order = feature_order

    def explain(self, scaled):
        shap_values = self.explainer.shap_values(scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        arr = np.array(shap_values).reshape(1, -1)

        return {
            col: float(round(val, 4)) for col, val in zip(self.feature_order, arr[0])
        }
