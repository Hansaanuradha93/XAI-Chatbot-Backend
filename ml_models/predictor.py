class LoanPredictor:
    def __init__(self, model, scaler, feature_order):
        self.model = model
        self.scaler = scaler
        self.feature_order = feature_order

    def predict(self, df):
        aligned = df[self.feature_order]
        scaled = self.scaler.transform(aligned)

        pred = int(self.model.predict(scaled)[0])
        proba = float(self.model.predict_proba(scaled)[0][1])
        label = "Approved" if pred == 1 else "Rejected"

        return label, proba, scaled
