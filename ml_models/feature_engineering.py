import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self):
        self.log_cols = [
            "income_annum",
            "loan_amount",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["has_residential_assets_value"] = (
            df["residential_assets_value"] > 0
        ).astype(int)
        df["has_commercial_assets_value"] = (df["commercial_assets_value"] > 0).astype(
            int
        )
        df["has_luxury_assets_value"] = (df["luxury_assets_value"] > 0).astype(int)
        df["has_bank_asset_value"] = (df["bank_asset_value"] > 0).astype(int)

        for col in self.log_cols:
            df[f"{col}_log"] = np.log1p(df[col])

        df["debt_to_income_ratio"] = np.where(
            df["income_annum"] > 0,
            df["loan_amount"] / df["income_annum"],
            0,
        )

        df["total_asset_value"] = (
            df["residential_assets_value"]
            + df["commercial_assets_value"]
            + df["luxury_assets_value"]
            + df["bank_asset_value"]
        )

        df["loan_to_asset_ratio"] = np.where(
            df["total_asset_value"] > 0,
            df["loan_amount"] / df["total_asset_value"],
            0,
        )

        df["cibil_category_encoded"] = pd.cut(
            df["cibil_score"],
            bins=[300, 600, 750, 900],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype(int)

        return df
