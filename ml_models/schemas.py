from pydantic import BaseModel, Field
from typing import Optional

# Loan Prediction Schema
class LoanApplication(BaseModel):
    """Loan form fields provided by the user."""
    no_of_dependents: int = Field(..., description="Number of dependents")
    education: int = Field(..., description="0 = Graduate, 1 = Not Graduate")
    self_employed: int = Field(..., description="1 = Yes, 0 = No")
    income_annum: float = Field(..., ge=0, description="Annual income (>=0)")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount (>=0)")
    loan_term: int = Field(..., ge=1, le=12, description="Repayment duration (1–12 months)")
    cibil_score: float = Field(..., ge=300, le=900, description="Credit score (300–900)")
    residential_assets_value: float = Field(0, ge=0, description="Value of residential properties")
    commercial_assets_value: float = Field(0, ge=0, description="Value of commercial properties")
    luxury_assets_value: float = Field(0, ge=0, description="Value of luxury assets")
    bank_asset_value: float = Field(0, ge=0, description="Total bank deposits")


# FAQ Query Schema
class FAQQuery(BaseModel):
    """FAQ query request schema for semantic search and GPT fallback."""
    query: str
    user_email: Optional[str] = None
