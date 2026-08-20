"""
FastAPI scoring endpoint -- the deployable interface for the champion PD
model, wired through the same decision threshold and compliant
explainability pipeline built in Phases 4-5.

Local run:  uvicorn api:app --reload --app-dir src
Docs at:    http://127.0.0.1:8000/docs

Deployment: see DEPLOYMENT.md in the project root. Reads $PORT from the
environment (set automatically by Render/Railway/Fly.io) and allows CORS
from any origin by default -- fine for a public portfolio demo scoring
synthetic/example applicants; tighten CORS_ALLOW_ORIGINS if this is ever
used for anything sensitive.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

import features
import explainability as ex

THRESHOLD = 0.430  # the validated, real-dollar profit-maximizing threshold (Phase 4)

# Business-rule overlay, separate from the statistical model: LendingClub's
# own "dti" field reflects EXISTING debt only, by definition -- it never
# includes the loan currently being requested. That means nothing in the
# model's 19 features directly checks whether THIS SPECIFIC loan's payment
# is affordable against THIS SPECIFIC income. A logistic regression summing
# many "looks fine" signals can approve a request where the new payment
# alone would exceed the applicant's entire income, if other features
# outweigh it (found via a live test: $2,000 income, still approved).
# Real underwriting commonly layers hard affordability rules like this one
# on top of a statistical score for exactly this reason.
PAYMENT_TO_INCOME_CEILING = 0.50
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "data", "processed", "champion_model_calibrated.joblib")

# Comma-separated list of allowed origins, e.g. "https://yourname.github.io".
# Defaults to "*" (any origin) -- appropriate for a public demo endpoint
# that only scores example applicant data, not real customer information.
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "*")

FEATURE_COLS = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_years", "fico_midpoint",
]

# Standard, publicly documented LendingClub category taxonomy -- these are
# the values the champion model's OneHotEncoder actually learned during
# training. Deliberately NOT left as free-text `str` fields: found via a
# live test that OneHotEncoder(handle_unknown="ignore") does not error on
# an unrecognized category -- it silently zeroes it out and still returns
# a confident-looking prediction for input the model never learned to
# interpret (confirmed with a garbage grade value and an XSS-style string
# in purpose; both produced a normal-looking 0.0000 PD with no error at
# all). That's a worse failure mode than crashing, since nothing signals
# anything went wrong. Constraining these to Literal types rejects anything
# outside the real, trained category set with a clear validation error
# instead.
_SUB_GRADES = tuple(f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6))
_EMP_LENGTHS = ("< 1 year", "1 year", *[f"{n} years" for n in range(2, 10)], "10+ years")
Grade = Literal["A", "B", "C", "D", "E", "F", "G"]
SubGrade = Literal[_SUB_GRADES]
HomeOwnership = Literal["RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"]
VerificationStatus = Literal["Verified", "Source Verified", "Not Verified"]
Purpose = Literal[
    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house",
    "wedding", "renewable_energy", "educational", "other",
]
EmpLength = Literal[_EMP_LENGTHS]

app = FastAPI(
    title="Credit Expansion Engine -- Scoring API",
    description="PD scoring with a validated decision threshold and compliant, "
                 "SHAP-driven adverse-action explanations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ALLOW_ORIGINS] if CORS_ALLOW_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_champion = None


def get_champion():
    """Lazy-load the model on first request rather than at import time --
    keeps module import fast and testable without a real model file present,
    and avoids a slow cold-start on every deployment health check."""
    global _champion
    if _champion is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at {MODEL_PATH}. Ensure "
                       f"data/processed/champion_model_calibrated.joblib is included "
                       f"in the deployed repository.",
            )
        _champion = joblib.load(MODEL_PATH)
    return _champion


class Applicant(BaseModel):
    loan_amnt: float = Field(..., gt=0, le=40000, description="Requested loan amount (LendingClub's real range tops out around $40,000 — the model has no basis to extrapolate meaningfully beyond that)")
    int_rate: float = Field(..., ge=0, le=40, description="Contractual APR, percent")
    installment: float = Field(..., gt=0, le=2000, description="Monthly payment")
    grade: Grade
    sub_grade: SubGrade
    home_ownership: HomeOwnership
    annual_inc: float = Field(..., ge=0, le=1000000, description="Self-reported gross annual income")
    verification_status: VerificationStatus
    purpose: Purpose
    dti: float = Field(..., ge=0, le=100)
    delinq_2yrs: int = Field(..., ge=0, le=20)
    inq_last_6mths: int = Field(..., ge=0, le=20)
    open_acc: int = Field(..., ge=0, le=50)
    pub_rec: int = Field(..., ge=0, le=10)
    revol_bal: float = Field(..., ge=0, le=500000)
    revol_util: float = Field(..., ge=0, le=150)
    total_acc: int = Field(..., ge=0, le=100)
    emp_length: EmpLength = Field(..., description="e.g. '< 1 year', '5 years', '10+ years'")
    fico_range_low: float = Field(..., ge=300, le=850)
    fico_range_high: float = Field(..., ge=300, le=850)

    @model_validator(mode="after")
    def _fico_range_is_logically_ordered(self):
        # The frontend always computes fico_range_high = low + 4 and can
        # never produce an inverted pair -- but the API is public, and
        # anyone calling it directly could send fico_range_high < low,
        # which is logically backwards (LendingClub always reports these
        # as an ordered low-high pair). Caught here rather than silently
        # accepted and fed into feature engineering as nonsense.
        if self.fico_range_high < self.fico_range_low:
            raise ValueError(
                f"fico_range_high ({self.fico_range_high}) cannot be less than "
                f"fico_range_low ({self.fico_range_low})"
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 12000, "int_rate": 13.5, "installment": 407.23,
                "grade": "C", "sub_grade": "C3", "home_ownership": "RENT",
                "annual_inc": 55000, "verification_status": "Verified",
                "purpose": "debt_consolidation", "dti": 22.4, "delinq_2yrs": 0,
                "inq_last_6mths": 1, "open_acc": 8, "pub_rec": 0,
                "revol_bal": 9500, "revol_util": 48.2, "total_acc": 18,
                "emp_length": "5 years", "fico_range_low": 680, "fico_range_high": 684,
            }
        }


class ScoreResponse(BaseModel):
    calibrated_pd: float
    decision: str
    threshold_used: float
    reason_codes: list[str] = []
    adverse_action_letter: str | None = None
    unmapped_features_flagged_for_review: list[str] = []


@app.get("/")
def root():
    return {
        "service": "Credit Expansion Engine -- Scoring API",
        "docs": "/docs",
        "health": "/health",
        "score_endpoint": "POST /score",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    champion = get_champion()

    raw = pd.DataFrame([applicant.model_dump()])
    engineered = features.engineer_features(raw)
    X = engineered[FEATURE_COLS]

    pd_estimate = float(champion.predict_proba(X)[0, 1])
    model_decision = "Approved" if pd_estimate < THRESHOLD else "Declined"

    annual_payment = applicant.installment * 12
    payment_to_income = annual_payment / applicant.annual_inc if applicant.annual_inc > 0 else float("inf")
    affordability_override = payment_to_income > PAYMENT_TO_INCOME_CEILING

    decision = "Declined" if affordability_override else model_decision

    response = ScoreResponse(
        calibrated_pd=round(pd_estimate, 4),
        decision=decision,
        threshold_used=THRESHOLD,
    )

    if decision == "Declined":
        if affordability_override and model_decision == "Approved":
            # The model itself would have approved this -- the decline comes
            # entirely from the affordability business rule, not from SHAP.
            # Reported separately and honestly rather than dressed up as a
            # model-driven reason it isn't.
            response.reason_codes = [
                f"R13: This loan's payment (${annual_payment:,.0f}/year) would be "
                f"{payment_to_income:.0%} of stated annual income -- exceeds the "
                f"{PAYMENT_TO_INCOME_CEILING:.0%} affordability threshold (business-rule "
                f"override; the statistical model alone would have approved this application)"
            ]
            response.adverse_action_letter = ex.generate_adverse_action_letter(
                "API-REQUEST", decision, [ex.ReasonCode(
                    "R13", "Requested payment is unaffordable relative to reported income"
                )]
            )
        else:
            # Real-time SHAP for a single request: a lightweight, fast
            # approximation is used rather than the full permutation explainer
            # from Phase 5 (which needs a background sample and is too slow
            # for interactive per-request latency). Production deployment would
            # cache SHAP background data and warm the explainer at startup --
            # noted as a scaling consideration, not implemented here.
            codes, unmapped = ex.map_shap_to_reason_codes(
                _approximate_feature_contributions(X.iloc[0]), FEATURE_COLS, n_reasons=3
            )
            response.reason_codes = [f"{c.code}: {c.approved_text}" for c in codes]
            response.adverse_action_letter = ex.generate_adverse_action_letter(
                "API-REQUEST", decision, codes
            )
            response.unmapped_features_flagged_for_review = unmapped

    return response


def _approximate_feature_contributions(row: pd.Series):
    """Placeholder single-request contribution proxy: compares each numeric
    feature to a plausible 'good applicant' reference point, in the
    direction that increases risk. This is NOT SHAP and should not be
    presented as such -- it exists so the endpoint can return reason codes
    without the latency cost of full SHAP per request. Swap for cached-
    background SHAP (see docstring above) before any real deployment."""
    import numpy as np
    reference = {
        "dti": 15.0, "fico_midpoint": 720.0, "delinq_2yrs": 0, "revol_util": 30.0,
        "emp_length_years": 5.0, "annual_inc": 70000.0, "inq_last_6mths": 0,
        "open_acc": 8, "pub_rec": 0, "total_acc": 15, "loan_amnt": 10000.0,
    }
    contributions = []
    for feat in FEATURE_COLS:
        if feat in reference and feat in row.index:
            direction = 1 if feat in ("dti", "delinq_2yrs", "revol_util", "inq_last_6mths",
                                       "pub_rec", "loan_amnt") else -1
            gap = (row[feat] - reference[feat]) * direction
            contributions.append(max(gap, 0))
        else:
            contributions.append(0)
    return np.array(contributions, dtype=float)


if __name__ == "__main__":
    # Convenience entry point for platforms that run `python api.py` directly
    # (e.g. Railway's auto-detection) rather than an explicit uvicorn command.
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
