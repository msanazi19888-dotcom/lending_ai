"""
The flagship idea: score the declined-loan population with a PD model
trained only on the fields shared between accepted and rejected files,
and quantify how many declined applicants look safe to approve.

Kept deliberately honest: this model is trained/evaluated only on
SHARED_FEATURES (see features.py) so the comparison between "approved and
repaid" and "declined but looks similar" is apples-to-apples, not inflated
by information the original decision never had access to either.
"""
import pandas as pd
import numpy as np


def score_declined_pool(shared_model, rejected_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Apply the shared-feature PD model to the declined population."""
    X = rejected_df[feature_cols]
    scores = shared_model.predict_proba(X)[:, 1]
    out = rejected_df.copy()
    out["predicted_pd"] = scores
    return out


def identify_safe_to_approve(scored_declined: pd.DataFrame, approval_threshold: float) -> pd.DataFrame:
    """Declined applicants whose predicted PD is below the bank's own
    approval threshold (derived in pricing.py from the cost matrix)."""
    return scored_declined[scored_declined["predicted_pd"] <= approval_threshold]


def business_case(safe_to_approve: pd.DataFrame, avg_loan_amount: float,
                   avg_interest_margin: float, avg_lgd: float) -> dict:
    """Rough expected-value estimate of expanding approvals to this group.

    expected_profit_per_loan = (1 - PD) * interest_margin * loan_amount
                                - PD * lgd * loan_amount
    """
    n = len(safe_to_approve)
    mean_pd = safe_to_approve["predicted_pd"].mean() if n else 0.0

    expected_profit_per_loan = (
        (1 - mean_pd) * avg_interest_margin * avg_loan_amount
        - mean_pd * avg_lgd * avg_loan_amount
    )

    return {
        "n_additional_approvals": n,
        "mean_predicted_pd": mean_pd,
        "expected_profit_per_loan": expected_profit_per_loan,
        "expected_total_incremental_profit": expected_profit_per_loan * n,
    }


def threshold_sensitivity(scored_declined: pd.DataFrame, thresholds: np.ndarray,
                           avg_loan_amount: float, avg_interest_margin: float,
                           avg_lgd: float) -> pd.DataFrame:
    """How the expansion opportunity changes as the approval threshold moves --
    the sensitivity table for Phase 3's writeup."""
    rows = []
    for t in thresholds:
        pool = identify_safe_to_approve(scored_declined, t)
        case = business_case(pool, avg_loan_amount, avg_interest_margin, avg_lgd)
        rows.append({"threshold": t, **case})
    return pd.DataFrame(rows)
