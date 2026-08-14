import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import joblib
import data_loading
import features
import pricing

cfg = data_loading.load_config()
AVG_LGD = 0.334  # from Phase 3, derived from train charged-off loans -- no lookahead bias

print("Reloading raw accepted-loans data (last time for this phase)...")
df = data_loading.load_accepted("data/raw/accepted_loans.csv")
df = data_loading.filter_to_terminal_population(df, cfg)
df = data_loading.filter_to_matured_population(df, cfg)

dates = pd.to_datetime(df["issue_d"], format="%b-%Y")
test_raw = df[dates.dt.year >= cfg["split"]["test_start_year"]].copy()
test_raw["realized_net"] = test_raw["total_pymnt"] + test_raw["recoveries"] - test_raw["funded_amnt"]
test_raw = test_raw.dropna(subset=["realized_net"])
print(f"Test population: {test_raw.shape}")

test_feat = features.engineer_features(test_raw)
feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_years", "fico_midpoint",
]
X_test = test_feat[feature_cols]

champion = joblib.load("data/processed/champion_model_calibrated.joblib")
calibrated_pd = champion.predict_proba(X_test)[:, 1]
test_feat["calibrated_pd"] = calibrated_pd

# --- Per-loan theoretical expected profit (vectorized amortization) ---
def total_interest_fraction(apr_pct, term_months=36):
    r = (apr_pct / 100) / 12
    r = np.where(r == 0, 1e-9, r)
    factor = r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)
    return factor * term_months - 1

test_feat["interest_margin"] = total_interest_fraction(test_feat["int_rate"].values)
test_feat["theoretical_expected_profit"] = (
    (1 - test_feat["calibrated_pd"]) * test_feat["interest_margin"] * test_feat["funded_amnt"]
    - test_feat["calibrated_pd"] * AVG_LGD * test_feat["funded_amnt"]
)

REAL_THRESHOLD = 0.430  # Phase 4's own properly-derived threshold, for the champion
# model on this same population -- fixes an inconsistency where 0.320 (derived from
# Phase 3's different, 5-feature model) was used here by mistake.
approved = test_feat[test_feat["calibrated_pd"] < REAL_THRESHOLD]

print("\n--- Validation: does the theoretical pricing model match REAL realized outcomes? ---")
print(f"Approved loans (calibrated PD < {REAL_THRESHOLD}): {len(approved):,}")
print(f"Theoretical total expected profit (approved loans): ${approved['theoretical_expected_profit'].sum():,.2f}")
print(f"REAL total realized profit (same approved loans):    ${approved['realized_net'].sum():,.2f}")
diff = approved['theoretical_expected_profit'].sum() - approved['realized_net'].sum()
pct_diff = diff / approved['realized_net'].sum() * 100
print(f"Difference: ${diff:,.2f} ({pct_diff:+.1f}%)")
print("(A positive gap here is expected and likely reflects prepayment: the amortization")
print("formula assumes loans run their full 36-month term, but many 'Fully Paid' loans")
print("pay off early in reality, collecting less total interest than the full-term calc assumes.")
print("Related to the same survival/censoring theme already scoped out as future work.)")

test_feat[["grade", "sub_grade", "int_rate", "calibrated_pd", "funded_amnt",
           "theoretical_expected_profit", "realized_net"]].to_csv(
    "reports/figures/phase4_pricing_detail.csv", index=False
)
print("\nSaved per-loan detail to reports/figures/phase4_pricing_detail.csv")
print("\nNote: rate optimization was deliberately scoped out -- see src/pricing.py's")
print("optimal_rate() docstring for the full explanation (no rate-elasticity-of-default")
print("mechanism, so the naive optimizer always pushes to the rate grid's ceiling).")
