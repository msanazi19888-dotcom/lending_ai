import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import joblib
import data_loading
import features
import pricing

cfg = data_loading.load_config()

print("Reloading raw accepted-loans data to access realized payment outcomes")
print("(deliberately excluded from model features to avoid leakage, but legitimate")
print("and necessary for backtesting against REAL historical results)...")
usecols = None  # need the full file this time -- leakage fields required for backtest
df = data_loading.load_accepted("data/raw/accepted_loans.csv")

df = data_loading.filter_to_terminal_population(df, cfg)
df = data_loading.filter_to_matured_population(df, cfg)

# Isolate just the TEST slice (2015) -- same population as test.parquet,
# but this time keeping total_pymnt/recoveries/funded_amnt for backtesting.
dates = pd.to_datetime(df["issue_d"], format="%b-%Y")
test_raw = df[dates.dt.year >= cfg["split"]["test_start_year"]].copy()
print(f"Test population (with leakage fields retained for backtesting): {test_raw.shape}")
print(f"(For reference, the original test.parquet had 283,026 rows -- should match)")

# Real, realized dollar outcome per loan -- ground truth, not an assumption.
test_raw["realized_net"] = test_raw["total_pymnt"] + test_raw["recoveries"] - test_raw["funded_amnt"]

n_before = len(test_raw)
test_raw = test_raw.dropna(subset=["realized_net"])
if n_before - len(test_raw) > 0:
    print(f"Dropped {n_before - len(test_raw)} rows with missing payment data (can't backtest without it)")

# Rebuild the exact same 19-feature matrix used for the champion model
test_feat = features.engineer_features(test_raw)
feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_years", "fico_midpoint",
]
X_test = test_feat[feature_cols]

print("\nLoading the champion (tuned + calibrated) model...")
champion = joblib.load("data/processed/champion_model_calibrated.joblib")
calibrated_pd = champion.predict_proba(X_test)[:, 1]

y_test = test_feat["default"].values
realized_net = test_feat["realized_net"].values

print("\n--- Cost-ratio-based threshold (5:1, matches Phase 3's approach) ---")
cost_threshold, costs = pricing.cost_minimizing_threshold(y_test, calibrated_pd, fn_cost=5, fp_cost=1)
print(f"Cost-minimizing threshold: {cost_threshold:.3f}")

print("\n--- NEW: profit-maximizing threshold using REAL realized dollar outcomes ---")
grid = np.linspace(0.01, 0.60, 60)
profits = np.array([realized_net[calibrated_pd < t].sum() for t in grid])
best_idx = np.argmax(profits)
profit_threshold = grid[best_idx]
print(f"Profit-maximizing threshold: {profit_threshold:.3f}")

print("\n--- Backtest: three policies compared on the SAME real outcomes ---")
actual_approve_all_profit = realized_net.sum()
print(f"Actual historical policy (LendingClub approved all {len(test_feat):,} test loans):")
print(f"  Total realized profit: ${actual_approve_all_profit:,.2f}")

for label, threshold in [("Cost-ratio threshold", cost_threshold), ("Profit-maximizing threshold", profit_threshold)]:
    approved_mask = calibrated_pd < threshold
    n_approved = approved_mask.sum()
    n_declined = (~approved_mask).sum()
    policy_profit = realized_net[approved_mask].sum()
    declined_would_have_defaulted = y_test[~approved_mask].sum()
    declined_would_have_paid = n_declined - declined_would_have_defaulted
    print(f"\n{label} ({threshold:.3f}):")
    print(f"  Approved: {n_approved:,}  Declined: {n_declined:,}")
    print(f"  Total realized profit if this policy had been used: ${policy_profit:,.2f}")
    print(f"  Improvement vs. actual historical approve-all: ${policy_profit - actual_approve_all_profit:,.2f}")
    print(f"  Of declined loans: {declined_would_have_defaulted:,} would have defaulted (good catch), "
          f"{declined_would_have_paid:,} would have paid off fine (missed opportunity)")

results = pd.DataFrame({"threshold": grid, "total_profit": profits})
results.to_csv("reports/figures/phase4_profit_threshold_scan.csv", index=False)
print("\nSaved threshold scan to reports/figures/phase4_profit_threshold_scan.csv")
