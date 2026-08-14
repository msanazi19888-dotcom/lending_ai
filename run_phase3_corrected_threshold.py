import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import joblib
import data_loading
import features
import expansion

cfg = data_loading.load_config()
shared_model_features = ["loan_amnt", "dti", "emp_length_years", "addr_state", "fico_midpoint"]

print("Reloading raw accepted-loans data (need realized payment outcomes, excluded from")
print("model features but required to derive a REAL profit-maximizing threshold)...")
df = data_loading.load_accepted("data/raw/accepted_loans.csv")
df = data_loading.filter_to_terminal_population(df, cfg)
df = data_loading.filter_to_matured_population(df, cfg)

# Isolate the TRAIN population (2007-2013) -- the same population the naive
# shared-feature model was originally fit on.
dates = pd.to_datetime(df["issue_d"], format="%b-%Y")
train_raw = df[dates.dt.year <= cfg["split"]["train_end_year"]].copy()
print(f"Train population (with leakage fields retained): {train_raw.shape}")

train_raw["realized_net"] = train_raw["total_pymnt"] + train_raw["recoveries"] - train_raw["funded_amnt"]
n_before = len(train_raw)
train_raw = train_raw.dropna(subset=["realized_net"])
print(f"Dropped {n_before - len(train_raw)} rows with missing payment data")

train_raw = features.engineer_features(train_raw)
n_before = len(train_raw)
train_raw = train_raw.dropna(subset=shared_model_features)
train_raw = features.filter_valid_shared_features(train_raw)
print(f"After shared-feature cleaning: {len(train_raw)} rows (dropped {n_before - len(train_raw)})")

X_train = train_raw[shared_model_features].copy()
realized_net = train_raw["realized_net"].values

print("\nLoading the saved naive shared-feature model + scaler (same one used throughout Phase 3)...")
naive_model = joblib.load("data/processed/naive_shared_model.joblib")
saved_columns = joblib.load("data/processed/shared_model_columns.joblib")
saved_scaler = joblib.load("data/processed/shared_feature_scaler.joblib")
numeric_cols = ["loan_amnt", "dti", "emp_length_years", "fico_midpoint"]

X_train_enc = pd.get_dummies(X_train, columns=["addr_state"], drop_first=True)
X_train_enc = X_train_enc.reindex(columns=saved_columns, fill_value=0)
X_train_enc[numeric_cols] = saved_scaler.transform(X_train_enc[numeric_cols])
naive_pd_train = naive_model.predict_proba(X_train_enc)[:, 1]

print("\n--- Deriving the REAL profit-maximizing threshold (replaces the flawed 5:1 assumption) ---")
grid = np.linspace(0.01, 0.80, 80)
profits = np.array([realized_net[naive_pd_train < t].sum() for t in grid])
best_idx = np.argmax(profits)
real_threshold = grid[best_idx]
print(f"Old (flawed) cost-ratio threshold: 0.170")
print(f"NEW profit-maximizing threshold:   {real_threshold:.3f}")

print("\n--- Re-running the Phase 3 business case with the corrected threshold ---")
declined = pd.read_parquet("data/processed/declined_scored.parquet")

AVG_LOAN_AMOUNT = 11836.32
AVG_LGD = 0.334
AVG_INTEREST_MARGIN = 0.2095

declined_naive = declined.rename(columns={"naive_predicted_pd": "predicted_pd"})
declined_corrected = declined.rename(columns={"corrected_predicted_pd": "predicted_pd"})

for label, pool in [("NAIVE model", declined_naive), ("CORRECTED (reject-inference) model", declined_corrected)]:
    safe = expansion.identify_safe_to_approve(pool, real_threshold)
    case = expansion.business_case(safe, AVG_LOAN_AMOUNT, AVG_INTEREST_MARGIN, AVG_LGD)
    print(f"\n--- {label} (corrected threshold {real_threshold:.3f}) ---")
    for k, v in case.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")

print("\n(For reference, at the OLD flawed threshold 0.170: naive=349,596 approvals/$597.6M, "
      "corrected=358,656 approvals/$612.0M)")
