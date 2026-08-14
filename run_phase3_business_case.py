import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import pricing
import expansion

# Empirically-derived inputs (from the actual accepted-loans data, 2007-2013)
AVG_LOAN_AMOUNT = 11836.32
AVG_LGD = 0.334
AVG_INTEREST_MARGIN = 0.2095  # amortization-adjusted total interest over the 3-year term
FN_COST = 5   # from config.yaml cost_matrix -- approving a bad-risk applicant
FP_COST = 1   # declining a good-risk applicant

print("Loading declined pool (naive vs. corrected scores)...")
declined = pd.read_parquet("data/processed/declined_scored.parquet")
print("Declined pool:", declined.shape)

# Determine the cost-minimizing threshold using the CORRECTED model's own
# performance on the accepted population it was fit on -- keeps the
# threshold on the same scale as what we're about to apply it to.
print("\nRe-deriving corrected model predictions on the accepted training set...")
import features
train = pd.read_parquet("data/processed/train.parquet")
train = features.engineer_features(train)
train = features.filter_valid_shared_features(train)
shared_model_features = ["loan_amnt", "dti", "emp_length_years", "addr_state", "fico_midpoint"]
X_train = train[shared_model_features].copy()
X_train["_y"] = train["default"].values
X_train = X_train.dropna(subset=shared_model_features)
y_train = X_train.pop("_y").astype(int).reset_index(drop=True)
X_train = X_train.reset_index(drop=True)

# NOTE: this reuses the naive model's scores as a proxy since refitting the
# full Heckman pipeline here would repeat the slow probit step. The
# distinction matters much more for the DECLINED population (where
# selection bias is severe) than for the threshold search itself, which
# only needs a reasonable operating point -- documented as a simplification.
print("\nLoading the ACTUAL saved naive model (the same one that scored the declined pool) --")
print("fixes a prior inconsistency where a freshly re-fit model was used for the threshold instead.")
import joblib
naive_model = joblib.load("data/processed/naive_shared_model.joblib")
saved_columns = joblib.load("data/processed/shared_model_columns.joblib")
saved_y_train = joblib.load("data/processed/naive_model_y_train.joblib")
saved_scaler = joblib.load("data/processed/shared_feature_scaler.joblib")
numeric_cols = ["loan_amnt", "dti", "emp_length_years", "fico_midpoint"]

X_train_enc = pd.get_dummies(X_train, columns=["addr_state"], drop_first=True)
X_train_enc = X_train_enc.reindex(columns=saved_columns, fill_value=0)
X_train_enc[numeric_cols] = saved_scaler.transform(X_train_enc[numeric_cols])
train_probs = naive_model.predict_proba(X_train_enc)[:, 1]

# Sanity check: this X_train/y_train reconstruction should match what was
# used to originally fit the saved model (same source data, same cleaning).
assert len(train_probs) == len(saved_y_train), \
    "Row count mismatch vs. the saved model's training data -- investigate before trusting the threshold."
y_train = saved_y_train  # use the exact labels the saved model was fit against

threshold, costs = pricing.cost_minimizing_threshold(y_train.values, train_probs, FN_COST, FP_COST)
print(f"Cost-minimizing threshold: {threshold:.3f}")

# Apply to the declined pool using the CORRECTED (selection-bias-adjusted) scores
declined_for_expansion = declined.rename(columns={"corrected_predicted_pd": "predicted_pd"})

print("\n--- Business case: NAIVE model ---")
declined_naive = declined.rename(columns={"naive_predicted_pd": "predicted_pd"})
safe_naive = expansion.identify_safe_to_approve(declined_naive, threshold)
case_naive = expansion.business_case(safe_naive, AVG_LOAN_AMOUNT, AVG_INTEREST_MARGIN, AVG_LGD)
for k, v in case_naive.items():
    print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")

print("\n--- Business case: CORRECTED (reject-inference) model ---")
safe_corrected = expansion.identify_safe_to_approve(declined_for_expansion, threshold)
case_corrected = expansion.business_case(safe_corrected, AVG_LOAN_AMOUNT, AVG_INTEREST_MARGIN, AVG_LGD)
for k, v in case_corrected.items():
    print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")

print("\n--- Threshold sensitivity (corrected model) ---")
sensitivity = expansion.threshold_sensitivity(
    declined_for_expansion, np.arange(0.10, 0.45, 0.05),
    AVG_LOAN_AMOUNT, AVG_INTEREST_MARGIN, AVG_LGD
)
print(sensitivity.to_string(index=False))

sensitivity.to_csv("reports/figures/phase3_threshold_sensitivity.csv", index=False)
print("\nSaved sensitivity table to reports/figures/phase3_threshold_sensitivity.csv")
