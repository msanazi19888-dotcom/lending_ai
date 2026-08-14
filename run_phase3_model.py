import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import features
import reject_inference as ri

shared_model_features = ["loan_amnt", "dti", "emp_length_years", "addr_state", "fico_midpoint"]
numeric_cols = ["loan_amnt", "dti", "emp_length_years", "fico_midpoint"]

print("Loading accepted (train) and declined populations...")
train = pd.read_parquet("data/processed/train.parquet")
train = features.engineer_features(train)
train = features.filter_valid_shared_features(train)

declined = pd.read_parquet("data/processed/rejected_clean_2007_2013.parquet")

X_accepted = train[shared_model_features].copy()
y_accepted = train["default"].copy()
X_declined = declined[shared_model_features].copy()

X_accepted["_y"] = y_accepted.values
before = len(X_accepted)
X_accepted = X_accepted.dropna(subset=shared_model_features)
print(f"Dropped {before - len(X_accepted)} accepted rows with missing shared features")
y_accepted = X_accepted.pop("_y").astype(int).reset_index(drop=True)
X_accepted = X_accepted.reset_index(drop=True)

print("Accepted (shared features):", X_accepted.shape)
print("Declined (shared features):", X_declined.shape)

X_accepted["_source"] = "accepted"
X_declined["_source"] = "declined"
combined = pd.concat([X_accepted, X_declined], ignore_index=True)
combined_encoded = pd.get_dummies(combined, columns=["addr_state"], drop_first=True)

X_accepted_enc = combined_encoded[combined_encoded["_source"] == "accepted"].drop(columns=["_source"]).reset_index(drop=True)
X_declined_enc = combined_encoded[combined_encoded["_source"] == "declined"].drop(columns=["_source"]).reset_index(drop=True)
X_all_enc = combined_encoded.drop(columns=["_source"]).reset_index(drop=True)
was_accepted = pd.Series((combined["_source"] == "accepted").astype(int))

# FIX: scale numeric features before fitting. loan_amnt (thousands), dti and
# emp_length_years (single/double digits), and fico_midpoint (hundreds) are
# on wildly different scales, which caused lbfgs to fail to converge in the
# unscaled version -- not just a cosmetic warning, but a real risk that the
# fitted coefficients weren't the true best fit. Scaler is fit on the FULL
# applicant pool (accepted + declined) since the selection equation needs
# consistent scaling across both populations; one-hot dummy columns are left
# unscaled (standard practice).
print("\nFitting StandardScaler on the combined applicant pool...")
scaler = StandardScaler().fit(X_all_enc[numeric_cols])

def scale(df):
    out = df.copy()
    out[numeric_cols] = scaler.transform(df[numeric_cols])
    return out

X_accepted_scaled = scale(X_accepted_enc)
X_declined_scaled = scale(X_declined_enc)
X_all_scaled = scale(X_all_enc)

print("Fitting naive shared-feature model (scaled, accepted only)...")
naive_model = LogisticRegression(max_iter=2000).fit(X_accepted_scaled, y_accepted)
naive_scores = naive_model.predict_proba(X_declined_scaled)[:, 1]
print(f"Naive mean predicted PD on declined pool: {naive_scores.mean():.4f}")

print("\nFitting Heckman selection equation + corrected outcome model (scaled)...")
print("(Slow step -- probit MLE on ~1.5M rows. Expect several minutes.)")
result = ri.heckman_two_stage(X_all_scaled, was_accepted, X_accepted_scaled, y_accepted)
converged = result["selection_model"].mle_retvals.get("converged", "unknown")
print(f"Selection model converged: {converged}")
print(f"Mills ratio coefficient: {result['mills_ratio_coefficient']:.4f}")
print(f"Selection bias detected: {result['selection_bias_detected']}")

comparison = ri.compare_naive_vs_corrected(naive_model, result, X_declined_scaled)
print("\n--- Naive vs. corrected predicted PD on declined pool (scaled, converged fit) ---")
print(comparison.describe())

# Save everything needed downstream -- single source of truth, no more
# ad-hoc refits in later scripts.
joblib.dump(naive_model, "data/processed/naive_shared_model.joblib")
joblib.dump(result["outcome_model"], "data/processed/corrected_outcome_model.joblib")
joblib.dump(result["selection_model"], "data/processed/selection_model.joblib")
joblib.dump(scaler, "data/processed/shared_feature_scaler.joblib")
joblib.dump(list(X_accepted_enc.columns), "data/processed/shared_model_columns.joblib")
joblib.dump(y_accepted, "data/processed/naive_model_y_train.joblib")

comparison.to_parquet("data/processed/phase3_naive_vs_corrected.parquet", index=False)
declined_scored = declined.reset_index(drop=True).join(comparison)
declined_scored.to_parquet("data/processed/declined_scored.parquet", index=False)
print("\nSaved all model artifacts and scored data to data/processed/")
