import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import data_loading
import features
import reject_inference as ri
import pricing
import expansion
import fairness

cfg = data_loading.load_config()
numeric_cols = ["loan_amnt", "dti", "emp_length_years", "fico_midpoint"]  # addr_state REMOVED

print("=" * 70)
print("FAIRNESS REMEDIATION: refitting without addr_state as a direct model input")
print("=" * 70)

print("\nReloading raw accepted-loans data (train population, keeping leakage")
print("fields this time -- needed for BOTH model fitting and threshold derivation,")
print("bundled into one load rather than two)...")
df = data_loading.load_accepted("data/raw/accepted_loans.csv")
df = data_loading.filter_to_terminal_population(df, cfg)
df = data_loading.filter_to_matured_population(df, cfg)
dates = pd.to_datetime(df["issue_d"], format="%b-%Y")
train_raw = df[dates.dt.year <= cfg["split"]["train_end_year"]].copy()

train_raw["realized_net"] = train_raw["total_pymnt"] + train_raw["recoveries"] - train_raw["funded_amnt"]
train_raw = train_raw.dropna(subset=["realized_net"])
train_raw = features.engineer_features(train_raw)
train_raw = train_raw.dropna(subset=numeric_cols)
train_raw = features.filter_valid_shared_features(train_raw)
print(f"Train population (accepted, 2007-2013): {train_raw.shape}")

declined = pd.read_parquet("data/processed/rejected_clean_2007_2013.parquet")
print(f"Declined population (already cleaned): {declined.shape}")

X_accepted = train_raw[numeric_cols].reset_index(drop=True)
y_accepted = train_raw["default"].reset_index(drop=True)
realized_net = train_raw["realized_net"].reset_index(drop=True).values
X_declined = declined[numeric_cols].reset_index(drop=True)

combined = pd.concat([X_accepted.assign(_s="a"), X_declined.assign(_s="d")], ignore_index=True)
X_all = combined[numeric_cols].reset_index(drop=True)
was_accepted = pd.Series((combined["_s"] == "a").astype(int))

print("\nFitting StandardScaler (4 numeric features only, no state dummies)...")
scaler = StandardScaler().fit(X_all)
X_all_s = pd.DataFrame(scaler.transform(X_all), columns=numeric_cols)
X_accepted_s = X_all_s[was_accepted == 1].reset_index(drop=True)
X_declined_s = X_all_s[was_accepted == 0].reset_index(drop=True)

print("Fitting naive shared-feature model (no addr_state)...")
naive_model = LogisticRegression(max_iter=2000).fit(X_accepted_s, y_accepted)
naive_scores_declined = naive_model.predict_proba(X_declined_s)[:, 1]
naive_scores_train = naive_model.predict_proba(X_accepted_s)[:, 1]
print(f"Naive mean predicted PD on declined pool: {naive_scores_declined.mean():.4f}")

print("\nFitting Heckman selection equation + corrected outcome model...")
print("(Should be noticeably faster than before -- much smaller design matrix)")
result = ri.heckman_two_stage(X_all_s, was_accepted, X_accepted_s, y_accepted)
print(f"Converged: {result['selection_model'].mle_retvals.get('converged')}")
print(f"Mills ratio coefficient: {result['mills_ratio_coefficient']:.4f}")

comparison = ri.compare_naive_vs_corrected(naive_model, result, X_declined_s)
print("\n--- Naive vs corrected on declined pool (no addr_state) ---")
print(comparison.describe())

print("\n--- Deriving the new profit-maximizing threshold (real dollar outcomes) ---")
grid = np.linspace(0.01, 0.80, 80)
profits = np.array([realized_net[naive_scores_train < t].sum() for t in grid])
new_threshold = grid[np.argmax(profits)]
print(f"New profit-maximizing threshold: {new_threshold:.3f}")

declined_scored = declined.reset_index(drop=True).join(comparison)
declined_naive = declined_scored.rename(columns={"naive_predicted_pd": "predicted_pd"})
declined_corrected = declined_scored.rename(columns={"corrected_predicted_pd": "predicted_pd"})

AVG_LOAN_AMOUNT = 11836.32
AVG_LGD = 0.334
AVG_INTEREST_MARGIN = 0.2095

print("\n--- Business case (no addr_state) ---")
for label, pool in [("NAIVE", declined_naive), ("CORRECTED", declined_corrected)]:
    safe = expansion.identify_safe_to_approve(pool, new_threshold)
    case = expansion.business_case(safe, AVG_LOAN_AMOUNT, AVG_INTEREST_MARGIN, AVG_LGD)
    print(f"\n{label} model:")
    for k, v in case.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")

print("\n--- RE-TEST: disparate impact by state, now that state is NOT a direct model input ---")
approve_flag = declined_corrected["predicted_pd"] < new_threshold
rates = fairness.approval_rate_by_group(approve_flag, declined_scored["addr_state"])
fairness_result = fairness.four_fifths_test(rates)
print(f"Approval rate: {approve_flag.mean():.1%}")
print(f"Most favored state: {fairness_result['most_favored_group']} ({rates[fairness_result['most_favored_group']]:.1%})")
print(f"Least favored state: {fairness_result['least_favored_group']} ({rates[fairness_result['least_favored_group']]:.1%})")
print(f"Impact ratio: {fairness_result['impact_ratio']:.3f}")
print(f"Passes four-fifths rule: {fairness_result['passes_four_fifths']}")
print("(For reference, WITH addr_state as a direct input: impact ratio 0.457, FAILED)")

# Save everything, overwriting the addr_state-inclusive artifacts
joblib.dump(naive_model, "data/processed/naive_shared_model.joblib")
joblib.dump(result["outcome_model"], "data/processed/corrected_outcome_model.joblib")
joblib.dump(result["selection_model"], "data/processed/selection_model.joblib")
joblib.dump(scaler, "data/processed/shared_feature_scaler.joblib")
joblib.dump(numeric_cols, "data/processed/shared_model_columns.joblib")
joblib.dump(y_accepted, "data/processed/naive_model_y_train.joblib")
declined_scored.to_parquet("data/processed/declined_scored.parquet", index=False)
rates.sort_values().to_csv("reports/figures/phase5_fairness_remediated_by_state.csv")
print("\nSaved all remediated artifacts, overwriting the addr_state-inclusive versions.")
