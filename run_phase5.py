import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import joblib
import features
import explainability as ex
import fairness

feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_years", "fico_midpoint",
]

print("Loading test set and champion model...")
test = pd.read_parquet("data/processed/test.parquet")
test_feat = features.engineer_features(test)
X_test = test_feat[feature_cols]

champion = joblib.load("data/processed/champion_model_calibrated.joblib")
calibrated_pd = champion.predict_proba(X_test)[:, 1]
test_feat = test_feat.reset_index(drop=True)
test_feat["calibrated_pd"] = calibrated_pd

THRESHOLD = 0.430
test_feat["decision"] = np.where(test_feat["calibrated_pd"] < THRESHOLD, "Approved", "Declined")
print(f"Decisions at threshold {THRESHOLD}: {test_feat['decision'].value_counts().to_dict()}")

# --- SHAP + reason codes for a sample of DECLINED applicants ---
print("\nComputing SHAP values for a sample of declined applicants...")
print("(SHAP's default masker can't handle raw string categorical columns --")
print("found via a crash, not assumed. Fix: encode categoricals to numeric codes")
print("for SHAP's masker, with a lookup to decode back before calling the model,")
print("so predict_proba still sees real categories.)")
declined_sample = test_feat[test_feat["decision"] == "Declined"].sample(n=20, random_state=42)
X_sample = declined_sample[feature_cols]

cat_cols = X_test.select_dtypes(exclude="number").columns.tolist()
category_maps = {c: dict(enumerate(X_test[c].astype("category").cat.categories)) for c in cat_cols}

def predict_proba_from_codes(X_coded):
    X_df = pd.DataFrame(X_coded, columns=feature_cols)
    for c in cat_cols:
        X_df[c] = X_df[c].round().astype(int).map(category_maps[c])
    return champion.predict_proba(X_df)

X_test_numeric = X_test.copy()
X_sample_numeric = X_sample.copy()
for c in cat_cols:
    X_test_numeric[c] = X_test[c].astype("category").cat.codes
    X_sample_numeric[c] = pd.Categorical(X_sample[c], categories=X_test[c].astype("category").cat.categories).codes

import shap
background = X_test_numeric.sample(100, random_state=42)
explainer = shap.Explainer(predict_proba_from_codes, background)
shap_result = explainer(X_sample_numeric)
shap_values_sample = shap_result.values[:, :, 1] if shap_result.values.ndim == 3 else shap_result.values
feature_names_transformed = X_sample.columns.tolist()

print("\n--- Sample adverse action letters (real declined applicants) ---")
all_unmapped = []
for i in range(min(5, len(declined_sample))):
    codes, unmapped = ex.map_shap_to_reason_codes(
        shap_values_sample[i], feature_names_transformed, n_reasons=3
    )
    all_unmapped.extend(unmapped)
    letter = ex.generate_adverse_action_letter(f"TEST-{declined_sample.index[i]}", "Declined", codes)
    print(f"\n{letter}")

print(f"\n\nGenuinely unmapped features encountered (compliance review queue): {set(all_unmapped)}")

# --- Fairness check: approval rate by state (proxy, no protected-class fields exist) ---
print("\n--- Disparate impact check: approval rate by state ---")
approve_flag = (test_feat["decision"] == "Approved")
rates = fairness.approval_rate_by_group(approve_flag, test_feat["addr_state"])
result = fairness.four_fifths_test(rates)
print(f"Most favored state: {result['most_favored_group']} ({rates[result['most_favored_group']]:.1%} approval)")
print(f"Least favored state: {result['least_favored_group']} ({rates[result['least_favored_group']]:.1%} approval)")
print(f"Impact ratio: {result['impact_ratio']:.3f}")
print(f"Passes four-fifths rule: {result['passes_four_fifths']}")

rates.sort_values().to_csv("reports/figures/phase5_approval_rate_by_state.csv")
print("\nSaved full state-level approval rates to reports/figures/phase5_approval_rate_by_state.csv")
