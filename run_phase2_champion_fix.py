import sys
sys.path.insert(0, "src")
import pandas as pd
import features
import modeling
import calibration
import evaluation
import joblib

print("Loading Phase 1/2 data...")
train = pd.read_parquet("data/processed/train.parquet")
calib = pd.read_parquet("data/processed/calib.parquet")
test = pd.read_parquet("data/processed/test.parquet")

train_feat = features.engineer_features(train)
calib_feat = features.engineer_features(calib)
test_feat = features.engineer_features(test)
# NOTE: features.py's engineer_features now includes the emp_length parsing
# fix found during Phase 3 ('< 1 year' correctly -> 0, not 1). This refit
# will incidentally pick that up too, since the module is shared code --
# flagged explicitly here rather than left as a silent side effect.

feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_years", "fico_midpoint",
]

X_train = features.build_feature_matrix(train_feat, feature_cols)
X_calib = features.build_feature_matrix(calib_feat, feature_cols)
X_test = features.build_feature_matrix(test_feat, feature_cols)
X_train = modeling.drop_zero_variance_features(X_train)
X_calib = X_calib[X_train.columns]
X_test = X_test[X_train.columns]

y_train = train_feat["default"]
y_calib = calib_feat["default"]
y_test = test_feat["default"]

print("\nRe-tuning logistic regression to recover the actual best hyperparameters...")
print("(This was never saved originally -- only printed and lost to truncation.)")
study = modeling.tune_logistic_regression(X_train, y_train, n_trials=15, cv_folds=3, subsample_n=50_000)
print(f"Tuned params: {study.best_params}")
print(f"Subsample CV AUC: {study.best_value:.4f}")

print("\nRefitting with these TUNED params on the FULL training set (this is the real champion model)...")
champion_tuned = modeling.refit_best_on_full_data("logistic_regression", study.best_params, X_train, y_train)

# Compare against the untuned baseline for the record
probs_tuned = champion_tuned.predict_proba(X_test)[:, 1]
metrics_tuned = evaluation.discriminatory_power(y_test.values, probs_tuned)
print(f"\nTuned champion test AUC: {metrics_tuned['auc']:.4f}  KS: {metrics_tuned['ks']:.4f}  Gini: {metrics_tuned['gini']:.4f}")
print("(For reference, the untuned/mismatched model previously reported: AUC=0.6810)")

print("\nCalibrating the ACTUAL tuned champion...")
calibrated_model = calibration.calibrate(champion_tuned, X_calib, y_calib)

probs_before = champion_tuned.predict_proba(X_test)[:, 1]
probs_after = calibrated_model.predict_proba(X_test)[:, 1]

print("\n--- Calibration report (tuned champion, test set) ---")
report = calibration.calibration_report(
    y_test.values, probs_before, probs_after,
    show=False, save_path="reports/figures/calibration_curve_tuned.png"
)

print("\n--- Discriminatory power, before vs after calibration ---")
for label, probs in [("Before", probs_before), ("After", probs_after)]:
    m = evaluation.discriminatory_power(y_test.values, probs)
    print(f"{label}: AUC={m['auc']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")

joblib.dump(calibrated_model, "data/processed/champion_model_calibrated.joblib")
joblib.dump(study.best_params, "data/processed/champion_tuned_params.joblib")
print("\nSaved the ACTUAL tuned + calibrated champion model, overwriting the previous mismatched one.")
