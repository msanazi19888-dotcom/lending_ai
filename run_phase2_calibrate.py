import sys
sys.path.insert(0, "src")
import pandas as pd
import features
import modeling
import calibration
import evaluation

train = pd.read_parquet("data/processed/train.parquet")
calib = pd.read_parquet("data/processed/calib.parquet")
test = pd.read_parquet("data/processed/test.parquet")

train_feat = features.engineer_features(train)
calib_feat = features.engineer_features(calib)
test_feat = features.engineer_features(test)

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

# Champion model: logistic regression -- statistically tied with the tuned
# tree ensembles on discriminatory power, so the interpretable model wins.
print("Fitting champion model (logistic regression) on full training set...")
preprocessor = modeling.build_preprocessor(X_train)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
champion = Pipeline([("prep", preprocessor),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
champion.fit(X_train, y_train)

print("\nCalibrating on the held-out calibration set...")
calibrated_model = calibration.calibrate(champion, X_calib, y_calib)

probs_before = champion.predict_proba(X_test)[:, 1]
probs_after = calibrated_model.predict_proba(X_test)[:, 1]

print("\n--- Calibration report (test set) ---")
report = calibration.calibration_report(
    y_test.values, probs_before, probs_after,
    show=False, save_path="reports/figures/calibration_curve.png"
)

print("\n--- Discriminatory power, before vs after calibration (should be identical -- calibration doesn't change ranking) ---")
for label, probs in [("Before", probs_before), ("After", probs_after)]:
    m = evaluation.discriminatory_power(y_test.values, probs)
    print(f"{label}: AUC={m['auc']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")

import joblib
joblib.dump(calibrated_model, "data/processed/champion_model_calibrated.joblib")
print("\nSaved calibrated champion model to data/processed/champion_model_calibrated.joblib")