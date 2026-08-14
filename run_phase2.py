import sys
sys.path.insert(0, "src")
import pandas as pd
import features
import modeling
import evaluation

print("Loading Phase 1 outputs...")
train = pd.read_parquet("data/processed/train.parquet")
calib = pd.read_parquet("data/processed/calib.parquet")
test = pd.read_parquet("data/processed/test.parquet")

train_feat = features.engineer_features(train)
calib_feat = features.engineer_features(calib)
test_feat = features.engineer_features(test)

# Cleaned feature set: dropped 'term'/'term_months' (constant after the
# maturity filter) and the raw emp_length/fico_range_low/high in favor of
# their derived numeric versions, to avoid double-counting the same signal.
feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "application_type",
    "emp_length_years", "fico_midpoint",
]

X_train = features.build_feature_matrix(train_feat, feature_cols)
X_calib = features.build_feature_matrix(calib_feat, feature_cols)
X_test = features.build_feature_matrix(test_feat, feature_cols)

# Safety net: auto-detect and drop anything still zero-variance
X_train = modeling.drop_zero_variance_features(X_train)
X_calib = X_calib[X_train.columns]
X_test = X_test[X_train.columns]

y_train = train_feat["default"]
y_calib = calib_feat["default"]
y_test = test_feat["default"]

print("\nFinal feature matrix shape:", X_train.shape)
print("Features used:", X_train.columns.tolist())

preprocessor = modeling.build_preprocessor(X_train)
candidates = modeling.candidate_models(preprocessor)
fitted = modeling.fit_all(candidates, X_train, y_train)

print("\n--- Discriminatory power (test set) ---")
for name, pipe in fitted.items():
    probs = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluation.discriminatory_power(y_test.values, probs)
    print(f"{name:22s}  AUC={metrics['auc']:.4f}   KS={metrics['ks']:.4f}   Gini={metrics['gini']:.4f}")