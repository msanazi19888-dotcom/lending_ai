import sys
sys.path.insert(0, "src")
import pandas as pd
import features
import modeling
import evaluation

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

train_feat = features.engineer_features(train)
test_feat = features.engineer_features(test)

feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_years", "fico_midpoint",
]

X_train = features.build_feature_matrix(train_feat, feature_cols)
X_test = features.build_feature_matrix(test_feat, feature_cols)
X_train = modeling.drop_zero_variance_features(X_train)
X_test = X_test[X_train.columns]

y_train = train_feat["default"]
y_test = test_feat["default"]

print("Baseline (untuned) Random Forest...")
preprocessor = modeling.build_preprocessor(X_train)
candidates = modeling.candidate_models(preprocessor)
rf_pipe = candidates["random_forest"].fit(X_train, y_train)
probs = rf_pipe.predict_proba(X_test)[:, 1]
metrics = evaluation.discriminatory_power(y_test.values, probs)
print(f"Baseline RF: AUC={metrics['auc']:.4f}  KS={metrics['ks']:.4f}  Gini={metrics['gini']:.4f}")

print("\nTuning Random Forest...")
study = modeling.tune_random_forest(X_train, y_train, n_trials=15, cv_folds=3, subsample_n=50_000)
print(f"Tuned params: {study.best_params}")

print("Refitting tuned RF on full training set...")
tuned_rf = modeling.refit_best_on_full_data("random_forest", study.best_params, X_train, y_train)
probs_tuned = tuned_rf.predict_proba(X_test)[:, 1]
metrics_tuned = evaluation.discriminatory_power(y_test.values, probs_tuned)
print(f"Tuned RF:    AUC={metrics_tuned['auc']:.4f}  KS={metrics_tuned['ks']:.4f}  Gini={metrics_tuned['gini']:.4f}")

print("\n--- For reference, previously reported (tuned) results ---")
print("Logistic Regression: AUC=0.6811")
print("Gradient Boosting:   AUC=0.6808")
print("XGBoost:              AUC=0.6816")