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

# Tuning budget kept modest for a personal machine -- 10 trials, 3-fold CV,
# on a 30K-row subsample per model. Expect this to take a while; gradient
# boosting is typically the slowest of the three.
TUNE_KWARGS = dict(n_trials=10, cv_folds=3, subsample_n=30_000)

results = {}
for name, tune_fn in [
    ("logistic_regression", modeling.tune_logistic_regression),
    ("gradient_boosting", modeling.tune_gradient_boosting),
    ("xgboost", modeling.tune_xgboost),
]:
    print(f"\n=== Tuning {name} ===")
    study = tune_fn(X_train, y_train, **TUNE_KWARGS)
    print(f"Best subsample CV AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    print(f"Refitting {name} on full training set with best params...")
    final_model = modeling.refit_best_on_full_data(name, study.best_params, X_train, y_train)
    probs = final_model.predict_proba(X_test)[:, 1]
    metrics = evaluation.discriminatory_power(y_test.values, probs)
    results[name] = {"model": final_model, "params": study.best_params, **metrics}
    print(f"Full-data test AUC: {metrics['auc']:.4f}   KS: {metrics['ks']:.4f}   Gini: {metrics['gini']:.4f}")

print("\n=== Final comparison (test set, full-data refit) ===")
for name, r in results.items():
    print(f"{name:22s}  AUC={r['auc']:.4f}   KS={r['ks']:.4f}   Gini={r['gini']:.4f}")

import joblib
best_name = max(results, key=lambda n: results[n]["auc"])
print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")
joblib.dump(results[best_name]["model"], "data/processed/best_pd_model.joblib")
print("Saved to data/processed/best_pd_model.joblib")