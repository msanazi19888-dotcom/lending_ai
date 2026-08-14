import sys
sys.path.insert(0, "src")
import pandas as pd
import features

rejected = pd.read_parquet("data/processed/rejected_2007_2015.parquet")
rejected["year"] = pd.to_datetime(rejected["issue_d"]).dt.year

rejected = rejected[rejected["year"] <= 2013].copy()
print("Rejected population, 2007-2013:", rejected.shape)

rejected = features.engineer_features(rejected)  # now with corrected emp_length parsing

shared_model_features = ["loan_amnt", "dti", "emp_length_years", "addr_state", "fico_midpoint"]
before = len(rejected)
rejected = rejected.dropna(subset=shared_model_features)
print(f"Dropped {before - len(rejected)} rows with missing shared features")

print("\nApplying data quality filters...")
rejected = features.filter_valid_shared_features(rejected)
print("\nFinal clean rejected population:", rejected.shape)

rejected.to_parquet("data/processed/rejected_clean_2007_2013.parquet", index=False)
print("Saved to data/processed/rejected_clean_2007_2013.parquet")

print("\nSummary stats on shared features:")
print(rejected[shared_model_features].describe())