import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
calib = pd.read_parquet("data/processed/calib.parquet")
test = pd.read_parquet("data/processed/test.parquet")

feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length",
    "fico_range_low", "fico_range_high",
]

missing = pd.DataFrame({
    "train_2007_2013": train[feature_cols].isna().mean(),
    "calib_2014": calib[feature_cols].isna().mean(),
    "test_2015": test[feature_cols].isna().mean(),
}) * 100

missing["max_abs_diff_pct_points"] = missing[["train_2007_2013", "calib_2014", "test_2015"]].max(axis=1) - \
                                       missing[["train_2007_2013", "calib_2014", "test_2015"]].min(axis=1)

print(missing.round(2).sort_values("max_abs_diff_pct_points", ascending=False).to_string())

flagged = missing[missing["max_abs_diff_pct_points"] > 2.0]
print(f"\nFeatures with >2 percentage point missingness swing across splits: {len(flagged)}")
if len(flagged):
    print(flagged.round(2).to_string())
