import sys
sys.path.insert(0, "src")
import pandas as pd
import data_loading
import features

cfg = data_loading.load_config()

print("Loading rejected loans (27.6M rows, this will take a few minutes)...")
usecols = ["Amount Requested", "Application Date", "Risk_Score",
           "Debt-To-Income Ratio", "Zip Code", "State", "Employment Length"]
rejected = pd.read_csv("data/raw/rejected_loans.csv", usecols=usecols, low_memory=False)
print("Raw rejected shape:", rejected.shape)

rejected = data_loading.normalize_whitespace(rejected)
rejected.columns = [c.strip().lower().replace(" ", "_") for c in rejected.columns]

rejected["application_date"] = pd.to_datetime(rejected["application_date"])
rejected = rejected[rejected["application_date"].dt.year <= 2015].copy()
print("Filtered to 2007-2015 (matching accepted population window):", rejected.shape)

rejected = features.align_rejected_schema(rejected)
print("\nAligned columns:", rejected.columns.tolist())
print("\nMissingness in aligned shared features:")
print(rejected[["loan_amnt", "dti", "emp_length", "addr_state", "fico_midpoint"]].isna().mean())

import os
os.makedirs("data/processed", exist_ok=True)
rejected.to_parquet("data/processed/rejected_2007_2015.parquet", index=False)
print("\nSaved to data/processed/rejected_2007_2015.parquet")