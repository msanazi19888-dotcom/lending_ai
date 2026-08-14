import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
import joblib
import data_loading
import features
import evaluation

cfg = data_loading.load_config()

print("Reloading raw accepted-loans data (full matured population, 2007-2015)...")
df = data_loading.load_accepted("data/raw/accepted_loans.csv")
df = data_loading.filter_to_terminal_population(df, cfg)
df = data_loading.filter_to_matured_population(df, cfg)
df["year"] = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
print(f"Full matured population: {df.shape}")

feature_cols = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_years", "fico_midpoint",
]

df_feat = features.engineer_features(df)
X = df_feat[feature_cols]

print("\nScoring every loan with the champion model...")
champion = joblib.load("data/processed/champion_model_calibrated.joblib")
df_feat["calibrated_pd"] = champion.predict_proba(X)[:, 1]

# Baseline: the training population's score distribution (2007-2013)
baseline_scores = df_feat.loc[df_feat["year"] <= cfg["split"]["train_end_year"], "calibrated_pd"].values

print("\n--- PSI by vintage year, vs. training population baseline (2007-2013) ---")
print(f"{'Year':<6}{'N loans':>10}{'Mean PD':>10}{'PSI':>10}  Status")
results = []
for year in sorted(df_feat["year"].unique()):
    year_scores = df_feat.loc[df_feat["year"] == year, "calibrated_pd"].values
    psi_value = evaluation.psi(baseline_scores, year_scores)
    status = "STABLE" if psi_value < 0.10 else ("INVESTIGATE" if psi_value < 0.25 else "SIGNIFICANT SHIFT")
    print(f"{year:<6}{len(year_scores):>10,}{year_scores.mean():>10.3f}{psi_value:>10.4f}  {status}")
    results.append({"year": year, "n_loans": len(year_scores), "mean_pd": year_scores.mean(),
                     "psi": psi_value, "status": status})

results_df = pd.DataFrame(results)
results_df.to_csv("reports/figures/phase6_psi_by_vintage.csv", index=False)
print("\nSaved to reports/figures/phase6_psi_by_vintage.csv")

print("\n--- Real-world context: does the PSI pattern line up with the 2008 crisis? ---")
print("2007-2008: pre/early crisis onset")
print("2009-2010: crisis trough / early recovery")
print("2011+: post-crisis recovery and platform scaling")
