import sys
sys.path.insert(0, "src")
import data_loading

cfg = data_loading.load_config()

print("Loading accepted loans (this may take a minute)...")
df = data_loading.load_accepted("data/raw/accepted_loans.csv")
print("Raw shape:", df.shape)

df = data_loading.filter_to_terminal_population(df, cfg)
print("Terminal-status population:", df.shape)

df = data_loading.filter_to_matured_population(df, cfg)
print("Matured population (36mo, issued <=2015):", df.shape)
print("Default rate: {:.1%}".format(df["default"].mean()))

df = data_loading.drop_leakage_fields(df, cfg)
df = data_loading.drop_identifier_fields(df, cfg)
print("Shape after dropping leakage/identifier fields:", df.shape)

train, calib, test = data_loading.time_based_split(df, cfg)
print("\nTime-based split:")
print("  Train:", train.shape, "default rate: {:.1%}".format(train["default"].mean()))
print("  Calib:", calib.shape, "default rate: {:.1%}".format(calib["default"].mean()))
print("  Test: ", test.shape, "default rate: {:.1%}".format(test["default"].mean()))

# Missingness audit on the training set -- determines which columns are
# usable as features vs. too sparse to trust.
print("\nMissingness (train set), worst 20 columns:")
missing_pct = train.isna().mean().sort_values(ascending=False)
print((missing_pct.head(20) * 100).round(1))

print("\nColumns with >50% missing (candidates to exclude):")
print(missing_pct[missing_pct > 0.5].index.tolist())

# Save the split for Phase 2 -- avoids re-running this ~2 minute load/filter
# pipeline every time we want to iterate on modeling.
import os
os.makedirs("data/processed", exist_ok=True)
train.to_parquet("data/processed/train.parquet", index=False)
calib.to_parquet("data/processed/calib.parquet", index=False)
test.to_parquet("data/processed/test.parquet", index=False)
print("\nSaved train/calib/test to data/processed/ as parquet.")