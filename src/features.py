"""
Feature engineering for the PD model.

Two feature sets are maintained deliberately:
  - FULL_FEATURES: everything available in the accepted-loans file, used
    for the primary PD model.
  - SHARED_FEATURES: only fields that also exist in the rejected-loans file,
    used for the second-look expansion model so that comparison across
    accepted/declined populations is apples-to-apples.
"""
import pandas as pd
import numpy as np

# Populated once the real column names are confirmed against the
# downloaded file (Kaggle's column names vary slightly by dataset version).
FULL_FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "application_type",
]

# The genuinely shared fields between accepted and rejected files
# (per LendingClub's own published schema -- see BIS 2019 conference notes).
SHARED_FEATURES = [
    "loan_amnt",       # "Amount Requested" in the rejected file
    "dti",             # "Debt-To-Income Ratio"
    "emp_length",      # "Employment Length"
    "addr_state",      # "State"
    "zip_code",        # "Zip Code" (3-digit)
    "risk_score",      # "Risk Score" -- FICO/VantageScore, present in rejected file
]

# Real column names confirmed against the actual Kaggle rejected-loans file
# (Phase 3, step 1) -- these do NOT match LendingClub's documentation
# naming and bear no resemblance to the accepted file's snake_case columns.
# Keys are post-normalization (data_loading.load_rejected lowercases and
# replaces spaces with underscores, but preserves hyphens).
REJECTED_COLUMN_MAP = {
    "amount_requested": "loan_amnt",
    "debt-to-income_ratio": "dti",
    "employment_length": "emp_length",
    "state": "addr_state",
    "zip_code": "zip_code",
    "risk_score": "fico_midpoint",   # approximate mapping -- see caveat below
    "application_date": "issue_d",
}

# CAVEAT: LendingClub's rejected-file "Risk_Score" is not guaranteed to be
# the same underlying scoring model as the accepted file's FICO range
# across the whole time window (LendingClub is known to have changed
# scoring vendors/methodology over its history). Treated here as
# comparable to fico_midpoint for modeling purposes -- a documented
# limitation, not a verified equivalence.


def align_rejected_schema(df_rejected: pd.DataFrame) -> pd.DataFrame:
    """Rename the rejected file's real columns to the accepted file's
    naming convention, and clean the two fields that need it: dti often
    carries a trailing '%' as a string, and issue_d needs to match the
    accepted file's date parsing downstream."""
    out = df_rejected.rename(columns=REJECTED_COLUMN_MAP)

    if "dti" in out.columns:
        out["dti"] = pd.to_numeric(
            out["dti"].astype(str).str.replace("%", "").str.strip(), errors="coerce"
        )

    return out


def parse_emp_length(series: pd.Series) -> pd.Series:
    """Correctly parse LendingClub's emp_length categories to years.
    '< 1 year' -> 0, '1 year' -> 1, ..., '10+ years' -> 10.

    BUG FIX (found in Phase 3): a naive regex extracting the first digit
    from the string maps '< 1 year' to 1, identical to '1 year' -- because
    the regex has no way to see the '<' sign, it just grabs the '1'. This
    silently collapsed two distinct categories into one. This function
    checks for '< 1' explicitly before falling back to digit extraction.
    Note: this same bug existed in the version of engineer_features used
    for the Phase 2 champion model -- documented as a known, low-impact
    imprecision there (single low-cardinality feature) rather than
    triggering a full model re-run, but the fix applies going forward."""
    s = series.astype(str).str.strip()
    result = pd.Series(np.nan, index=s.index)
    result[s.str.contains("< 1", na=False)] = 0
    remaining = result.isna()
    digits = s.str.extract(r"(\d+)")[0].astype(float)
    result[remaining] = digits[remaining]
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a handful of standard credit-risk ratios on top of the raw
    fields. Kept separate from raw ingestion so it's easy to unit test."""
    out = df.copy()

    if "emp_length" in out.columns:
        out["emp_length_years"] = parse_emp_length(out["emp_length"])

    if {"fico_range_low", "fico_range_high"}.issubset(out.columns):
        out["fico_midpoint"] = (out["fico_range_low"] + out["fico_range_high"]) / 2

    if {"revol_bal", "revol_util"}.issubset(out.columns):
        # revol_util is already a utilization percentage in the raw data;
        # keep as-is but guard against string '%' artifacts.
        out["revol_util"] = pd.to_numeric(
            out["revol_util"].astype(str).str.replace("%", ""), errors="coerce"
        )

    if "term" in out.columns:
        out["term_months"] = out["term"].astype(str).str.extract(r"(\d+)").astype(float)

    return out


def filter_valid_shared_features(df: pd.DataFrame) -> pd.DataFrame:
    """Data quality filter for the shared accepted/rejected feature set --
    found necessary in Phase 3 after discovering corrupted values in the
    rejected file: dti values up to 50 million (nonsensical for a
    percentage), fico_midpoint of 0 or 990 (outside the valid 300-850
    range for FICO/VantageScore, almost certainly error/placeholder
    codes), and loan_amnt of 0 (not a valid application). Prints how many
    rows each filter removes so the cleaning is never a silent surprise."""
    out = df.copy()
    n0 = len(out)

    if "loan_amnt" in out.columns:
        out = out[out["loan_amnt"] > 0]
        print(f"[features] Dropped {n0 - len(out)} rows with loan_amnt <= 0")

    if "dti" in out.columns:
        n1 = len(out)
        # 0-100 is the plausible range for the vast majority of legitimate
        # applicants; values above this are treated as data errors rather
        # than genuine (if unusual) financial situations, given the
        # multi-million-percent outliers observed.
        out = out[(out["dti"] >= 0) & (out["dti"] <= 100)]
        print(f"[features] Dropped {n1 - len(out)} rows with dti outside [0, 100]")

    if "fico_midpoint" in out.columns:
        n2 = len(out)
        out = out[(out["fico_midpoint"] >= 300) & (out["fico_midpoint"] <= 850)]
        print(f"[features] Dropped {n2 - len(out)} rows with fico_midpoint outside [300, 850]")

    return out


def build_feature_matrix(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Select the modeling feature set, keeping only columns that actually
    exist -- avoids hard failures if a Kaggle version has slightly different
    column names, at the cost of a printed warning for visibility."""
    available = [c for c in feature_list if c in df.columns]
    missing = set(feature_list) - set(available)
    if missing:
        print(f"[features] Warning: {len(missing)} expected columns not found: {sorted(missing)}")
    return df[available]
