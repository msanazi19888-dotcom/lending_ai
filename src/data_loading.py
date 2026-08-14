"""
Ingestion and schema validation for LendingClub accepted/rejected loan files.

The accepted and rejected files have almost no overlapping columns (see
project README) -- this module keeps that distinction explicit rather than
silently joining or reshaping them into a false common schema.
"""
from pathlib import Path
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_accepted(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load the accepted-loans file. Large file -- nrows lets you prototype
    on a slice before running the full dataset."""
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = normalize_whitespace(df)
    return df


def normalize_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """LendingClub's raw export has leading/trailing whitespace on several
    text columns (discovered in Phase 1: 'term' comes through as
    ' 36 months' with a leading space, which silently breaks exact-match
    filters without raising an error). Strip whitespace on every object
    (text) column rather than patching around it column-by-column."""
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].str.strip()
    return df


def load_rejected(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load the rejected-loans file. Expect far fewer columns than accepted --
    do not assume schema parity."""
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def filter_to_terminal_population(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only loans with a known, final outcome. Loans still 'Current' or
    in a grace/late bucket have no resolved label yet and would either leak
    information or introduce label noise if included."""
    good = config["terminal_statuses"]["good"]
    bad = config["terminal_statuses"]["bad"]
    mask = df["loan_status"].isin(good + bad)
    out = df.loc[mask].copy()
    out["default"] = out["loan_status"].isin(bad).astype(int)
    return out


def filter_to_matured_population(df: pd.DataFrame, config: dict,
                                  date_col: str = "issue_d") -> pd.DataFrame:
    """Restrict to loans that had time to reach natural maturity by the
    dataset's snapshot date (see config.yaml modeling_scope for the full
    rationale). Must be called with a 'term' column already whitespace-
    normalized (see normalize_whitespace / load_accepted)."""
    scope = config["modeling_scope"]
    dates = pd.to_datetime(df[date_col], format="%b-%Y")
    mask = (df["term"] == scope["term_filter"]) & (dates.dt.year <= scope["max_issue_year"])
    return df.loc[mask].copy()


def drop_leakage_fields(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop fields that are only known post-origination -- keeping the
    modeling feature set honestly restricted to application-time information."""
    to_drop = [c for c in config["leakage_fields"] if c in df.columns]
    return df.drop(columns=to_drop)


def drop_identifier_fields(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop unique IDs and free-text fields -- not leakage in the temporal
    sense, but not usable structured features either."""
    to_drop = [c for c in config.get("identifier_fields", []) if c in df.columns]
    return df.drop(columns=to_drop)


def time_based_split(df: pd.DataFrame, config: dict, date_col: str = "issue_d"):
    """Split by loan issue date rather than randomly -- train on earlier
    vintages, validate/test on later ones. More representative of how the
    model would actually be evaluated before a production deployment."""
    dates = pd.to_datetime(df[date_col], format="%b-%Y")
    year = dates.dt.year

    train = df[year <= config["split"]["train_end_year"]]
    calib = df[(year > config["split"]["train_end_year"]) &
               (year <= config["split"]["calib_end_year"])]
    test = df[year >= config["split"]["test_start_year"]]

    return train, calib, test


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded. Waiting on data/raw/accepted_loans.csv to proceed.")
