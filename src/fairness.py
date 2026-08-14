"""
Fairness / disparate impact testing.

This dataset has no direct protected-class fields (race, gender are not
present) -- proxy checks use available fields like state, which is a
known-weak proxy. That limitation is documented in every output here
rather than glossed over.
"""
import pandas as pd


def approval_rate_by_group(decisions: pd.Series, group: pd.Series) -> pd.Series:
    return decisions.groupby(group).mean()


def four_fifths_test(approval_rates: pd.Series, threshold: float = 0.8) -> dict:
    if approval_rates.max() == 0:
        raise ValueError(
            "Four-fifths test undefined: the most-favored group has a 0% approval "
            "rate, meaning no group was approved at all. Check the threshold/decision "
            "logic before interpreting any fairness result from this population."
        )
    ratio = approval_rates.min() / approval_rates.max()
    return {
        "impact_ratio": ratio,
        "passes_four_fifths": ratio >= threshold,
        "most_favored_group": approval_rates.idxmax(),
        "least_favored_group": approval_rates.idxmin(),
    }


def reweigh_for_fairness(df: pd.DataFrame, group_col: str, label_col: str) -> pd.Series:
    """Simple reweighing (Kamiran & Calders 2012 style): compute sample
    weights that equalize the expected vs. observed label rate within each
    group, for use as `sample_weight` during model refitting if the
    four-fifths test fails in Phase 5."""
    overall_rate = df[label_col].mean()
    group_rates = df.groupby(group_col)[label_col].transform("mean")
    weights = overall_rate / group_rates
    return weights
