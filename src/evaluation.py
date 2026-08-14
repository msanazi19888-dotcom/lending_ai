"""Model validation: discriminatory power, stability (PSI), backtesting."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp


def discriminatory_power(y_true, probs) -> dict:
    auc = roc_auc_score(y_true, probs)
    ks = ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic
    gini = 2 * auc - 1
    return {"auc": auc, "ks": ks, "gini": gini}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two score distributions.
    < 0.10 stable, 0.10-0.25 investigate, > 0.25 significant drift."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e_pct = np.clip(np.histogram(expected, breakpoints)[0] / len(expected), 1e-4, None)
    a_pct = np.clip(np.histogram(actual, breakpoints)[0] / len(actual), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_by_vintage(scores_by_year: dict[int, np.ndarray], baseline_year: int) -> dict[int, float]:
    """Run PSI of every vintage against a baseline year -- useful for the
    2008-crisis / COVID stress-test angle in the roadmap's Phase 6."""
    baseline = scores_by_year[baseline_year]
    return {year: psi(baseline, scores) for year, scores in scores_by_year.items() if year != baseline_year}
