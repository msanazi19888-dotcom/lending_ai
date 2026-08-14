"""
Reject inference: correcting for sample selection bias when extending a
PD model trained only on accepted loans to the declined population.

Why this matters: a model trained only on accepts is trained on a
population that was already filtered by the original accept/decline
decision. Applying it to declines without correction assumes the
feature-to-default relationship learned on accepts holds identically for
applicants the model never saw funded -- a strong and often false
assumption (classic Heckman-style selection bias). Two corrections are
implemented here so their effect can be compared directly:

  1. fuzzy_augmentation -- practical industry-standard technique. Score
     declines with the accepts-only model, then re-inject them into
     training as duplicated soft-labeled records (weighted by predicted
     good/bad probability), and refit.

  2. heckman_two_stage -- econometrically rigorous approach. Fit a probit
     "selection equation" (accepted vs. declined, over the population of
     ALL applicants) to get each accepted applicant's inverse Mills ratio,
     then include that ratio as a regressor in the outcome equation
     (default vs. not, over accepts only). This corrects the outcome
     model's coefficients for the fact that accepts are a
     non-random/selected sample, rather than correcting the *population*
     the way augmentation does.

Both are approximations of the joint bivariate probit MLE that is the
textbook-exact solution for a binary selection + binary outcome pair;
the two-step version here is the standard, defensible practical
implementation used across the credit-scoring literature (Crook &
Banasik 2004; Anderson 2007).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Probit
from scipy.stats import norm


# ---------------------------------------------------------------------
# 1. Fuzzy augmentation / parceling
# ---------------------------------------------------------------------

def fuzzy_augmentation(accepts_model, X_accepts: pd.DataFrame, y_accepts: pd.Series,
                        X_rejects: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Score declines with the accepts-only model, then duplicate each
    declined record twice -- once labeled 'good' and once labeled 'bad' --
    weighted by the model's predicted probability. Returns an augmented
    training set combining real accepts (weight 1) with soft-labeled
    declines, ready to refit a weighted classifier on.
    """
    reject_scores = accepts_model.predict_proba(X_rejects)[:, 1]  # P(default)

    good_half = X_rejects.copy()
    good_half["_label"] = 0
    good_half["_weight"] = 1 - reject_scores

    bad_half = X_rejects.copy()
    bad_half["_label"] = 1
    bad_half["_weight"] = reject_scores

    augmented_rejects = pd.concat([good_half, bad_half], ignore_index=True)

    accepts_frame = X_accepts.copy()
    accepts_frame["_label"] = y_accepts.values
    accepts_frame["_weight"] = 1.0

    combined = pd.concat([accepts_frame, augmented_rejects], ignore_index=True)
    weights = combined.pop("_weight").values
    labels = combined.pop("_label")

    return combined, labels, weights


def refit_with_augmentation(pipeline, X_augmented: pd.DataFrame, y_augmented: pd.Series,
                             weights: np.ndarray):
    """Refit a sklearn pipeline using the augmented, weighted training set.
    Requires the final estimator step to accept sample_weight."""
    return pipeline.fit(X_augmented, y_augmented, clf__sample_weight=weights)


# ---------------------------------------------------------------------
# 2. Heckman two-stage selection correction
# ---------------------------------------------------------------------

def fit_selection_equation(X_all_applicants: pd.DataFrame, was_accepted: pd.Series) -> Probit:
    """Stage 1: probit model of P(accepted | applicant features), fit over
    ALL applicants (accepts + declines). This is the 'selection equation' --
    it models the original accept/decline policy itself, not default risk.

    Uses a layered convergence strategy rather than statsmodels' bare
    defaults: the generic default (Newton's method, maxiter=35) is tuned
    for small textbook datasets and was found (Phase 3, real data) to fail
    to converge on ~1.5M rows / ~55 columns. Newton is tried first with a
    much larger iteration budget; if it still doesn't converge, BFGS and
    then L-BFGS are tried as more numerically robust fallbacks. Every
    attempt's convergence status is reported explicitly -- this function
    never silently returns an unconverged fit without saying so."""
    X_with_const = X_all_applicants.astype(float).copy()
    X_with_const.insert(0, "const", 1.0)
    model = Probit(was_accepted.values, X_with_const.values)

    result = model.fit(method="newton", maxiter=200, disp=False)
    if not result.mle_retvals.get("converged", False):
        print("[reject_inference] Newton (200 iter) did not converge -- retrying with BFGS...")
        result = model.fit(method="bfgs", maxiter=1000, disp=False)
    if not result.mle_retvals.get("converged", False):
        print("[reject_inference] BFGS did not converge -- retrying with L-BFGS (larger budget)...")
        result = model.fit(method="lbfgs", maxiter=3000, disp=False)

    converged = result.mle_retvals.get("converged", False)
    print(f"[reject_inference] Selection equation final convergence status: {converged}")
    if not converged:
        print("[reject_inference] WARNING: none of the three methods converged. "
              "Results should be treated as provisional -- do not trust downstream "
              "business-case numbers without investigating further (e.g. checking "
              "for near-perfect separation in a rare state category).")
    return result


def inverse_mills_ratio(selection_model: Probit, X_all_applicants: pd.DataFrame) -> np.ndarray:
    """Compute the inverse Mills ratio (lambda) for each applicant from the
    fitted selection equation: lambda_i = phi(x_i*beta) / Phi(x_i*beta).
    This is the correction term that captures 'how selected' each
    applicant's acceptance was."""
    X_with_const = X_all_applicants.astype(float).copy()
    X_with_const.insert(0, "const", 1.0)
    linear_pred = X_with_const.values @ selection_model.params
    return norm.pdf(linear_pred) / np.clip(norm.cdf(linear_pred), 1e-6, None)


def heckman_two_stage(X_all_applicants: pd.DataFrame, was_accepted: pd.Series,
                       X_accepts: pd.DataFrame, y_default: pd.Series) -> dict:
    """Full two-stage procedure. Returns the fitted selection model, the
    inverse Mills ratio for the accepted subsample, and a bias-corrected
    outcome model (logistic regression on accepts, with the Mills ratio
    included as an additional regressor) whose *other* coefficients are
    now corrected for selection bias -- this is the actual object of
    interest, not just prediction accuracy on rejects."""
    selection_model = fit_selection_equation(X_all_applicants, was_accepted)
    mills_all = inverse_mills_ratio(selection_model, X_all_applicants)

    # Restrict Mills ratio to the accepted subsample, aligned by index
    mills_accepts = pd.Series(mills_all, index=X_all_applicants.index).loc[X_accepts.index]

    X_outcome = X_accepts.copy()
    X_outcome["_inverse_mills_ratio"] = mills_accepts.values

    outcome_model = LogisticRegression(max_iter=2000)
    outcome_model.fit(X_outcome, y_default)

    mills_coef = outcome_model.coef_[0][-1]

    return {
        "selection_model": selection_model,
        "outcome_model": outcome_model,
        "mills_ratio_accepts": mills_accepts,
        "mills_ratio_coefficient": mills_coef,
        "selection_bias_detected": abs(mills_coef) > 0.1,  # rule-of-thumb flag, not a formal test
    }


def compare_naive_vs_corrected(naive_model, corrected_result: dict,
                                X_rejects: pd.DataFrame) -> pd.DataFrame:
    """Score the same declined pool with both the naive (accepts-only) model
    and the Heckman-corrected outcome model, to quantify how much the
    'safe to approve' population changes once selection bias is corrected.
    This comparison table is the key Phase 3 deliverable -- it's the
    evidence that reject inference mattered, not just a methodology note."""
    naive_scores = naive_model.predict_proba(X_rejects)[:, 1]

    X_rejects_mills = X_rejects.copy()
    selection_model = corrected_result["selection_model"]
    reject_mills = inverse_mills_ratio(selection_model, X_rejects)
    X_rejects_mills["_inverse_mills_ratio"] = reject_mills
    corrected_scores = corrected_result["outcome_model"].predict_proba(X_rejects_mills)[:, 1]

    return pd.DataFrame({
        "naive_predicted_pd": naive_scores,
        "corrected_predicted_pd": corrected_scores,
        "pd_shift": corrected_scores - naive_scores,
    })
