"""Cost/profit-based decision thresholds and rate optimization."""
import numpy as np
from sklearn.metrics import confusion_matrix


def cost_minimizing_threshold(y_true, probs, fn_cost: float, fp_cost: float,
                               grid: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Scan thresholds and return the one minimizing total expected cost,
    rather than defaulting to a naive 0.5 cutoff. Same approach as the
    German Credit notebook, parameterized here for reuse."""
    if grid is None:
        grid = np.linspace(0.02, 0.98, 97)

    costs = []
    for t in grid:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        costs.append(fn * fn_cost + fp * fp_cost)

    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    return grid[best_idx], costs


def amortized_interest_fraction(annual_rate: float, term_months: int) -> float:
    """Total interest collected over a fully-amortizing loan's life, as a
    fraction of principal -- accounts for the declining outstanding balance
    as monthly payments are made, unlike a naive rate x years calculation.
    Same formula validated in Phase 3 (12.8% APR, 36 months -> 0.2095,
    vs. a naive calculation that would overstate this as 0.384)."""
    r = annual_rate / 12
    if r == 0:
        return 0.0
    factor = r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)
    return factor * term_months - 1


def expected_profit(pd_estimate: float, loan_amount: float, interest_rate: float,
                     term_years: float, lgd: float) -> float:
    """Expected profit for a single loan given its predicted PD and terms.

    FIX (Phase 4): previously used a naive interest_income = loan_amount *
    interest_rate * term_years, which ignores amortization -- the
    outstanding balance shrinks every month as principal is paid down, so
    total interest actually collected is well below a flat rate x years
    calculation. This was flagged in this function's own docstring as a
    known simplification to fix in Phase 4, and was accidentally still in
    use when Phase 4's rate-optimization step first ran, producing an
    implausible result (100% of loans "underpriced", ~$2.2B in phantom
    uplift) that was caught by a plausibility check against the total
    portfolio size, not a silent bug."""
    term_months = round(term_years * 12)
    interest_fraction = amortized_interest_fraction(interest_rate, term_months)
    interest_income = loan_amount * interest_fraction
    expected_loss = pd_estimate * lgd * loan_amount
    return (1 - pd_estimate) * interest_income - expected_loss


def optimal_rate(pd_estimate: float, loan_amount: float, term_years: float, lgd: float,
                  rate_grid: np.ndarray | None = None) -> tuple[float, float]:
    """Given an applicant's PD, find the interest rate within a competitive
    band that maximizes expected profit.

    KNOWN LIMITATION (Phase 4, not fixed -- scoped out deliberately): this
    will ALWAYS return a rate at or near rate_grid's maximum. expected_profit
    treats pd_estimate as fixed regardless of the rate charged, so profit is
    mathematically monotonically increasing in rate with no natural optimum
    -- there is no mechanism here representing that raising a borrower's
    rate can itself increase their risk of default (adverse selection /
    affordability strain), a well-documented real effect in lending.
    Fixing this properly requires estimating rate elasticity of default,
    which is NOT straightforward from this dataset: LendingClub's own rates
    were assigned based on their own risk assessment (grade/subgrade), so
    higher-rate loans defaulting more is confounded with pre-existing risk,
    not evidence that the rate itself caused the additional risk. A
    rigorous estimate would need something like within-subgrade rate
    variation as a natural experiment -- deliberately scoped out of this
    project rather than attempted hastily and reported as if reliable.
    This function is kept for reference/future extension but its output
    should NOT be used as a real pricing recommendation as currently built.
    """
    if rate_grid is None:
        rate_grid = np.arange(0.06, 0.30, 0.005)

    profits = [expected_profit(pd_estimate, loan_amount, r, term_years, lgd) for r in rate_grid]
    best_idx = int(np.argmax(profits))
    return rate_grid[best_idx], profits[best_idx]
