# Credit Expansion Engine

An AI credit decisioning system that does three things a plain "predict default" classifier doesn't:

1. **Scores default risk** on approved-loan outcomes (a standard PD model, done rigorously — calibrated, validated, cost-aware)
2. **Finds safe-to-approve applicants in the declined pool** — the "second-look" idea: quantify how many rejected applicants look statistically like good borrowers, and what approving them would do to the bank's loss rate and profit
3. **Explains every decision** with SHAP-driven, plain-language reason codes — the ECOA-required adverse-action explanation, generated automatically rather than templated

This mirrors what Upstart and Zest AI are running commercially in 2026 (expanded approvals at equal or lower loss rates), built end-to-end on public data so every step is inspectable.

## Project structure

```
lending_ai/
├── data/
│   ├── raw/            # original downloaded CSVs (accepted + rejected loans)
│   └── processed/       # cleaned, feature-engineered parquet files
├── notebooks/           # exploratory + narrative notebooks, one per phase
├── src/                 # importable pipeline code
│   ├── data_loading.py      # ingest + schema validation for accepted/rejected files
│   ├── features.py          # feature engineering, shared between accepted & rejected schemas
│   ├── modeling.py          # PD model training, tuning, model comparison
│   ├── calibration.py       # isotonic/Platt calibration + calibration diagnostics
│   ├── evaluation.py        # AUC/KS/Gini, PSI, backtesting utilities
│   ├── expansion.py         # second-look scoring of the declined pool
│   ├── pricing.py           # expected-profit threshold & rate optimization
│   ├── explainability.py    # SHAP values -> plain-language reason codes
│   └── fairness.py          # disparate impact testing across proxy groups
├── reports/
│   └── figures/          # exported charts for the final writeup
├── tests/                # unit tests for src/
└── requirements.txt
```

## Roadmap

### Phase 1 — Data & foundation (days 1-3)
- Ingest accepted + rejected loan files, document schema and known limitations (see prior discussion: rejected file only has ~9 shared fields with accepted)
- Data quality audit: missingness, leakage checks (exclude any post-origination fields like `total_pymnt`, `recoveries`, `last_pymnt_d` from the feature set — these wouldn't be known at application time)
- Define the modeling population: filter to loans with a terminal status (Fully Paid / Charged Off), exclude in-progress loans
- Train/validation/test split, stratified and **time-based** (train on earlier vintages, test on later ones — this is closer to how a real deployment would be validated than a random split)

### Phase 2 — PD model development (days 4-8)
- Baseline logistic regression (interpretable) + gradient boosting (XGBoost/LightGBM) comparison
- Hyperparameter tuning (Optuna) with proper cross-validation
- Discriminatory power validation: AUC, KS, Gini on the time-based holdout
- Isotonic calibration on a held-out calibration slice; Brier score before/after

### Phase 3 — Second-look expansion model (days 9-13)
- Identify the ~9 fields common to accepted and rejected files
- **Reject inference / selection bias correction** — the core methodological piece of this phase. A model trained only on accepted loans and applied to declines conflates the bank's original accept/decline policy with the applicant's true default risk (classic Heckman-style sample selection bias). Two corrections are fit and compared (`src/reject_inference.py`):
  - *Fuzzy augmentation* — score declines with the accepts-only model, re-inject them as soft-labeled weighted records, refit
  - *Heckman two-stage* — probit selection equation over all applicants (accept vs. decline) → inverse Mills ratio → included as a regressor in the outcome equation, correcting the coefficients themselves rather than just the population
  - Deliverable: a naive-vs-corrected comparison table quantifying how much the "safe to approve" population and its risk estimate shift once selection bias is corrected — this is the evidence that reject inference mattered, not just a methodology footnote
- Score the declined population using the corrected model
- Define "safe to approve": declined applicants whose corrected predicted PD falls below the bank's approval threshold
- Quantify the business case: estimated additional approved volume, expected incremental defaults, expected incremental profit
- Sensitivity analysis: how the safe-to-approve population size changes as the threshold moves, and how that differs between the naive and corrected models

### Phase 4 — Pricing & decision policy (days 14-16)
- Convert PD + loan terms into an expected-profit calculation per applicant
- Cost-based decision threshold (not a naive 0.5 cutoff) — extend the cost-matrix approach used in the German Credit notebook, calibrated to Lending Club's actual loss-given-default patterns
- Optional: a simple rate-optimization layer — given PD, what interest rate maximizes expected profit within a competitive rate band

### Phase 5 — Explainability & fairness (days 17-20)
- SHAP values per decision, mapped to plain-language reason codes (the adverse-action letter generator)
- Disparate impact testing across available proxy demographic signals (state/zip-based proxies, since this dataset has no direct protected-class fields — same caveat as the German Credit project, stated explicitly)
- Bias mitigation pass if disparities are found: threshold adjustment or reweighing, documented before/after

### Phase 6 — Monitoring, packaging & writeup (days 21-25)
- PSI-based stability monitoring across loan vintages (using the real 2007-2018 time range — includes the 2008 crisis, a genuine stress-test opportunity)
- Package the pipeline into a scorable function/API (FastAPI) — a single endpoint that takes an applicant profile and returns decision + reason codes
- Final model documentation writeup, governed by **SR 26-2** (the April 2026 interagency Revised Guidance on Model Risk Management, which superseded SR 11-7) rather than the older framework — extended with the expansion/pricing business case and an explicit note on where the reject-inference model and the SHAP explainability layer each sit under SR 26-2's tightened model definition and its generative-AI carveout

## Status
Scaffolding in place. Waiting on `data/raw/accepted_loans.csv` (and `rejected_loans.csv` if pursuing Phase 3) to begin Phase 1.
