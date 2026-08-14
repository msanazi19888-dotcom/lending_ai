"""PD model training and comparison across model families."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import pandas as pd
import numpy as np


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric and categorical branches each get imputation before their
    normal transform. Phase 1's missingness audit found real gaps (e.g.
    emp_length at 4.3%, revol_util at 0.1%) -- logistic regression and
    gradient boosting cannot handle NaN directly, so this isn't optional."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])


def candidate_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Standard candidate set: interpretable baseline + three tree ensembles
    spanning two genuinely distinct ensemble mechanisms. Logistic regression
    is included even though it may lose on raw AUC -- see the German Credit
    project notes on why that's not automatically a reason to discard it for
    a regulated use case. NOTE: gradient_boosting and xgboost are both
    BOOSTING methods (sequential, error-correcting) -- they are not
    independent evidence of "model capacity doesn't matter" on their own.
    random_forest uses BAGGING (parallel, variance-reducing) instead, a
    genuinely different mechanism, added specifically to test whether the
    Phase 2 "feature ceiling, not model ceiling" conclusion holds up
    against a third, structurally different model family."""
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                                learning_rate=0.05, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("prep", preprocessor),
            ("clf", xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       eval_metric="auc", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                            min_samples_leaf=20, class_weight="balanced",
                                            random_state=42, n_jobs=-1)),
        ]),
    }


def fit_all(models: dict[str, Pipeline], X_train, y_train) -> dict[str, Pipeline]:
    fitted = {}
    for name, pipe in models.items():
        print(f"[modeling] Fitting {name}...")
        fitted[name] = pipe.fit(X_train, y_train)
    return fitted


def drop_zero_variance_features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop any feature with only one distinct non-null value -- carries no
    signal and, for categorical columns, one-hot encodes into a useless
    constant column. Prints what it drops so this is never a silent
    surprise (this caught 'term'/'term_months' becoming constant after the
    Phase 1 maturity filter restricted the population to 36-month loans)."""
    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var:
        print(f"[modeling] Dropping zero-variance features: {zero_var}")
    return X.drop(columns=zero_var)


# ---------------------------------------------------------------------
# Hyperparameter tuning
#
# Tuned on a subsample for tractability (hyperparameter RANKINGS are
# generally stable across sample size even though absolute CV scores
# shift slightly), then the winning config is refit on the full training
# set for the real reported numbers. This is a deliberate compute/rigor
# tradeoff, documented rather than hidden.
# ---------------------------------------------------------------------

def _subsample(X: pd.DataFrame, y: pd.Series, n: int, random_state: int = 42):
    if len(X) <= n:
        return X, y
    idx = X.sample(n=n, random_state=random_state).index
    return X.loc[idx], y.loc[idx]


def tune_logistic_regression(X, y, n_trials=15, cv_folds=3, subsample_n=50_000, random_state=42):
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_s, y_s = _subsample(X, y, subsample_n, random_state)
    preprocessor = build_preprocessor(X_s)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear"  # supports both l1 and l2
        pipe = Pipeline([("prep", preprocessor),
                          ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver,
                                                      max_iter=2000, class_weight="balanced"))])
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_s, y_s, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="logistic_regression")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def tune_gradient_boosting(X, y, n_trials=15, cv_folds=3, subsample_n=50_000, random_state=42):
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_s, y_s = _subsample(X, y, subsample_n, random_state)
    preprocessor = build_preprocessor(X_s)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        }
        pipe = Pipeline([("prep", preprocessor),
                          ("clf", GradientBoostingClassifier(random_state=random_state, **params))])
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_s, y_s, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="gradient_boosting")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def tune_xgboost(X, y, n_trials=15, cv_folds=3, subsample_n=50_000, random_state=42):
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_s, y_s = _subsample(X, y, subsample_n, random_state)
    preprocessor = build_preprocessor(X_s)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        pipe = Pipeline([("prep", preprocessor),
                          ("clf", xgb.XGBClassifier(eval_metric="auc", random_state=random_state, **params))])
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_s, y_s, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="xgboost")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def tune_random_forest(X, y, n_trials=15, cv_folds=3, subsample_n=50_000, random_state=42):
    """Random Forest uses BAGGING (parallel trees on bootstrap samples,
    variance reduction), a genuinely different ensemble mechanism from
    gradient_boosting/xgboost's sequential error-correction. Added to give
    the "feature ceiling, not model ceiling" conclusion real independent
    evidence, rather than resting on two implementations of the same
    underlying method."""
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_s, y_s = _subsample(X, y, subsample_n, random_state)
    preprocessor = build_preprocessor(X_s)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        pipe = Pipeline([("prep", preprocessor),
                          ("clf", RandomForestClassifier(class_weight="balanced", random_state=random_state,
                                                          n_jobs=-1, **params))])
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X_s, y_s, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="random_forest")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def refit_best_on_full_data(model_name: str, best_params: dict, X_full, y_full, random_state=42):
    """Take the winning hyperparameters from tuning (found on the
    subsample) and refit on the FULL training set -- this is what actually
    gets evaluated and reported, not the subsample CV score."""
    preprocessor = build_preprocessor(X_full)

    if model_name == "logistic_regression":
        clf = LogisticRegression(solver="liblinear", max_iter=2000,
                                  class_weight="balanced", **best_params)
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=random_state, **best_params)
    elif model_name == "xgboost":
        clf = xgb.XGBClassifier(eval_metric="auc", random_state=random_state, **best_params)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(class_weight="balanced", random_state=random_state,
                                      n_jobs=-1, **best_params)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    return pipe.fit(X_full, y_full)
