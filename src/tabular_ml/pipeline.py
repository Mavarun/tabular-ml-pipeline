"""Sklearn pipelines that keep preprocessing inside the estimator."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_pipeline(
    numeric_features: list[str] | list[int] | None = None,
    C: float = 1.0,
    random_state: int = 0,
) -> Pipeline:
    """Logistic regression with impute + scale fitted only on train folds.

    If ``numeric_features`` is None, all columns are treated as numeric
    (works for sklearn toy matrices and DataFrames of floats).
    """
    if numeric_features is None:
        preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                )
            ],
            remainder="drop",
        )

    clf = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )
    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])


def build_boosting_pipeline(
    max_depth: int = 3,
    learning_rate: float = 0.1,
    max_iter: int = 100,
    random_state: int = 0,
) -> Pipeline:
    """HistGradientBoosting with median impute (no scale needed).

    Kept as a second model so the baseline comparison is real, not
    "logistic vs itself".
    """
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )
    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
