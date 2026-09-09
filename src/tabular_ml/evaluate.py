"""Grouped CV evaluation that never fits prep on the full dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from tabular_ml.calibration import (
    ReliabilityDiagram,
    expected_calibration_error,
    reliability_bins,
)


@dataclass
class FoldMetrics:
    fold: int
    accuracy: float
    roc_auc: float
    brier: float
    ece: float
    n_train: int
    n_test: int


@dataclass
class CVResult:
    folds: list[FoldMetrics]
    mean_accuracy: float
    mean_roc_auc: float
    mean_brier: float
    mean_ece: float
    calibrated: bool
    calibration_method: str | None = None
    y_true_oof: np.ndarray | None = field(default=None, repr=False)
    y_prob_oof: np.ndarray | None = field(default=None, repr=False)
    reliability: ReliabilityDiagram | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "calibrated": self.calibrated,
            "calibration_method": self.calibration_method,
            "mean_accuracy": self.mean_accuracy,
            "mean_roc_auc": self.mean_roc_auc,
            "mean_brier": self.mean_brier,
            "mean_ece": self.mean_ece,
            "folds": [f.__dict__ for f in self.folds],
        }
        if self.reliability is not None:
            out["reliability"] = self.reliability.as_dict()
        return out


class GroupAwareCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """CalibratedClassifierCV that always receives ``groups`` for GroupKFold.

    sklearn 1.x metadata routing can drop ``groups`` silently. This wrapper
    precomputes GroupKFold splits from the fit-time groups and hands those
    index pairs to CalibratedClassifierCV, so calibration never sees the
    outer test groups.
    """

    def __init__(
        self,
        estimator: Any = None,
        method: str = "sigmoid",
        n_splits: int = 3,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.n_splits = n_splits

    def fit(self, X, y, groups):
        if self.estimator is None:
            raise ValueError("estimator is required")
        groups = np.asarray(groups).ravel()
        n_groups = np.unique(groups).size
        n_inner = min(self.n_splits, n_groups)
        if n_inner < 2:
            raise ValueError("need >=2 groups to calibrate with GroupKFold")
        splits = list(GroupKFold(n_splits=n_inner).split(X, y, groups))
        self.calibrated_ = CalibratedClassifierCV(
            estimator=clone(self.estimator),
            method=self.method,
            cv=splits,
        )
        self.calibrated_.fit(X, y)
        self.classes_ = self.calibrated_.classes_
        return self

    def predict_proba(self, X):
        return self.calibrated_.predict_proba(X)

    def predict(self, X):
        return self.calibrated_.predict(X)


def evaluate_grouped_cv(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
    random_state: int = 0,
    return_oof: bool = False,
    n_bins: int = 10,
) -> CVResult:
    """Score an estimator with GroupKFold; fit only on each train split.

    When ``calibrate`` is True, wraps the clone in GroupAwareCalibratedClassifier
    so inner calibration folds never see the outer test groups.

    When ``return_oof`` is True, also stores concatenated out-of-fold
    labels/probabilities and a pooled reliability diagram (useful for
    reliability PNGs and ECE beyond per-fold means).
    """
    del random_state  # reserved for future shuffled group variants
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    groups = np.asarray(groups).ravel()
    if X.shape[0] != y.shape[0] or y.shape[0] != groups.shape[0]:
        raise ValueError("X, y, and groups must share the same row count")

    unique_groups = np.unique(groups)
    if unique_groups.size < n_splits:
        raise ValueError(
            f"need at least {n_splits} groups for GroupKFold, got {unique_groups.size}"
        )

    if calibrate and calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be 'sigmoid' or 'isotonic'")

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: list[FoldMetrics] = []
    oof_true: list[np.ndarray] = []
    oof_prob: list[np.ndarray] = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        if calibrate:
            model = GroupAwareCalibratedClassifier(
                estimator=clone(estimator),
                method=calibration_method,
                n_splits=3,
            )
            model.fit(X_train, y_train, groups=groups_train)
        else:
            model = clone(estimator)
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_metrics.append(
            FoldMetrics(
                fold=fold_i,
                accuracy=float(accuracy_score(y_test, pred)),
                roc_auc=float(roc_auc_score(y_test, proba)),
                brier=float(brier_score_loss(y_test, proba)),
                ece=float(expected_calibration_error(y_test, proba, n_bins=n_bins)),
                n_train=int(train_idx.size),
                n_test=int(test_idx.size),
            )
        )
        if return_oof:
            oof_true.append(y_test)
            oof_prob.append(proba)

    y_true_oof = np.concatenate(oof_true) if oof_true else None
    y_prob_oof = np.concatenate(oof_prob) if oof_prob else None
    reliability = None
    if y_true_oof is not None and y_prob_oof is not None:
        reliability = reliability_bins(y_true_oof, y_prob_oof, n_bins=n_bins)

    return CVResult(
        folds=fold_metrics,
        mean_accuracy=float(np.mean([f.accuracy for f in fold_metrics])),
        mean_roc_auc=float(np.mean([f.roc_auc for f in fold_metrics])),
        mean_brier=float(np.mean([f.brier for f in fold_metrics])),
        mean_ece=float(np.mean([f.ece for f in fold_metrics])),
        calibrated=calibrate,
        calibration_method=calibration_method if calibrate else None,
        y_true_oof=y_true_oof,
        y_prob_oof=y_prob_oof,
        reliability=reliability,
    )


def compare_calibration_methods(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    methods: tuple[str, ...] = ("raw", "sigmoid", "isotonic"),
    n_bins: int = 10,
) -> dict[str, CVResult]:
    """GroupKFold eval for raw vs Platt (sigmoid) vs isotonic calibration.

    Reuses ``GroupAwareCalibratedClassifier`` so calibration folds never
    see outer test groups. Returns a dict keyed by method name.
    """
    results: dict[str, CVResult] = {}
    for method in methods:
        if method == "raw":
            results[method] = evaluate_grouped_cv(
                estimator,
                X,
                y,
                groups,
                n_splits=n_splits,
                calibrate=False,
                return_oof=True,
                n_bins=n_bins,
            )
        else:
            results[method] = evaluate_grouped_cv(
                estimator,
                X,
                y,
                groups,
                n_splits=n_splits,
                calibrate=True,
                calibration_method=method,
                return_oof=True,
                n_bins=n_bins,
            )
    return results
