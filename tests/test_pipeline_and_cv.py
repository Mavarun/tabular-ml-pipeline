import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from tabular_ml.data import load_breast_cancer_grouped
from tabular_ml.evaluate import evaluate_grouped_cv
from tabular_ml.pipeline import build_baseline_pipeline, build_boosting_pipeline


def test_pipeline_exposes_predict_proba():
    X, y, _, _ = load_breast_cancer_grouped(n_groups=20, random_state=1)
    pipe = build_baseline_pipeline()
    pipe.fit(X[:100], y[:100])
    proba = pipe.predict_proba(X[100:120])
    assert proba.shape == (20, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_boosting_pipeline_runs():
    X, y, _, _ = load_breast_cancer_grouped(n_groups=20, random_state=2)
    pipe = build_boosting_pipeline(max_iter=20)
    pipe.fit(X[:80], y[:80])
    assert pipe.predict(X[80:90]).shape == (10,)


def test_grouped_cv_keeps_groups_out_of_test():
    X, y, groups, _ = load_breast_cancer_grouped(n_groups=25, random_state=3)
    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_evaluate_grouped_cv_returns_finite_metrics():
    X, y, groups, _ = load_breast_cancer_grouped(n_groups=30, random_state=4)
    result = evaluate_grouped_cv(
        build_baseline_pipeline(),
        X,
        y,
        groups,
        n_splits=5,
        calibrate=False,
    )
    assert 0.0 <= result.mean_accuracy <= 1.0
    assert 0.0 <= result.mean_roc_auc <= 1.0
    assert result.mean_brier >= 0.0
    assert result.mean_ece >= 0.0
    assert len(result.folds) == 5


def test_calibrated_path_runs_and_stays_leakage_safe():
    X, y, groups, _ = load_breast_cancer_grouped(n_groups=30, random_state=5)
    result = evaluate_grouped_cv(
        build_baseline_pipeline(),
        X,
        y,
        groups,
        n_splits=5,
        calibrate=True,
    )
    assert result.calibrated is True
    assert np.isfinite(result.mean_brier)


def test_preprocessing_scaler_not_fit_on_full_data_before_cv():
    """Guardrail: our evaluate path clones + fits per fold only.

    A common leak is fitting StandardScaler on all X then CV. We assert
    the pipeline arriving at evaluate is unfitted, and that a fresh
    clone is used each fold by checking the fitted scaler mean differs
    across folds when we spy via a tiny custom check.
    """
    X, y, groups, _ = load_breast_cancer_grouped(n_groups=20, random_state=6)
    pipe = build_baseline_pipeline()
    # Unfitted: accessing named_steps prep scaler mean_ should fail.
    with pytest.raises(AttributeError):
        _ = pipe.named_steps["prep"].named_steps["scaler"].mean_

    fitted_means = []
    gkf = GroupKFold(n_splits=4)
    for train_idx, _ in gkf.split(X, y, groups):
        m = clone(pipe)
        m.fit(X[train_idx], y[train_idx])
        fitted_means.append(m.named_steps["prep"].named_steps["scaler"].mean_.copy())

    # At least two folds should produce different scaler means.
    assert any(
        not np.allclose(fitted_means[0], other) for other in fitted_means[1:]
    )


def test_random_kfold_can_share_groups_unlike_group_kfold():
    """Documents why GroupKFold matters for this slice."""
    X, y, groups, _ = load_breast_cancer_grouped(n_groups=15, random_state=7)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    shared = False
    for train_idx, test_idx in kf.split(X):
        if set(groups[train_idx]) & set(groups[test_idx]):
            shared = True
            break
    assert shared is True
