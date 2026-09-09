"""Tests for reliability bins / ECE properties (slice 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tabular_ml.calibration import (
    expected_calibration_error,
    format_reliability_ascii,
    reliability_bins,
    save_reliability_diagram,
)
from tabular_ml.data import load_covertype_grouped
from tabular_ml.evaluate import compare_calibration_methods, evaluate_grouped_cv
from tabular_ml.pipeline import build_baseline_pipeline


def test_reliability_bins_match_ece():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0.0, 1.0, size=500)
    y_true = (rng.uniform(0.0, 1.0, size=500) < y_prob).astype(float)
    diagram = reliability_bins(y_true, y_prob, n_bins=10)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert diagram.n_bins == 10
    assert diagram.bin_count.sum() == 500
    assert abs(diagram.ece - ece) < 1e-12


def test_reliability_bins_perfect_calibration_low_ece():
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(0.05, 0.95, size=3000)
    y_true = (rng.uniform(0.0, 1.0, size=3000) < y_prob).astype(float)
    diagram = reliability_bins(y_true, y_prob, n_bins=10)
    assert diagram.ece < 0.05


def test_reliability_bins_systematic_bias_high_ece():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    y_prob = np.full(8, 0.95)
    diagram = reliability_bins(y_true, y_prob, n_bins=5)
    assert diagram.ece > 0.3
    assert diagram.bin_count.sum() == 8


def test_reliability_bins_empty_bin_is_nan():
    # All probs in [0, 0.2] → higher bins empty.
    y_prob = np.array([0.05, 0.1, 0.15, 0.08], dtype=float)
    y_true = np.array([0, 0, 1, 0], dtype=float)
    diagram = reliability_bins(y_true, y_prob, n_bins=5)
    assert diagram.bin_count[0] == 4
    assert diagram.bin_count[4] == 0
    assert np.isnan(diagram.bin_confidence[4])
    assert np.isnan(diagram.bin_accuracy[4])


def test_reliability_ascii_mentions_ece():
    y_true = np.array([0, 1, 0, 1], dtype=float)
    y_prob = np.array([0.2, 0.8, 0.3, 0.7], dtype=float)
    text = format_reliability_ascii(reliability_bins(y_true, y_prob, n_bins=4))
    assert "ECE=" in text
    assert "bin" in text


def test_save_reliability_diagram_writes_file(tmp_path: Path):
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=float)
    out = tmp_path / "rel.png"
    path = save_reliability_diagram(
        reliability_bins(y_true, y_prob, n_bins=5),
        out,
        title="unit",
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_covertype_loader_has_multiple_groups():
    X, y, groups, names = load_covertype_grouped(
        n_samples=800, random_state=0, min_group_size=20
    )
    assert X.shape[0] == 800
    assert X.shape[1] == 14
    assert set(np.unique(y)).issubset({0, 1})
    assert np.unique(groups).size >= 5
    assert len(names) == 14
    # Soil one-hots must not appear as features (group leakage).
    assert X.shape[1] < 54


def test_compare_calibration_methods_runs_on_covertype_subsample():
    X, y, groups, _ = load_covertype_grouped(
        n_samples=1200, random_state=2, min_group_size=25
    )
    # Need enough groups for outer 3-fold + inner calibration.
    if np.unique(groups).size < 6:
        pytest.skip("too few soil groups in subsample")
    results = compare_calibration_methods(
        build_baseline_pipeline(),
        X,
        y,
        groups,
        n_splits=3,
        methods=("raw", "sigmoid"),
        n_bins=10,
    )
    assert "raw" in results and "sigmoid" in results
    assert results["raw"].reliability is not None
    assert results["raw"].y_prob_oof is not None
    assert results["raw"].mean_ece >= 0.0


def test_evaluate_return_oof_shapes():
    X, y, groups, _ = load_covertype_grouped(
        n_samples=1000, random_state=3, min_group_size=25
    )
    n_splits = 3 if np.unique(groups).size >= 3 else 2
    result = evaluate_grouped_cv(
        build_baseline_pipeline(),
        X,
        y,
        groups,
        n_splits=n_splits,
        calibrate=False,
        return_oof=True,
    )
    assert result.y_true_oof is not None
    assert result.y_true_oof.shape == result.y_prob_oof.shape
    assert result.y_true_oof.shape[0] == X.shape[0]
