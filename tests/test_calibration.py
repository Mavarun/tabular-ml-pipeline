import numpy as np
import pytest

from tabular_ml.calibration import expected_calibration_error


def test_ece_perfectly_calibrated_is_near_zero():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0.0, 1.0, size=2000)
    y_true = (rng.uniform(0.0, 1.0, size=2000) < y_prob).astype(float)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05


def test_ece_constant_wrong_confidence_is_high():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    y_prob = np.full(8, 0.9)
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece > 0.3


def test_ece_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.1]))
