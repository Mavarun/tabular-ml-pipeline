"""Probability calibration metrics for research evaluation."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE for binary probabilities in [0, 1].

    Bins predicted probabilities, then averages |confidence - accuracy|
    weighted by bin mass. Not a production guarantee — just a diagnostic.
    """
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive last bin so 1.0 is counted
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    ece = 0.0
    n = y_true.size
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
