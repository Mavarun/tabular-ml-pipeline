"""Public toy data with synthetic entity groups for grouped CV demos."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_grouped(
    n_groups: int = 40,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """sklearn breast_cancer with synthetic patient/site group ids.

    Groups are assigned so nearby rows (similar feature space after a
    shuffle) tend to share a group — enough to make random KFold leak
    across "patients" while GroupKFold keeps them OOS. This is a demo
    construct, not a clinical claim.
    """
    bunch = load_breast_cancer()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=int)
    feature_names = list(bunch.feature_names)

    rng = np.random.default_rng(random_state)
    order = rng.permutation(X.shape[0])
    X = X[order]
    y = y[order]

    # Chunk into n_groups contiguous blocks after shuffle.
    groups = np.zeros(X.shape[0], dtype=int)
    edges = np.linspace(0, X.shape[0], n_groups + 1, dtype=int)
    for g in range(n_groups):
        groups[edges[g] : edges[g + 1]] = g

    return X, y, groups, feature_names
