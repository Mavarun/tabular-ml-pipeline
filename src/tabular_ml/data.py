"""Public datasets with entity groups for grouped CV demos."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_covtype, load_breast_cancer


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


def load_covertype_grouped(
    n_samples: int = 5000,
    random_state: int = 0,
    positive_class: int = 2,
    min_group_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Harder sklearn covertype subsample with *real* soil-type groups.

    Dataset
    -------
    ``sklearn.datasets.fetch_covtype`` (UCI Covertype). Much larger and
    less linearly separable than breast_cancer, so raw logistic/HGB
    probabilities are typically more miscalibrated (higher ECE).

    Groups
    ------
    Covertype columns 10–13 are wilderness-area one-hots and 14–53 are
    soil-type one-hots. We take ``groups = argmax(soil_type)`` — a real
    categorical present in the public table, not invented after labeling.
    Tiny soil types (< ``min_group_size`` rows in the subsample) are
    merged into a residual group so GroupKFold stays stable.

    Label
    -----
    Binary: ``cover_type == positive_class`` (default Lodgepole Pine = 2)
    vs rest. Features kept: 10 continuous cartographic vars + 4 wilderness
    indicators (soil one-hots are *not* used as features — they define
    groups, so including them would leak group identity into X).

    First call downloads/caches covertype via sklearn. Research demo only;
    not a production forestry claim.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100")

    bunch = fetch_covtype()
    X_full = np.asarray(bunch.data, dtype=float)
    y_full = np.asarray(bunch.target, dtype=int)

    soil = X_full[:, 14:54]
    groups_full = soil.argmax(axis=1).astype(int)

    # Features: continuous (0-9) + wilderness (10-13); drop soil one-hots.
    X_feat = X_full[:, :14]
    feature_names = [f"f{i}" for i in range(10)] + [
        "wilderness_0",
        "wilderness_1",
        "wilderness_2",
        "wilderness_3",
    ]

    y_bin = (y_full == positive_class).astype(int)

    rng = np.random.default_rng(random_state)
    n = min(n_samples, X_feat.shape[0])
    idx = rng.choice(X_feat.shape[0], size=n, replace=False)
    X = X_feat[idx]
    y = y_bin[idx]
    groups = groups_full[idx].copy()

    # Merge rare soil types in this subsample into residual group -1 -> remap.
    unique, counts = np.unique(groups, return_counts=True)
    rare = set(unique[counts < min_group_size].tolist())
    if rare:
        residual = int(unique.max()) + 1
        for g in rare:
            groups[groups == g] = residual

    # Remap group ids to 0..G-1 for readability.
    _, groups = np.unique(groups, return_inverse=True)

    return X, y, groups.astype(int), feature_names
