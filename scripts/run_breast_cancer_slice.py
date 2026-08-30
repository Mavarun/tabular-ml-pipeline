#!/usr/bin/env python3
"""Run the leakage-safe tabular slice on sklearn breast_cancer.

Hypothesis (research demo, not production claims):
1. Fitting prep on the full matrix before CV leaks into fold metrics.
2. Entity-grouped rows need GroupKFold or OOS scores are optimistic.
3. Sigmoid calibration inside train folds should cut Brier/ECE vs raw
   logistic probs without claiming clinical accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabular_ml.data import load_breast_cancer_grouped
from tabular_ml.evaluate import evaluate_grouped_cv
from tabular_ml.pipeline import build_baseline_pipeline, build_boosting_pipeline


def main() -> None:
    X, y, groups, _ = load_breast_cancer_grouped(n_groups=40, random_state=0)
    n_splits = 5

    logistic = build_baseline_pipeline()
    boosting = build_boosting_pipeline()

    results = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_groups": int(len(set(groups))),
        "n_splits": n_splits,
        "logistic_raw": evaluate_grouped_cv(
            logistic, X, y, groups, n_splits=n_splits, calibrate=False
        ).as_dict(),
        "logistic_calibrated": evaluate_grouped_cv(
            logistic, X, y, groups, n_splits=n_splits, calibrate=True
        ).as_dict(),
        "hgb_raw": evaluate_grouped_cv(
            boosting, X, y, groups, n_splits=n_splits, calibrate=False
        ).as_dict(),
    }

    print(json.dumps(results, indent=2))
    print(
        "\nNote: metrics are OOS under GroupKFold on a public toy set. "
        "Not live accuracy, not clinical performance."
    )


if __name__ == "__main__":
    main()
