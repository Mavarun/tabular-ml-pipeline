#!/usr/bin/env python3
"""Run reliability / calibration slice on grouped covertype subsample.

Hypothesis (research demo, not production claims):
1. Reliability diagrams + ECE make miscalibration visible beyond accuracy.
2. On harder grouped covertype (not easy breast_cancer), raw logistic/HGB
   show higher ECE than on breast_cancer.
3. Isotonic/Platt calibration on train folds may cut ECE — measure whether
   it helps or hurts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabular_ml.calibration import format_reliability_ascii, save_reliability_diagram
from tabular_ml.data import load_breast_cancer_grouped, load_covertype_grouped
from tabular_ml.evaluate import compare_calibration_methods, evaluate_grouped_cv
from tabular_ml.pipeline import build_baseline_pipeline, build_boosting_pipeline


def _summary(result) -> dict:
    return {
        "mean_accuracy": round(result.mean_accuracy, 4),
        "mean_roc_auc": round(result.mean_roc_auc, 4),
        "mean_brier": round(result.mean_brier, 4),
        "mean_ece": round(result.mean_ece, 4),
        "pooled_ece": round(result.reliability.ece, 4)
        if result.reliability is not None
        else None,
    }


def main() -> None:
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Reference: breast_cancer raw logistic ECE (slice 1 scale).
    X_bc, y_bc, g_bc, _ = load_breast_cancer_grouped(n_groups=40, random_state=0)
    bc_raw = evaluate_grouped_cv(
        build_baseline_pipeline(),
        X_bc,
        y_bc,
        g_bc,
        n_splits=5,
        calibrate=False,
        return_oof=True,
    )

    X, y, groups, _ = load_covertype_grouped(
        n_samples=5000, random_state=0, min_group_size=40
    )
    n_splits = 5
    logistic = build_baseline_pipeline()
    boosting = build_boosting_pipeline(max_iter=80)

    log_results = compare_calibration_methods(
        logistic,
        X,
        y,
        groups,
        n_splits=n_splits,
        methods=("raw", "sigmoid", "isotonic"),
        n_bins=10,
    )
    hgb_raw = evaluate_grouped_cv(
        boosting,
        X,
        y,
        groups,
        n_splits=n_splits,
        calibrate=False,
        return_oof=True,
    )
    hgb_sigmoid = evaluate_grouped_cv(
        boosting,
        X,
        y,
        groups,
        n_splits=n_splits,
        calibrate=True,
        calibration_method="sigmoid",
        return_oof=True,
    )

    # Save reliability diagrams for logistic raw vs isotonic.
    paths = {}
    for name, res in [
        ("logistic_raw", log_results["raw"]),
        ("logistic_sigmoid", log_results["sigmoid"]),
        ("logistic_isotonic", log_results["isotonic"]),
        ("hgb_raw", hgb_raw),
    ]:
        if res.reliability is not None:
            p = save_reliability_diagram(
                res.reliability,
                fig_dir / f"reliability_{name}.png",
                title=f"covertype GroupKFold — {name}",
            )
            paths[name] = str(p.relative_to(ROOT))

    payload = {
        "dataset": "sklearn.datasets.fetch_covtype (subsample)",
        "group_column": "soil_type (argmax of soil one-hots; not used as features)",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_groups": int(np_unique_groups(groups)),
        "n_splits": n_splits,
        "breast_cancer_logistic_raw_reference": _summary(bc_raw),
        "logistic_raw": _summary(log_results["raw"]),
        "logistic_sigmoid": _summary(log_results["sigmoid"]),
        "logistic_isotonic": _summary(log_results["isotonic"]),
        "hgb_raw": _summary(hgb_raw),
        "hgb_sigmoid": _summary(hgb_sigmoid),
        "figures": paths,
        "note": (
            "OOS under GroupKFold on a public covertype subsample. "
            "Not live accuracy, not a production/forestry claim."
        ),
    }

    print(json.dumps(payload, indent=2))
    print()
    print(format_reliability_ascii(log_results["raw"].reliability))
    print()
    print(format_reliability_ascii(log_results["isotonic"].reliability))


def np_unique_groups(groups) -> int:
    import numpy as np

    return int(np.unique(groups).size)


if __name__ == "__main__":
    main()
