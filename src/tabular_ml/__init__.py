"""Leakage-safe tabular ML research utilities."""

from tabular_ml.calibration import expected_calibration_error
from tabular_ml.evaluate import GroupAwareCalibratedClassifier, evaluate_grouped_cv
from tabular_ml.pipeline import build_baseline_pipeline, build_boosting_pipeline

__all__ = [
    "GroupAwareCalibratedClassifier",
    "build_baseline_pipeline",
    "build_boosting_pipeline",
    "evaluate_grouped_cv",
    "expected_calibration_error",
]

__version__ = "0.1.0"
