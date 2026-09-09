"""Leakage-safe tabular ML research utilities."""

from tabular_ml.calibration import (
    ReliabilityDiagram,
    expected_calibration_error,
    format_reliability_ascii,
    reliability_bins,
    save_reliability_diagram,
)
from tabular_ml.evaluate import (
    GroupAwareCalibratedClassifier,
    compare_calibration_methods,
    evaluate_grouped_cv,
)
from tabular_ml.pipeline import build_baseline_pipeline, build_boosting_pipeline

__all__ = [
    "GroupAwareCalibratedClassifier",
    "ReliabilityDiagram",
    "build_baseline_pipeline",
    "build_boosting_pipeline",
    "compare_calibration_methods",
    "evaluate_grouped_cv",
    "expected_calibration_error",
    "format_reliability_ascii",
    "reliability_bins",
    "save_reliability_diagram",
]

__version__ = "0.2.0"
