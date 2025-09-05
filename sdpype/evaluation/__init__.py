"""
SDPype Evaluation Module

Unified evaluation framework for both original and synthetic data quality assessment.
"""

from .intrinsic import evaluate_data_quality, compare_quality_metrics

__all__ = [
    "evaluate_data_quality",
    "compare_quality_metrics"
]