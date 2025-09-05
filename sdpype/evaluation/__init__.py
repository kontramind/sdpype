"""
SDPype Evaluation Module

Unified evaluation framework for both original and synthetic data quality assessment.
Includes intrinsic quality evaluation and statistical similarity evaluation.
"""

from .intrinsic import evaluate_data_quality, compare_quality_metrics
from .statistical import evaluate_statistical_similarity, generate_statistical_report  # ✨ NEW

__all__ = [
    "evaluate_data_quality",
    "compare_quality_metrics",
    "evaluate_statistical_similarity",  # ✨ NEW
    "generate_statistical_report"       # ✨ NEW
]
