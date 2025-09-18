"""
SDPype Evaluation Module

Statistical similarity evaluation framework for comparing original and synthetic data.
"""

from .statistical import evaluate_statistical_similarity, generate_statistical_report

__all__ = [
    "evaluate_statistical_similarity",
    "generate_statistical_report"
]
