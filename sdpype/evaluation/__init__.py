"""
SDPype Evaluation Module

Statistical metrics evaluation framework using Alpha Precision Score.
"""

from .statistical import evaluate_statistical_metrics, generate_statistical_report

__all__ = [
    "evaluate_statistical_metrics",
    "generate_statistical_report"
]
