# sdpype/evaluation/statistical.py
"""
Statistical metrics evaluation using Alpha Precision and PRDC Score
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime

# Synthcity imports for statistical metrics
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.metrics.eval_statistical import PRDCScore

import warnings

warnings.filterwarnings('ignore')

from rich import print
from rich.pretty import pprint
# from rich.traceback import install

def ensure_json_serializable(obj: Any) -> Any:
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj

class AlphaPrecisionMetric:
    """Alpha Precision metric implementation"""

    def __init__(self, **parameters):
        self.evaluator = AlphaPrecision()
        self.parameters = parameters  # Store for reporting

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate Alpha Precision metric"""

        start_time = time.time()

        try:
            # Create data loaders
            real_loader = GenericDataLoader(original)
            synth_loader = GenericDataLoader(synthetic)

            # Run evaluation
            result = self.evaluator.evaluate(real_loader, synth_loader)
            scores = {
                "delta_precision_alpha_OC": float(result.get("delta_precision_alpha_OC", 0.0)),
                "delta_coverage_beta_OC": float(result.get("delta_coverage_beta_OC", 0.0)),
                "authenticity_OC": float(result.get("authenticity_OC", 0.0)),
                "delta_precision_alpha_naive": float(result.get("delta_precision_alpha_naive", 0.0)),
                "delta_coverage_beta_naive": float(result.get("delta_coverage_beta_naive", 0.0)),
                "authenticity_naive": float(result.get("authenticity_naive", 0.0))
            }

            return {
                "scores": scores,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "scores": {
                    "delta_precision_alpha_OC": 0.0,
                    "delta_coverage_beta_OC": 0.0,
                    "authenticity_OC": 0.0,
                    "delta_precision_alpha_naive": 0.0,
                    "delta_coverage_beta_naive": 0.0,
                    "authenticity_naive": 0.0
                },
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

class PRDCScoreMetric:
    """PRDC Score metric implementation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.evaluator = PRDCScore(nearest_k=parameters.get("nearest_k", 5))

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate PRDC Score metric"""
        start_time = time.time()

        try:
            # Create data loaders
            real_loader = GenericDataLoader(original)
            synth_loader = GenericDataLoader(synthetic)

            # Run evaluation
            result = self.evaluator.evaluate(real_loader, synth_loader)

            # PRDC returns a dict with precision, recall, density, coverage
            return {
                "precision": float(result.get("precision", 0.0)),
                "recall": float(result.get("recall", 0.0)),
                "density": float(result.get("density", 0.0)),
                "coverage": float(result.get("coverage", 0.0)),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "density": 0.0,
                "coverage": 0.0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


def evaluate_statistical_metrics(original: pd.DataFrame,
                                synthetic: pd.DataFrame,
                                metrics_config: list,
                                experiment_name: str = "unknown") -> Dict[str, Any]:
    """
    Evaluate configured statistical metrics

    Args:
        original: Original dataset
        synthetic: Synthetic dataset
        metrics_config: List of metric configurations
        experiment_name: Experiment identifier

    Returns:
        Complete statistical metrics results
    """

    print(f"Evaluating statistical metrics for experiment: {experiment_name}")
    print(f"Original shape: {original.shape}, Synthetic shape: {synthetic.shape}")

    results = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "original_shape": list(original.shape),
            "synthetic_shape": list(synthetic.shape),
            "evaluation_type": "statistical_metrics"
        },
        "metrics": {}
    }

    # Run each configured metric
    metric_scores = []

    for metric_config in metrics_config:
        metric_name = metric_config.get("name")
        parameters = metric_config.get("parameters", {})

        print(f"Running {metric_name} metric...")

        try:
            evaluator = get_metric_evaluator(metric_name, parameters)
            metric_result = evaluator.evaluate(original, synthetic)
            results["metrics"][metric_name] = metric_result

            # Collect scores for overall calculation
            if metric_result["status"] == "success":
                if metric_name == "alpha_precision":
                    # Individual scores handled in report - no aggregation
                    pass
                elif metric_name == "prdc_score":
                    # Individual scores handled in report - no aggregation
                    pass

        except Exception as e:
            results["metrics"][metric_name] = {
                "status": "error",
                "error_message": str(e),
                "parameters": parameters
            }

    print("Individual statistical metrics completed - see detailed scores")

    # Ensure all results are JSON serializable
    results = ensure_json_serializable(results)

    return results


def get_metric_evaluator(metric_name: str, parameters: Dict[str, Any]):
    """Factory function to create metric evaluators"""

    if metric_name == "alpha_precision":
        return AlphaPrecisionMetric(**parameters)
    elif metric_name == "prdc_score":
        return PRDCScoreMetric(**parameters)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def generate_statistical_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable statistical metrics report"""
    
    report = f"""
Statistical Metrics Evaluation Report
=======================================

Experiment: {results['metadata']['experiment_name']}
Timestamp: {results['metadata']['evaluation_timestamp']}
Dataset Shapes: Original {tuple(results['metadata']['original_shape'])}, Synthetic {tuple(results['metadata']['synthetic_shape'])}

Note: Individual metric scores provided

Metrics Results
---------------
"""
    metrics = results.get("metrics", {})

    # Alpha Precision results
    if "alpha_precision" in metrics:
        alpha_result = metrics["alpha_precision"]
        if alpha_result["status"] == "success":
            scores = alpha_result['scores']
            report += f"""Alpha Precision Results:
  Parameters: {alpha_result['parameters'] if alpha_result['parameters'] else 'none'}
  Execution time: {alpha_result['execution_time']:.2f}s

  Individual Scores:
    Optimally-Corrected (OC) Variant:
      → Delta Precision Alpha: {scores['delta_precision_alpha_OC']:.3f}
      → Delta Coverage Beta:   {scores['delta_coverage_beta_OC']:.3f}
      → Authenticity:          {scores['authenticity_OC']:.3f}

    Naive Variant:
      → Delta Precision Alpha: {scores['delta_precision_alpha_naive']:.3f}
      → Delta Coverage Beta:   {scores['delta_coverage_beta_naive']:.3f}
      → Authenticity:          {scores['authenticity_naive']:.3f}
"""
        else:
            report += f"""Alpha Precision: ERROR
  Error: {alpha_result.get('error_message', 'Unknown error')}
"""
    
    # PRDC Score results
    if "prdc_score" in metrics:
        prdc_result = metrics["prdc_score"]
        if prdc_result["status"] == "success":
            report += f"""PRDC Score Results:
  Parameters: {prdc_result['parameters'] if prdc_result['parameters'] else 'default settings'}
  Execution time: {prdc_result['execution_time']:.2f}s

  Individual Scores:
  Precision: {prdc_result['precision']:.3f}
  Recall: {prdc_result['recall']:.3f}
  Density: {prdc_result['density']:.3f}
  Coverage: {prdc_result['coverage']:.3f}
"""
    
        else:
            report += f"""PRDC Score: ERROR
  Error: {prdc_result.get('error_message', 'Unknown error')}
"""

    # Individual metric insights
    insights = []

    if "alpha_precision" in metrics and metrics["alpha_precision"]["status"] == "success":
        auth_oc = metrics["alpha_precision"]["scores"]["authenticity_OC"]
        if auth_oc >= 0.8:
            insights.append("Strong authenticity (OC)")
        elif auth_oc >= 0.6:
            insights.append("Moderate authenticity (OC)")
        else:
            insights.append("Low authenticity (OC)")

    if "prdc_score" in metrics and metrics["prdc_score"]["status"] == "success":
        prdc = metrics["prdc_score"]
        precision = prdc["precision"]
        recall = prdc["recall"]
        coverage = prdc["coverage"]

        # Overall PRDC assessment based on average
        prdc_avg = (precision + recall + coverage) / 3  # Density often lower, so exclude
        if prdc_avg >= 0.8:
            insights.append("Strong PRDC performance")
        elif prdc_avg >= 0.6:
            insights.append("Moderate PRDC performance")
        else:
            insights.append("Low PRDC performance")

    assessment = ", ".join(insights) if insights else "No successful metrics"

    report += f"""
Assessment: {assessment}
"""

    return report
