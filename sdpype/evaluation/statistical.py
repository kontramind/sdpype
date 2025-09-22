# sdpype/evaluation/statistical.py
"""
Statistical metrics evaluation
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from sdv.metadata import SingleTableMetadata

# Synthcity imports for statistical metrics
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.metrics.eval_statistical import PRDCScore

# SDMetrics imports
from sdmetrics.single_table import TableStructure
from sdmetrics.single_column import BoundaryAdherence
from sdmetrics.single_column import CategoryAdherence
from sdmetrics.single_column import KSComplement
from sdmetrics.single_column import TVComplement
from sdmetrics.single_table.new_row_synthesis import NewRowSynthesis

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

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
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

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
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


class NewRowSynthesisMetric:
    """NewRowSynthesis metric implementation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.numerical_match_tolerance = parameters.get("numerical_match_tolerance", 0.01)
        self.synthetic_sample_size = parameters.get("synthetic_sample_size", None)

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
        """Evaluate NewRowSynthesis metric"""
        start_time = time.time()

        try:
            # Run evaluation using SDMetrics
            result = NewRowSynthesis.compute_breakdown(
                real_data=original,
                synthetic_data=synthetic,
                metadata=metadata,
                numerical_match_tolerance=self.numerical_match_tolerance,
                synthetic_sample_size=self.synthetic_sample_size
            )

            return {
                "score": float(result.get("score", 0.0)),
                "num_new_rows": int(result.get("num_new_rows", 0)),
                "num_matched_rows": int(result.get("num_matched_rows", 0)),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "num_new_rows": 0,
                "num_matched_rows": len(synthetic) if synthetic is not None else 0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class KSComplementMetric:
    """KSComplement metric implementation for column-wise distribution similarity"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all numerical/datetime

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
        """Evaluate KSComplement metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (numerical and datetime)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to user-specified columns
                compatible_columns = [col for col in compatible_columns if col in self.target_columns]

            if not compatible_columns:
                # No compatible columns is not an error - just return empty results
                return {
                    "aggregate_score": None,  # Use None to indicate N/A
                    "column_scores": {},
                    "compatible_columns": [],
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "No compatible numerical/datetime columns found"
                }

            column_scores = {}
            for column in compatible_columns:
                try:
                    score = KSComplement.compute(
                        real_data=original[column],
                        synthetic_data=synthetic[column]
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    # Handle individual column failures
                    column_scores[column] = 0.0
                    print(f"Warning: KSComplement failed for column '{column}': {e}")

            # Calculate aggregate score
            aggregate_score = sum(column_scores.values()) / len(column_scores) if column_scores else 0.0

            return {
                "aggregate_score": float(aggregate_score),
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with KSComplement (numerical and datetime) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['numerical', 'datetime']:
                compatible_columns.append(column_name)

        return compatible_columns


class TVComplementMetric:
    """TVComplement metric implementation for column-wise categorical distribution similarity"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all categorical/boolean

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
        """Evaluate TVComplement metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (categorical and boolean)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to user-specified columns
                compatible_columns = [col for col in compatible_columns if col in self.target_columns]

            if not compatible_columns:
                # No compatible columns is not an error - just return empty results
                return {
                    "aggregate_score": None,  # Use None to indicate N/A
                    "column_scores": {},
                    "compatible_columns": [],
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "No compatible categorical/boolean columns found"
                }

            column_scores = {}
            for column in compatible_columns:
                try:
                    score = TVComplement.compute(
                        real_data=original[column],
                        synthetic_data=synthetic[column]
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    # Handle individual column failures
                    column_scores[column] = 0.0
                    print(f"Warning: TVComplement failed for column '{column}': {e}")

            # Calculate aggregate score
            aggregate_score = sum(column_scores.values()) / len(column_scores) if column_scores else 0.0

            return {
                "aggregate_score": float(aggregate_score),
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with TVComplement (categorical and boolean) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['categorical', 'boolean']:
                compatible_columns.append(column_name)

        return compatible_columns


class TableStructureMetric:
    """TableStructure metric implementation for table structure validation"""

    def __init__(self, **parameters):
        self.parameters = parameters

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
        """Evaluate TableStructure metric"""
        start_time = time.time()

        try:
            # Run evaluation using SDMetrics
            score = TableStructure.compute(
                real_data=original,
                synthetic_data=synthetic
            )

            return {
                "score": float(score),
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "score": 0.0,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }


class BoundaryAdherenceMetric:
    """BoundaryAdherence metric implementation for column-wise boundary validation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all numerical/datetime

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
        """Evaluate BoundaryAdherence metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (numerical and datetime)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to only requested columns that are also compatible
                target_set = set(self.target_columns)
                compatible_set = set(compatible_columns)
                valid_targets = list(target_set.intersection(compatible_set))
                invalid_targets = list(target_set - compatible_set)

                if invalid_targets:
                    print(f"Warning: These target columns are not compatible with BoundaryAdherence: {invalid_targets}")

                if not valid_targets:
                    return {
                        "aggregate_score": None,
                        "column_scores": {},
                        "compatible_columns": compatible_columns,
                        "parameters": self.parameters,
                        "execution_time": time.time() - start_time,
                        "status": "success",
                        "message": f"No compatible columns found from target list: {self.target_columns}"
                    }

                columns_to_evaluate = valid_targets
            else:
                columns_to_evaluate = compatible_columns

            if not columns_to_evaluate:
                return {
                    "aggregate_score": None,
                    "column_scores": {},
                    "compatible_columns": compatible_columns,
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "No compatible numerical/datetime columns found in the dataset"
                }

            # Calculate BoundaryAdherence for each compatible column
            column_scores = {}
            for column in columns_to_evaluate:
                try:
                    score = BoundaryAdherence.compute(
                        real_data=original[column],
                        synthetic_data=synthetic[column]
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    print(f"Error computing BoundaryAdherence for column {column}: {e}")
                    # Skip this column but continue with others

            if not column_scores:
                return {
                    "aggregate_score": None,
                    "column_scores": {},
                    "compatible_columns": compatible_columns,
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": "All column evaluations failed"
                }

            # Calculate aggregate score as mean of individual column scores
            aggregate_score = float(np.mean(list(column_scores.values())))

            return {
                "aggregate_score": aggregate_score,
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with BoundaryAdherence (numerical and datetime) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['numerical', 'datetime']:
                compatible_columns.append(column_name)

        return compatible_columns


class CategoryAdherenceMetric:
    """CategoryAdherence metric implementation for categorical/boolean column validation"""

    def __init__(self, **parameters):
        self.parameters = parameters
        self.target_columns = parameters.get("target_columns", None)  # None = all categorical/boolean

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> Dict[str, Any]:
        """Evaluate CategoryAdherence metric across compatible columns"""
        start_time = time.time()

        try:
            # Identify compatible columns (categorical and boolean)
            compatible_columns = self._get_compatible_columns(metadata)

            if self.target_columns:
                # Filter to only requested columns that are also compatible
                target_set = set(self.target_columns)
                compatible_set = set(compatible_columns)
                valid_targets = list(target_set.intersection(compatible_set))

                # Warn about invalid target columns
                invalid_targets = target_set - compatible_set
                if invalid_targets:
                    print(f"Warning: Columns {invalid_targets} are not compatible with CategoryAdherence (not categorical/boolean)")

                columns_to_evaluate = valid_targets
            else:
                columns_to_evaluate = compatible_columns

            if not columns_to_evaluate:
                return {
                    "aggregate_score": None,
                    "column_scores": {},
                    "compatible_columns": compatible_columns,
                    "parameters": self.parameters,
                    "execution_time": time.time() - start_time,
                    "status": "success",
                    "message": f"No compatible categorical/boolean columns found for evaluation"
                }

            # Evaluate each column
            column_scores = {}
            for column in columns_to_evaluate:
                try:
                    score = CategoryAdherence.compute(
                        real_data=original[column],
                        synthetic_data=synthetic[column]
                    )
                    column_scores[column] = float(score)
                except Exception as e:
                    print(f"Warning: Failed to compute CategoryAdherence for column '{column}': {str(e)}")
                    column_scores[column] = 0.0

            # Calculate aggregate score (average of all column scores)
            if column_scores:
                aggregate_score = sum(column_scores.values()) / len(column_scores)
            else:
                aggregate_score = 0.0

            return {
                "aggregate_score": aggregate_score,
                "column_scores": column_scores,
                "compatible_columns": compatible_columns,
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "aggregate_score": 0.0,
                "column_scores": {},
                "compatible_columns": [],
                "parameters": self.parameters,
                "execution_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }

    def _get_compatible_columns(self, metadata: SingleTableMetadata) -> List[str]:
        """Get columns compatible with CategoryAdherence (categorical and boolean) from SDV metadata"""

        compatible_columns = []
        for column_name, column_info in metadata.columns.items():
            sdtype = column_info.get('sdtype', 'unknown')
            if sdtype in ['categorical', 'boolean']:
                compatible_columns.append(column_name)

        return compatible_columns


def evaluate_statistical_metrics(original: pd.DataFrame,
                                synthetic: pd.DataFrame,
                                metrics_config: list,
                                experiment_name: str,
                                metadata: SingleTableMetadata) -> Dict[str, Any]:
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
            metric_result = evaluator.evaluate(original, synthetic, metadata)
            results["metrics"][metric_name] = metric_result

            # Collect scores for overall calculation
            if metric_result["status"] == "success":
                match metric_name:
                    case "alpha_precision" | "prdc_score" | "new_row_synthesis" | "ks_complement" | "tv_complement":
                        # Individual scores handled in report - no aggregation
                        pass
                    case _:
                        # Future metrics that might need aggregation
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

    match metric_name:
        case "table_structure":
            return TableStructureMetric(**parameters)
        case "boundary_adherence":
            return BoundaryAdherenceMetric(**parameters)
        case "category_adherence":
            return CategoryAdherenceMetric(**parameters)
        case "new_row_synthesis":
            return NewRowSynthesisMetric(**parameters)
        case "ks_complement":
            return KSComplementMetric(**parameters)
        case "tv_complement":
            return TVComplementMetric(**parameters)
        case "alpha_precision":
            return AlphaPrecisionMetric(**parameters)
        case "prdc_score":
            return PRDCScoreMetric(**parameters)
        case _:
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

    # NewRowSynthesis results
    if "new_row_synthesis" in metrics:
        nrs_result = metrics["new_row_synthesis"]
        if nrs_result["status"] == "success":
            params_info = nrs_result["parameters"]
            tolerance = params_info.get("numerical_match_tolerance", 0.01)
            sample_size = params_info.get("synthetic_sample_size", "all rows")

            report += f"""NewRowSynthesis Results:
  Parameters: tolerance={tolerance}, sample_size={sample_size}
  Execution time: {nrs_result['execution_time']:.2f}s

  Synthesis Quality:
  → New Row Score:     {nrs_result['score']:.3f}
  → New Rows:          {nrs_result['num_new_rows']:,}
  → Matched Rows:      {nrs_result['num_matched_rows']:,}
"""

    # KSComplement results
    if "ks_complement" in metrics:
        ks_result = metrics["ks_complement"]
        if ks_result["status"] == "success":
            params_info = ks_result["parameters"]
            target_cols = params_info.get("target_columns", "all numerical/datetime")

            if ks_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""KSComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {ks_result['execution_time']:.2f}s
  Status: {ks_result['message']}
"""
            else:
                report += f"""KSComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {ks_result['execution_time']:.2f}s

  Distribution Similarity:
  → Aggregate Score:   {ks_result['aggregate_score']:.3f}
  → Columns Evaluated: {len(ks_result['compatible_columns'])}

  Individual Column Scores:"""
                for col, score in ks_result['column_scores'].items():
                    report += f"""
    → {col}: {score:.3f}"""
                report += "\n"
        else:
            report += f"""KSComplement: ERROR
  Error: {ks_result.get('error_message', 'Unknown error')}
"""

    # TVComplement results
    if "tv_complement" in metrics:
        tv_result = metrics["tv_complement"]
        if tv_result["status"] == "success":
            params_info = tv_result["parameters"]
            target_cols = params_info.get("target_columns", "all categorical/boolean")

            if tv_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""TVComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {tv_result['execution_time']:.2f}s
  Status: {tv_result['message']}
"""
            else:
                report += f"""TVComplement Results:
  Parameters: target_columns={target_cols}
  Execution time: {tv_result['execution_time']:.2f}s

  Categorical Distribution Similarity:
  → Aggregate Score:   {tv_result['aggregate_score']:.3f}
  → Columns Evaluated: {len(tv_result['compatible_columns'])}

  Individual Column Scores:"""
                for col, score in tv_result['column_scores'].items():
                    report += f"""
    → {col}: {score:.3f}"""
                report += "\n"
        else:
            report += f"""TVComplement: ERROR
  Error: {tv_result.get('error_message', 'Unknown error')}
"""
    # TableStructure results
    if "table_structure" in metrics:
        ts_result = metrics["table_structure"]
        if ts_result["status"] == "success":
            report += f"""TableStructure Results:
  Parameters: {ts_result['parameters'] if ts_result['parameters'] else 'none'}
  Execution time: {ts_result['execution_time']:.2f}s

  Structure Similarity:
  → Table Structure Score: {ts_result['score']:.3f}
"""
        else:
            report += f"""TableStructure: ERROR
  Error: {ts_result.get('error_message', 'Unknown error')}
"""

    # BoundaryAdherence results
    if "boundary_adherence" in metrics:
        ba_result = metrics["boundary_adherence"]
        if ba_result["status"] == "success":
            params_info = ba_result["parameters"]
            target_cols = params_info.get("target_columns", "all numerical/datetime")

            if ba_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""BoundaryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ba_result['execution_time']:.2f}s
  Status: {ba_result['message']}
"""
            else:
                report += f"""BoundaryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ba_result['execution_time']:.2f}s

  Boundary Compliance:
  → Aggregate Score:   {ba_result['aggregate_score']:.3f}
  → Columns Evaluated: {len(ba_result['compatible_columns'])}

  Individual Column Scores:"""
                for col, score in ba_result['column_scores'].items():
                    report += f"""
    → {col}: {score:.3f}"""
                report += "\n"
        else:
            report += f"""BoundaryAdherence: ERROR
  Error: {ba_result.get('error_message', 'Unknown error')}
"""

    # CategoryAdherence results
    if "category_adherence" in metrics:
        ca_result = metrics["category_adherence"]
        if ca_result["status"] == "success":
            params_info = ca_result["parameters"]
            target_cols = params_info.get("target_columns", "all categorical/boolean")

            if ca_result.get("message"):
                # Handle case where no compatible columns found
                report += f"""CategoryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ca_result['execution_time']:.2f}s
  Status: {ca_result['message']}
"""
            else:
                report += f"""CategoryAdherence Results:
  Parameters: target_columns={target_cols}
  Execution time: {ca_result['execution_time']:.2f}s

  Category Compliance:
  → Aggregate Score:   {ca_result['aggregate_score']:.3f}
  → Columns Evaluated: {len(ca_result['compatible_columns'])}

  Individual Column Scores:"""
                for col, score in ca_result['column_scores'].items():
                    report += f"""
    → {col}: {score:.3f}"""
                report += "\n"
        else:
            report += f"""CategoryAdherence: ERROR
  Error: {ca_result.get('error_message', 'Unknown error')}
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

    if "new_row_synthesis" in metrics and metrics["new_row_synthesis"]["status"] == "success":
        nrs_score = metrics["new_row_synthesis"]["score"]
        if nrs_score >= 0.9:
            insights.append("Excellent synthesis novelty")
        elif nrs_score >= 0.7:
            insights.append("Good synthesis novelty")
        else:
            insights.append("Low synthesis novelty")

    if "ks_complement" in metrics and metrics["ks_complement"]["status"] == "success":
        ks_score = metrics["ks_complement"]["aggregate_score"]
        if ks_score is not None:  # Only evaluate if we have compatible columns
            if ks_score >= 0.9:
                insights.append("Excellent distribution similarity")
            elif ks_score >= 0.7:
                insights.append("Good distribution similarity")
            else:
                insights.append("Poor distribution similarity")
        # If ks_score is None, we simply don't add any insight (no numerical columns to evaluate)

    if "tv_complement" in metrics and metrics["tv_complement"]["status"] == "success":
        tv_score = metrics["tv_complement"]["aggregate_score"]
        if tv_score is not None:  # Only evaluate if we have compatible columns
            if tv_score >= 0.9:
                insights.append("Excellent categorical similarity")
            elif tv_score >= 0.7:
                insights.append("Good categorical similarity")
            else:
                insights.append("Poor categorical similarity")
        # If tv_score is None, we simply don't add any insight (no categorical columns to evaluate)

    if "table_structure" in metrics and metrics["table_structure"]["status"] == "success":
        ts_score = metrics["table_structure"]["score"]
        if ts_score >= 0.95:
            insights.append("Perfect table structure match")
        elif ts_score >= 0.8:
            insights.append("Good table structure match")
        else:
            insights.append("Poor table structure match")

    if "boundary_adherence" in metrics and metrics["boundary_adherence"]["status"] == "success":
        ba_score = metrics["boundary_adherence"]["aggregate_score"]
        if ba_score is not None:  # Only evaluate if we have compatible columns
            if ba_score >= 0.95:
                insights.append("Excellent boundary adherence")
            elif ba_score >= 0.8:
                insights.append("Good boundary adherence")
            else:
                insights.append("Poor boundary adherence")
        # If ba_score is None, we simply don't add any insight (no numerical/datetime columns to evaluate)

    if "category_adherence" in metrics and metrics["category_adherence"]["status"] == "success":
        ca_score = metrics["category_adherence"]["aggregate_score"]
        if ca_score is not None:  # Only evaluate if we have compatible columns
            if ca_score >= 0.95:
                insights.append("Excellent category adherence")
            elif ca_score >= 0.8:
                insights.append("Good category adherence")
            else:
                insights.append("Poor category adherence")
        # If ca_score is None, we simply don't add any insight (no categorical/boolean columns to evaluate)

    assessment = ", ".join(insights) if insights else "No successful metrics"

    report += f"""
Assessment: {assessment}
"""

    return report
