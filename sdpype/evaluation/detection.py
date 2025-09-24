"""
Detection-based evaluation metrics for synthetic data quality assessment.
Using synthcity's proven real-vs-synthetic detection approach.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata

# Import synthcity detection evaluators
from synthcity.metrics.eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionXGB, 
    SyntheticDetectionMLP,
    SyntheticDetectionLinear
)
from synthcity.plugins.core.dataloader import GenericDataLoader


def evaluate_detection_metrics(
    original: pd.DataFrame, 
    synthetic: pd.DataFrame, 
    metadata: SingleTableMetadata,
    methods_config: List[Dict[str, Any]],
    common_params: Dict[str, Any],
    experiment_name: str
) -> Dict[str, Any]:
    """
    Evaluate detection-based metrics using synthcity's proven implementations.
    
    Args:
        original: Original dataset
        synthetic: Synthetic dataset  
        metadata: SDV metadata
        methods_config: List of detection methods to run
        experiment_name: Experiment identifier
        
    Returns:
        Complete detection metrics results
    """
    
    print(f"Evaluating detection metrics for experiment: {experiment_name}")
    print(f"Original shape: {original.shape}, Synthetic shape: {synthetic.shape}")
    
    results = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "original_shape": list(original.shape),
            "synthetic_shape": list(synthetic.shape),
            "evaluation_type": "detection_metrics"
        },
        "individual_scores": {},
        "ensemble_score": None
    }
    
    # Convert to synthcity DataLoader objects
    try:
        original_loader = _convert_to_dataloader(original, metadata)
        synthetic_loader = _convert_to_dataloader(synthetic, metadata)
        
        if original_loader is None or synthetic_loader is None:
            results["error"] = "Failed to convert data to synthcity DataLoader format"
            return results
            
    except Exception as e:
        results["error"] = f"Data conversion error: {str(e)}"
        return results
    
    print(f"Data converted to synthcity DataLoader format successfully")
    
    # Run each configured detection method
    individual_scores = []
    
    for method_config in methods_config:
        method_name = method_config.get("name")
        parameters = method_config.get("parameters") or {}  # Handle None case

        # Merge common params with method-specific params (method params override common)
        final_parameters = common_params.copy()
        final_parameters.update(parameters)

        print(f"Running {method_name} detection...")
        
        try:
            evaluator = get_synthcity_evaluator(method_name, final_parameters)
            method_result = evaluator.evaluate(original_loader, synthetic_loader)
            
            # Extract AUC score from synthcity result format
            auc_score = method_result.get("mean", method_result.get("avg", None))
            if auc_score is None:
                # Fallback: try to get first value if it's a single-value dict
                if len(method_result) == 1:
                    auc_score = list(method_result.values())[0]
            
            processed_result = {
                "auc_score": float(auc_score) if auc_score is not None else 0.5,
                "raw_result": method_result,
                "parameters": final_parameters,
                "status": "success"
            }
            
            results["individual_scores"][method_name] = processed_result
            
            # Collect successful scores for ensemble calculation
            if auc_score is not None:
                individual_scores.append(float(auc_score))
                
        except Exception as e:
            results["individual_scores"][method_name] = {
                "status": "error", 
                "error_message": str(e),
                "parameters": final_parameters,
                "auc_score": 0.5
            }
    
    # Calculate ensemble score (arithmetic mean)
    if individual_scores:
        ensemble_score = float(np.mean(individual_scores))
        results["ensemble_score"] = ensemble_score
        print(f"Ensemble AUC score: {ensemble_score:.3f}")
    else:
        results["ensemble_score"] = None
        print("No successful individual scores for ensemble calculation")
    
    return results


def _convert_to_dataloader(df: pd.DataFrame, metadata: SingleTableMetadata) -> GenericDataLoader:
    """
    Convert pandas DataFrame to synthcity DataLoader with proper encoding.
    """

    try:
        loader = GenericDataLoader(df)
        return loader
        
    except Exception as e:
        print(f"Error converting DataFrame to DataLoader: {e}")
        return None


def get_synthcity_evaluator(method_name: str, parameters: Dict[str, Any]):
    """Factory function to create synthcity detection evaluators"""
    
    # Only use parameters that synthcity DetectionEvaluator actually accepts
    synthcity_params = {}

    # Add parameters if they exist (with defaults)
    if "n_folds" in parameters:
        synthcity_params["n_folds"] = parameters["n_folds"]
    if "random_state" in parameters:
        synthcity_params["random_state"] = parameters["random_state"]
    if "reduction" in parameters:
        synthcity_params["reduction"] = parameters["reduction"]
    
    match method_name:
        case "detection_gmm":
            return SyntheticDetectionGMM(**synthcity_params)
        case "detection_xgb":
            return SyntheticDetectionXGB(**synthcity_params)
        case "detection_mlp":
            return SyntheticDetectionMLP(**synthcity_params)
        case "detection_linear":
            return SyntheticDetectionLinear(**synthcity_params)
        case _:
            raise ValueError(f"Unknown detection method: {method_name}")


def ensure_json_serializable(obj):
    """Convert OmegaConf objects to plain Python for JSON serialization"""
    from omegaconf import ListConfig, DictConfig
    
    if isinstance(obj, (ListConfig, DictConfig)):
        # Convert OmegaConf to plain Python
        from omegaconf import OmegaConf
        return OmegaConf.to_object(obj)
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(v) for v in obj]
    else:
        return obj


def generate_detection_report(results: Dict[str, Any]) -> str:
    """Generate human-readable detection evaluation report"""
    
    report = f"""Detection-Based Evaluation Report
=====================================

Experiment: {results['metadata']['experiment_name']}
Evaluation Date: {results['metadata']['evaluation_timestamp']}
Original Data Shape: {results['metadata']['original_shape']}
Synthetic Data Shape: {results['metadata']['synthetic_shape']}

Detection Performance Summary:
------------------------------
"""
    
    # Individual method results
    individual_scores = results.get("individual_scores", {})
    
    for method_name, method_result in individual_scores.items():
        if method_result["status"] == "success":
            auc = method_result["auc_score"]
            
            # Interpret AUC score
            if auc <= 0.55:
                quality = "Excellent (very hard to detect)"
            elif auc <= 0.65:
                quality = "Good (moderately hard to detect)"
            elif auc <= 0.75:
                quality = "Fair (somewhat detectable)"
            else:
                quality = "Poor (easily detectable)"
            
            report += f"""
{method_name.replace('_', ' ').title()}:
  → AUC Score: {auc:.3f}
  → Quality Assessment: {quality}
  → Raw Synthcity Result: {method_result.get('raw_result', {})}
"""
        else:
            report += f"""
{method_name.replace('_', ' ').title()}:
  → Status: ERROR
  → Error: {method_result.get('error_message', 'Unknown error')}
"""
    
    # Ensemble results
    ensemble_score = results.get("ensemble_score")
    if ensemble_score is not None:
        if ensemble_score <= 0.55:
            ensemble_quality = "Excellent synthetic data realism"
        elif ensemble_score <= 0.65:
            ensemble_quality = "Good synthetic data realism"
        elif ensemble_score <= 0.75:
            ensemble_quality = "Fair synthetic data realism"
        else:
            ensemble_quality = "Poor synthetic data realism"
        
        report += f"""
Ensemble Results:
  → Mean AUC Score: {ensemble_score:.3f}
  → Overall Assessment: {ensemble_quality}
"""
    else:
        report += f"""
Ensemble Results:
  → Status: No successful individual scores available
"""
    
    report += """
Interpretation Guide:
--------------------
- AUC Score close to 0.5: Classifier cannot distinguish real from synthetic (IDEAL)
- AUC Score close to 1.0: Classifier easily distinguishes real from synthetic (POOR)
- Lower AUC scores indicate higher synthetic data quality

This evaluation uses synthcity's proven detection implementations.
"""

    return report
