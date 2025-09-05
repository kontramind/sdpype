"""
Intrinsic data quality evaluation - works on any dataset (original or synthetic)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


def evaluate_completeness(data: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate data completeness metrics"""
    
    total_cells = data.shape[0] * data.shape[1]
    missing_cells = data.isnull().sum().sum()
    
    return {
        "total_cells": total_cells,
        "missing_cells": int(missing_cells),
        "completeness_rate": (total_cells - missing_cells) / total_cells,
        "missing_per_column": data.isnull().sum().to_dict(),
        "complete_rows": int((~data.isnull().any(axis=1)).sum()),
        "complete_rows_rate": (~data.isnull().any(axis=1)).mean()
    }


def evaluate_validity(data: pd.DataFrame, schema: Optional[Dict] = None) -> Dict[str, Any]:
    """Evaluate data validity against expected schema and constraints"""
    
    validity_results = {
        "total_rows": len(data),
        "data_types": data.dtypes.astype(str).to_dict(),
        "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    # Basic validity checks
    validity_issues = []
    
    # Check for infinite values in numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(data[col]).sum()
        if inf_count > 0:
            validity_issues.append(f"Column '{col}' has {inf_count} infinite values")
    
    # Check for extremely large values that might indicate errors
    for col in numeric_cols:
        if len(data[col].dropna()) > 0:
            q99 = data[col].quantile(0.99)
            q01 = data[col].quantile(0.01)
            outlier_threshold = 10 * (q99 - q01)
            extreme_values = ((data[col] > q99 + outlier_threshold) | 
                            (data[col] < q01 - outlier_threshold)).sum()
            if extreme_values > 0:
                validity_issues.append(f"Column '{col}' has {extreme_values} extreme outliers")
    
    validity_results["validity_issues"] = validity_issues
    validity_results["validity_score"] = 1.0 - (len(validity_issues) / (len(data.columns) * 2))
    
    return validity_results


def evaluate_uniqueness(data: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate data uniqueness and diversity"""
    
    uniqueness_results = {
        "total_rows": len(data),
        "unique_rows": len(data.drop_duplicates()),
        "duplicate_rows": len(data) - len(data.drop_duplicates()),
        "uniqueness_rate": len(data.drop_duplicates()) / len(data) if len(data) > 0 else 0
    }
    
    # Per-column uniqueness
    column_uniqueness = {}
    for col in data.columns:
        col_unique = data[col].nunique()
        col_total = len(data[col].dropna())
        column_uniqueness[col] = {
            "unique_values": col_unique,
            "total_non_null": col_total,
            "uniqueness_rate": col_unique / col_total if col_total > 0 else 0
        }
    
    uniqueness_results["column_uniqueness"] = column_uniqueness
    
    return uniqueness_results


def evaluate_consistency(data: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate internal data consistency"""
    
    consistency_results = {
        "total_rows": len(data),
        "consistency_checks": []
    }
    
    # Check for consistent data types within columns
    type_consistency_issues = []
    for col in data.select_dtypes(include=['object']).columns:
        # Check if column contains mixed types
        sample_values = data[col].dropna().head(100)
        if len(sample_values) > 0:
            # Simple check for numeric strings mixed with text
            numeric_like = sample_values.astype(str).str.match(r'^-?\d+\.?\d*$').sum()
            if 0 < numeric_like < len(sample_values):
                type_consistency_issues.append(f"Column '{col}' has mixed numeric/text values")
    
    consistency_results["type_consistency_issues"] = type_consistency_issues
    
    # Check for consistent formatting patterns
    format_consistency = {}
    for col in data.select_dtypes(include=['object']).columns:
        sample_values = data[col].dropna().astype(str).head(100)
        if len(sample_values) > 0:
            # Check length variation
            lengths = sample_values.str.len()
            format_consistency[col] = {
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "avg_length": float(lengths.mean()),
                "length_std": float(lengths.std()) if len(lengths) > 1 else 0
            }
    
    consistency_results["format_consistency"] = format_consistency
    consistency_results["consistency_score"] = 1.0 - (len(type_consistency_issues) / len(data.columns))
    
    return consistency_results


def evaluate_data_quality(data: pd.DataFrame, data_source: str = "unknown") -> Dict[str, Any]:
    """
    Unified data quality evaluation that works on any dataset
    
    Args:
        data: Dataset to evaluate (original or synthetic)
        data_source: Source identifier (e.g., "original", "synthetic_seed_42")
    
    Returns:
        Complete quality assessment results
    """
    
    print(f"Evaluating data quality for: {data_source}")
    print(f"Dataset shape: {data.shape}")
    
    # Run all intrinsic quality evaluations
    results = {
        "metadata": {
            "data_source": data_source,
            "evaluation_timestamp": datetime.now().isoformat(),
            "dataset_shape": data.shape,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
        },
        "completeness": evaluate_completeness(data),
        "validity": evaluate_validity(data),
        "uniqueness": evaluate_uniqueness(data),
        "consistency": evaluate_consistency(data)
    }
    
    # Calculate overall quality score
    scores = [
        results["completeness"]["completeness_rate"],
        results["validity"]["validity_score"],
        results["uniqueness"]["uniqueness_rate"],
        results["consistency"]["consistency_score"]
    ]
    
    results["overall_quality_score"] = np.mean(scores)
    
    print(f"Overall quality score: {results['overall_quality_score']:.3f}")
    
    return results


def compare_quality_metrics(original_results: Dict[str, Any], 
                          synthetic_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare quality metrics between original and synthetic data"""
    
    comparison = {
        "metadata": {
            "comparison_timestamp": datetime.now().isoformat(),
            "original_source": original_results["metadata"]["data_source"],
            "synthetic_source": synthetic_results["metadata"]["data_source"]
        }
    }
    
    # Compare overall scores
    comparison["overall_score_comparison"] = {
        "original_score": original_results["overall_quality_score"],
        "synthetic_score": synthetic_results["overall_quality_score"],
        "score_difference": synthetic_results["overall_quality_score"] - original_results["overall_quality_score"],
        "quality_preservation_rate": synthetic_results["overall_quality_score"] / original_results["overall_quality_score"]
    }
    
    # Compare specific metrics
    comparison["metric_comparisons"] = {
        "completeness": {
            "original": original_results["completeness"]["completeness_rate"],
            "synthetic": synthetic_results["completeness"]["completeness_rate"],
            "difference": synthetic_results["completeness"]["completeness_rate"] - original_results["completeness"]["completeness_rate"]
        },
        "uniqueness": {
            "original": original_results["uniqueness"]["uniqueness_rate"],
            "synthetic": synthetic_results["uniqueness"]["uniqueness_rate"],
            "difference": synthetic_results["uniqueness"]["uniqueness_rate"] - original_results["uniqueness"]["uniqueness_rate"]
        }
    }
    
    return comparison
