# sdpype/evaluation/statistical.py
"""
Statistical similarity evaluation for comparing original and synthetic data
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def ensure_json_serializable(obj):
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


def ks_test_all_columns(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Kolmogorov-Smirnov tests for all numeric columns"""
    
    ks_results = {
        "test_type": "kolmogorov_smirnov",
        "column_results": {},
        "overall_statistics": {}
    }
    
    numeric_cols = original.select_dtypes(include=[np.number]).columns
    p_values = []
    statistics = []
    
    for col in numeric_cols:
        if col in synthetic.columns:
            # Get non-null values
            orig_values = original[col].dropna()
            synth_values = synthetic[col].dropna()
            
            if len(orig_values) > 0 and len(synth_values) > 0:
                ks_stat, p_value = stats.ks_2samp(orig_values, synth_values)
                
                ks_results["column_results"][col] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "distributions_similar": bool(float(p_value) > 0.05),  # α = 0.05
                    "original_samples": int(len(orig_values)),
                    "synthetic_samples": int(len(synth_values))
                }
                
                p_values.append(p_value)
                statistics.append(ks_stat)
    
    # Overall statistics
    if p_values:
        similar_count = sum(1 for p in p_values if float(p) > 0.05)
        ks_results["overall_statistics"] = {
            "mean_p_value": float(np.mean(p_values)),
            "mean_ks_statistic": float(np.mean(statistics)),
            "columns_similar": int(similar_count),
            "total_columns_tested": int(len(p_values)),
            "similarity_rate": float(similar_count / len(p_values))
        }
    
    return ks_results


def correlation_analysis(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Compare correlation matrices between original and synthetic data"""
    
    correlation_results = {
        "test_type": "correlation_analysis",
        "numeric_correlations": {},
        "correlation_preservation": {}
    }
    
    # Get numeric columns that exist in both datasets
    numeric_cols = original.select_dtypes(include=[np.number]).columns
    common_numeric = [col for col in numeric_cols if col in synthetic.columns]
    
    if len(common_numeric) > 1:
        # Calculate correlation matrices
        orig_corr = original[common_numeric].corr()
        synth_corr = synthetic[common_numeric].corr()
        
        # Flatten upper triangular matrices (excluding diagonal)
        mask = np.triu(np.ones_like(orig_corr, dtype=bool), k=1)
        orig_corr_flat = orig_corr.values[mask]
        synth_corr_flat = synth_corr.values[mask]
        
        # Calculate correlation between correlation matrices
        if len(orig_corr_flat) > 0:
            corr_preservation, _ = stats.pearsonr(orig_corr_flat, synth_corr_flat)
            
            correlation_results["correlation_preservation"] = {
                "correlation_of_correlations": float(corr_preservation),
                "mean_absolute_difference": float(np.mean(np.abs(orig_corr_flat - synth_corr_flat))),
                "max_absolute_difference": float(np.max(np.abs(orig_corr_flat - synth_corr_flat))),
                "correlation_pairs_analyzed": int(len(orig_corr_flat))
            }
            
            # Detailed correlation comparison
            correlation_results["detailed_comparison"] = {}
            for i, col1 in enumerate(common_numeric):
                for j, col2 in enumerate(common_numeric):
                    if i < j:  # Upper triangular only
                        pair_name = f"{col1}_vs_{col2}"
                        correlation_results["detailed_comparison"][pair_name] = {
                            "original_correlation": float(orig_corr.iloc[i, j]),
                            "synthetic_correlation": float(synth_corr.iloc[i, j]),
                            "difference": float(abs(orig_corr.iloc[i, j] - synth_corr.iloc[i, j]))
                        }
    
    return correlation_results


def distribution_comparison(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Compare statistical moments and distributions"""
    
    distribution_results = {
        "test_type": "distribution_comparison",
        "moment_analysis": {},
        "distribution_metrics": {}
    }
    
    numeric_cols = original.select_dtypes(include=[np.number]).columns
    common_numeric = [col for col in numeric_cols if col in synthetic.columns]
    
    moment_diffs = []
    js_divergences = []
    
    for col in common_numeric:
        orig_values = original[col].dropna()
        synth_values = synthetic[col].dropna()
        
        if len(orig_values) > 0 and len(synth_values) > 0:
            # Calculate moments
            orig_moments = {
                "mean": float(orig_values.mean()),
                "std": float(orig_values.std()),
                "skewness": float(stats.skew(orig_values)),
                "kurtosis": float(stats.kurtosis(orig_values))
            }
            
            synth_moments = {
                "mean": float(synth_values.mean()),
                "std": float(synth_values.std()),
                "skewness": float(stats.skew(synth_values)),
                "kurtosis": float(stats.kurtosis(synth_values))
            }
            
            # Calculate moment differences
            moment_diff = sum(abs(orig_moments[k] - synth_moments[k]) for k in orig_moments.keys())
            moment_diffs.append(moment_diff)
            
            # Jensen-Shannon divergence (for histogram comparison)
            try:
                # Create histograms with same bins
                min_val = min(orig_values.min(), synth_values.min())
                max_val = max(orig_values.max(), synth_values.max())
                bins = np.linspace(min_val, max_val, 50)
                
                orig_hist, _ = np.histogram(orig_values, bins=bins, density=True)
                synth_hist, _ = np.histogram(synth_values, bins=bins, density=True)
                
                # Normalize to probability distributions
                orig_hist = orig_hist / orig_hist.sum() if orig_hist.sum() > 0 else orig_hist
                synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist
                
                # Add small epsilon to avoid zeros
                epsilon = 1e-10
                orig_hist = orig_hist + epsilon
                synth_hist = synth_hist + epsilon
                
                js_div = jensenshannon(orig_hist, synth_hist)
                js_divergences.append(float(js_div))
                
            except:
                js_div = float('nan')
            
            distribution_results["moment_analysis"][col] = {
                "original_moments": orig_moments,
                "synthetic_moments": synth_moments,
                "moment_difference": float(moment_diff),
                "jensen_shannon_divergence": float(js_div) if not np.isnan(js_div) else None
            }
    
    # Overall distribution metrics
    if moment_diffs:
        distribution_results["distribution_metrics"] = {
            "mean_moment_difference": float(np.mean(moment_diffs)),
            "mean_js_divergence": float(np.nanmean(js_divergences)) if js_divergences else None,
            "columns_analyzed": int(len(moment_diffs))
        }
    
    return distribution_results


def categorical_analysis(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Analyze categorical variable distributions"""
    
    categorical_results = {
        "test_type": "categorical_analysis",
        "chi_square_tests": {},
        "distribution_comparison": {}
    }
    
    # Get categorical columns
    categorical_cols = original.select_dtypes(include=['object', 'category']).columns
    common_categorical = [col for col in categorical_cols if col in synthetic.columns]
    
    chi_square_results = []
    
    for col in common_categorical:
        orig_values = original[col].dropna()
        synth_values = synthetic[col].dropna()
        
        if len(orig_values) > 0 and len(synth_values) > 0:
            # Get value counts
            orig_counts = orig_values.value_counts()
            synth_counts = synth_values.value_counts()
            
            # Find common categories
            common_categories = set(orig_counts.index) & set(synth_counts.index)
            
            if len(common_categories) > 1:
                # Prepare contingency table for chi-square test
                try:
                    # Create aligned frequency arrays
                    categories = sorted(list(common_categories))
                    orig_freq = [orig_counts.get(cat, 0) for cat in categories]
                    synth_freq = [synth_counts.get(cat, 0) for cat in categories]
                    
                    # Chi-square test
                    if sum(orig_freq) > 0 and sum(synth_freq) > 0:
                        chi2_stat, p_value = stats.chisquare(synth_freq, orig_freq)
                        
                        categorical_results["chi_square_tests"][col] = {
                            "chi_square_statistic": float(chi2_stat),
                            "p_value": float(p_value),
                            "distributions_similar": bool(float(p_value) > 0.05),
                            "categories_tested": int(len(categories)),
                            "total_original_samples": int(sum(orig_freq)),
                            "total_synthetic_samples": int(sum(synth_freq))
                        }
                        
                        chi_square_results.append(p_value)
                        
                except Exception as e:
                    categorical_results["chi_square_tests"][col] = {
                        "error": f"Chi-square test failed: {str(e)}"
                    }
    
    # Overall categorical statistics
    if chi_square_results:
        similar_count = sum(1 for p in chi_square_results if float(p) > 0.05)
        categorical_results["overall_statistics"] = {
            "mean_p_value": float(np.mean(chi_square_results)),
            "columns_similar": int(similar_count),
            "total_columns_tested": int(len(chi_square_results)),
            "similarity_rate": float(similar_count / len(chi_square_results))
        }
    
    return categorical_results


def evaluate_statistical_similarity(original: pd.DataFrame, 
                                   synthetic: pd.DataFrame,
                                   experiment_name: str = "unknown") -> Dict[str, Any]:
    """
    Comprehensive statistical similarity evaluation
    
    Args:
        original: Original dataset
        synthetic: Synthetic dataset  
        experiment_name: Experiment identifier
    
    Returns:
        Complete statistical similarity results
    """
    
    print(f"Evaluating statistical similarity for experiment: {experiment_name}")
    print(f"Original shape: {original.shape}, Synthetic shape: {synthetic.shape}")
    
    results = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "original_shape": list(original.shape),
            "synthetic_shape": list(synthetic.shape),
            "evaluation_type": "statistical_similarity"
        }
    }
    
    # Run all statistical tests
    print("Running Kolmogorov-Smirnov tests...")
    results["kolmogorov_smirnov"] = ks_test_all_columns(original, synthetic)
    
    print("Analyzing correlations...")
    results["correlation_analysis"] = correlation_analysis(original, synthetic)
    
    print("Comparing distributions...")
    results["distribution_comparison"] = distribution_comparison(original, synthetic)
    
    print("Analyzing categorical variables...")
    results["categorical_analysis"] = categorical_analysis(original, synthetic)
    
    # Calculate overall similarity score
    similarity_scores = []
    
    # KS test similarity rate
    if "overall_statistics" in results["kolmogorov_smirnov"]:
        similarity_scores.append(float(results["kolmogorov_smirnov"]["overall_statistics"]["similarity_rate"]))
    
    # Correlation preservation  
    if "correlation_preservation" in results["correlation_analysis"]:
        corr_score = float(results["correlation_analysis"]["correlation_preservation"]["correlation_of_correlations"])
        if not np.isnan(corr_score):
            similarity_scores.append(float(max(0.0, corr_score)))  # Ensure non-negative
    
    # Categorical similarity rate
    if "overall_statistics" in results["categorical_analysis"]:
        similarity_scores.append(float(results["categorical_analysis"]["overall_statistics"]["similarity_rate"]))
    
    # Overall similarity score
    if similarity_scores:
        results["overall_similarity_score"] = float(np.mean(similarity_scores))
    else:
        results["overall_similarity_score"] = 0.0
    
    print(f"Overall statistical similarity score: {results['overall_similarity_score']:.3f}")
    
    # Ensure all results are JSON serializable
    results = ensure_json_serializable(results)
    
    return results


def generate_statistical_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable statistical similarity report"""
    
    report = f"""
Statistical Similarity Evaluation Report
=======================================

Experiment: {results['metadata']['experiment_name']}
Timestamp: {results['metadata']['evaluation_timestamp']}
Dataset Shapes: Original {tuple(results['metadata']['original_shape'])}, Synthetic {tuple(results['metadata']['synthetic_shape'])}

Overall Similarity Score: {results['overall_similarity_score']:.3f}

Distribution Tests (Kolmogorov-Smirnov)
--------------------------------------
"""
    
    if "overall_statistics" in results["kolmogorov_smirnov"]:
        ks_stats = results["kolmogorov_smirnov"]["overall_statistics"]
        report += f"""Columns with similar distributions: {ks_stats['columns_similar']}/{ks_stats['total_columns_tested']}
Similarity rate: {ks_stats['similarity_rate']:.3f}
Mean p-value: {ks_stats['mean_p_value']:.4f}
"""
    
    # Correlation Analysis
    if "correlation_preservation" in results["correlation_analysis"]:
        corr_stats = results["correlation_analysis"]["correlation_preservation"]
        report += f"""
Correlation Analysis
-------------------
Correlation preservation: {corr_stats['correlation_of_correlations']:.3f}
Mean absolute difference: {corr_stats['mean_absolute_difference']:.4f}
"""
    
    # Categorical Analysis
    if "overall_statistics" in results["categorical_analysis"]:
        cat_stats = results["categorical_analysis"]["overall_statistics"]
        report += f"""
Categorical Variables
--------------------
Columns with similar distributions: {cat_stats['columns_similar']}/{cat_stats['total_columns_tested']}
Similarity rate: {cat_stats['similarity_rate']:.3f}
"""
    
    # Quality Assessment
    score = results['overall_similarity_score']
    if score >= 0.9:
        assessment = "Excellent similarity"
    elif score >= 0.8:
        assessment = "Good similarity"
    elif score >= 0.7:
        assessment = "Moderate similarity"
    else:
        assessment = "Poor similarity"
    
    report += f"""
Assessment: {assessment}
"""
    
    return report
