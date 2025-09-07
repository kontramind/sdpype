# sdpype/evaluation/downstream.py
"""
Downstream Task Evaluation - ML Performance Comparison

This module evaluates synthetic data utility by training ML models on both
original and synthetic data and comparing their performance.
"""

import warnings
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ML Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def detect_task_type(data: pd.DataFrame, target_column: str) -> str:
    """
    Automatically detect if this is a classification or regression task
    
    Args:
        data: DataFrame containing the target column
        target_column: Name of the target column
        
    Returns:
        'classification' or 'regression'
    """
    
    target = data[target_column]

    # Check if target is clearly categorical
    if target.dtype == 'object' or target.dtype.name == 'category':
        return 'classification'

    # For numeric targets, use heuristics
    unique_values = target.nunique()
    total_values = len(target)

    # If very few unique values relative to total, likely classification
    if unique_values <= 20 or (unique_values / total_values) < 0.05:
        return 'classification'
    else:
        return 'regression'


def prepare_ml_data(data: pd.DataFrame, target_column: str, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for ML training by handling encoding and scaling

    Args:
        data: Input DataFrame
        target_column: Name of the target column
        task_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (X, y) arrays ready for ML training
    """

    # Separate features and target
    X = data.drop(columns=[target_column]).copy()
    y = data[target_column].copy()

    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle target encoding for classification
    if task_type == 'classification':
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def get_ml_models(task_type: str, random_state: int = 42) -> Dict[str, Any]:
    """
    Get appropriate ML models for the task type

    Args:
        task_type: 'classification' or 'regression'
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of model name -> model instance
    """

    if task_type == 'classification':
        return {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
            'SVM': SVC(random_state=random_state, probability=True),
            'MLP': MLPClassifier(random_state=random_state, max_iter=500, hidden_layer_sizes=(100,))
        }
    else:  # regression
        return {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(),
            'MLP': MLPRegressor(random_state=random_state, max_iter=500, hidden_layer_sizes=(100,))
        }


def evaluate_model_performance(X: np.ndarray, y: np.ndarray, model: Any, task_type: str, cv_folds: int = 5) -> Dict[str, float]:
    """
    Evaluate model performance using cross-validation
    
    Args:
        X: Feature matrix
        y: Target vector
        model: ML model to evaluate
        task_type: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary of metric name -> score
    """

    try:
        if task_type == 'classification':
            # Use stratified CV for classification
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Get cross-validation scores
            accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            
            # Train model for additional metrics
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': float(np.mean(accuracy_scores)),
                'accuracy_std': float(np.std(accuracy_scores)),
                'f1_weighted': float(np.mean(f1_scores)),
                'f1_weighted_std': float(np.std(f1_scores)),
                'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            }
            
            # Add AUC for binary classification
            if len(np.unique(y)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
                except:
                    pass  # Skip if predict_proba not available

        else:  # regression
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Get cross-validation scores
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            neg_mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            neg_mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')

            # Train model for predictions
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                'r2': float(np.mean(r2_scores)),
                'r2_std': float(np.std(r2_scores)),
                'mse': float(-np.mean(neg_mse_scores)),
                'mse_std': float(np.std(neg_mse_scores)),
                'mae': float(-np.mean(neg_mae_scores)),
                'mae_std': float(np.std(neg_mae_scores)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))
            }

        return metrics

    except Exception as e:
        print(f"Warning: Error evaluating model: {e}")
        # Return default metrics on error
        if task_type == 'classification':
            return {'accuracy': 0.0, 'f1_weighted': 0.0, 'precision_weighted': 0.0, 'recall_weighted': 0.0}
        else:
            return {'r2': 0.0, 'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}


def evaluate_downstream_tasks(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    task_type: Optional[str] = None,
    experiment_name: str = "downstream_evaluation",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Main function to evaluate downstream task performance
    
    Args:
        original_data: Original dataset
        synthetic_data: Synthetic dataset
        target_column: Name of the target column for ML tasks
        task_type: 'classification' or 'regression', auto-detected if None
        experiment_name: Name for this evaluation experiment
        random_state: Random seed for reproducibility

    Returns:
        Comprehensive evaluation results
    """

    print(f"🎯 Starting downstream task evaluation: {experiment_name}")
    print(f"📊 Original data shape: {original_data.shape}")
    print(f"📊 Synthetic data shape: {synthetic_data.shape}")

    # Auto-detect task type if not provided
    if task_type is None:
        task_type = detect_task_type(original_data, target_column)

    print(f"🎯 Detected task type: {task_type}")

    # Validate target column exists in both datasets
    if target_column not in original_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in original data")
    if target_column not in synthetic_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in synthetic data")

    # Prepare data for ML
    print("🔄 Preparing original data for ML...")
    X_orig, y_orig = prepare_ml_data(original_data, target_column, task_type)

    print("🔄 Preparing synthetic data for ML...")
    X_synth, y_synth = prepare_ml_data(synthetic_data, target_column, task_type)

    # Get appropriate models
    models = get_ml_models(task_type, random_state)

    # Evaluate each model on both datasets
    results = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "target_column": target_column,
            "original_data_shape": original_data.shape,
            "synthetic_data_shape": synthetic_data.shape,
            "models_evaluated": list(models.keys())
        },
        "original_performance": {},
        "synthetic_performance": {},
        "performance_comparison": {},
        "utility_scores": {}
    }

    print(f"📈 Evaluating {len(models)} models...")

    for model_name, model in models.items():
        print(f"  🤖 Evaluating {model_name}...")

        # Evaluate on original data
        try:
            orig_metrics = evaluate_model_performance(X_orig, y_orig, model, task_type)
            results["original_performance"][model_name] = orig_metrics
            print(f"    ✅ Original data: {get_primary_metric(orig_metrics, task_type):.3f}")
        except Exception as e:
            print(f"    ❌ Original data failed: {e}")
            results["original_performance"][model_name] = {}

        # Evaluate on synthetic data
        try:
            synth_metrics = evaluate_model_performance(X_synth, y_synth, model, task_type)
            results["synthetic_performance"][model_name] = synth_metrics
            print(f"    ✅ Synthetic data: {get_primary_metric(synth_metrics, task_type):.3f}")
        except Exception as e:
            print(f"    ❌ Synthetic data failed: {e}")
            results["synthetic_performance"][model_name] = {}

        # Calculate utility scores (how well synthetic preserves performance)
        if model_name in results["original_performance"] and model_name in results["synthetic_performance"]:
            orig_perf = results["original_performance"][model_name]
            synth_perf = results["synthetic_performance"][model_name]
            
            utility_score = calculate_utility_score(orig_perf, synth_perf, task_type)
            results["utility_scores"][model_name] = utility_score
            print(f"    📊 Utility score: {utility_score:.3f}")

    # Calculate overall utility score
    if results["utility_scores"]:
        results["overall_utility_score"] = float(np.mean(list(results["utility_scores"].values())))
        print(f"\n🎯 Overall utility score: {results['overall_utility_score']:.3f}")
    else:
        results["overall_utility_score"] = 0.0
        print(f"\n❌ No successful model evaluations")

    return results


def get_primary_metric(metrics: Dict[str, float], task_type: str) -> float:
    """Get the primary metric for a task type"""
    if task_type == 'classification':
        return metrics.get('accuracy', 0.0)
    else:
        return metrics.get('r2', 0.0)


def calculate_utility_score(orig_metrics: Dict[str, float], synth_metrics: Dict[str, float], task_type: str) -> float:
    """
    Calculate utility score (how well synthetic data preserves ML performance)

    Args:
        orig_metrics: Performance metrics on original data
        synth_metrics: Performance metrics on synthetic data
        task_type: 'classification' or 'regression'

    Returns:
        Utility score between 0 and 1 (1 = perfect preservation)
    """

    if not orig_metrics or not synth_metrics:
        return 0.0

    if task_type == 'classification':
        # Use accuracy as primary metric
        orig_acc = orig_metrics.get('accuracy', 0.0)
        synth_acc = synth_metrics.get('accuracy', 0.0)

        if orig_acc == 0.0:
            return 0.0

        # Calculate preservation ratio, capped at 1.0
        utility = min(synth_acc / orig_acc, 1.0)

    else:  # regression
        # Use R² as primary metric
        orig_r2 = orig_metrics.get('r2', 0.0)
        synth_r2 = synth_metrics.get('r2', 0.0)

        # Handle negative R² scores
        if orig_r2 <= 0:
            return 0.0

        # Calculate preservation ratio, capped at 1.0
        utility = min(max(synth_r2 / orig_r2, 0.0), 1.0)

    return float(utility)


def generate_downstream_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable report of downstream evaluation results

    Args:
        results: Results from evaluate_downstream_tasks

    Returns:
        Formatted text report
    """

    metadata = results["metadata"]
    task_type = metadata["task_type"]
    
    report = f"""
=============================================================================
DOWNSTREAM TASK EVALUATION REPORT
=============================================================================

Experiment: {metadata["experiment_name"]}
Generated: {metadata["evaluation_timestamp"]}
Task Type: {task_type.upper()}
Target Column: {metadata["target_column"]}

Data Summary:
- Original data shape: {metadata["original_data_shape"]}
- Synthetic data shape: {metadata["synthetic_data_shape"]}
- Models evaluated: {', '.join(metadata["models_evaluated"])}

=============================================================================
MODEL PERFORMANCE COMPARISON
=============================================================================
"""

    # Performance table
    if task_type == 'classification':
        primary_metric = 'accuracy'
        report += f"\n{'Model':<20} {'Original Acc':<15} {'Synthetic Acc':<15} {'Utility Score':<15}\n"
        report += "-" * 70 + "\n"
    else:
        primary_metric = 'r2'
        report += f"\n{'Model':<20} {'Original R²':<15} {'Synthetic R²':<15} {'Utility Score':<15}\n"
        report += "-" * 70 + "\n"

    for model_name in metadata["models_evaluated"]:
        orig_perf = results["original_performance"].get(model_name, {}).get(primary_metric, 0.0)
        synth_perf = results["synthetic_performance"].get(model_name, {}).get(primary_metric, 0.0)
        utility = results["utility_scores"].get(model_name, 0.0)

        report += f"{model_name:<20} {orig_perf:<15.3f} {synth_perf:<15.3f} {utility:<15.3f}\n"

    # Overall assessment
    overall_utility = results["overall_utility_score"]

    report += f"\n=============================================================================\n"
    report += f"OVERALL ASSESSMENT\n"
    report += f"=============================================================================\n"
    report += f"Overall Utility Score: {overall_utility:.3f}\n\n"

    if overall_utility >= 0.9:
        report += "✅ EXCELLENT: Synthetic data preserves ML performance very well\n"
    elif overall_utility >= 0.8:
        report += "✅ GOOD: Synthetic data preserves most ML performance\n"
    elif overall_utility >= 0.7:
        report += "⚠️  MODERATE: Synthetic data has reasonable ML utility\n"
    elif overall_utility >= 0.5:
        report += "⚠️  POOR: Synthetic data has limited ML utility\n"
    else:
        report += "❌ VERY POOR: Synthetic data has very low ML utility\n"

    report += "\nInterpretation:\n"
    report += "- Utility Score = (Synthetic Performance) / (Original Performance)\n"
    report += "- Score of 1.0 = Perfect preservation of ML performance\n"
    report += "- Score of 0.8+ = Good synthetic data quality for ML tasks\n"
    report += "- Score below 0.7 suggests synthetic data may not be suitable for ML\n"

    return report
