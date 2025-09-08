# sdpype/evaluate_downstream.py
"""
Downstream Task Evaluation - Main Module

Evaluates synthetic data utility by training ML models on both original and 
synthetic data and comparing their performance on downstream tasks.
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from sdpype.evaluation.downstream import evaluate_downstream_tasks, generate_downstream_report


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Main downstream task evaluation function"""
    
    print("🎯 Starting Downstream Task Evaluation")
    print(f"🎲 Experiment seed: {cfg.experiment.seed}")
    
    # Check if downstream evaluation is enabled
    if not cfg.evaluation.downstream_tasks.enabled:
        print("⚠️  Downstream task evaluation is disabled in configuration")
        print("💡 Enable with: evaluation.downstream_tasks.enabled=true")
        return
    
    # Define data paths
    original_data_path = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
    synthetic_data_path = f"experiments/data/synthetic/synthetic_data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
    
    # Check if data files exist
    if not Path(original_data_path).exists():
        print(f"❌ Original data not found: {original_data_path}")
        print("💡 Run preprocessing first: dvc repro -s preprocess")
        raise FileNotFoundError(f"Original data file not found: {original_data_path}")
    
    if not Path(synthetic_data_path).exists():
        print(f"❌ Synthetic data not found: {synthetic_data_path}")
        print("💡 Run generation first: dvc repro -s generate_synthetic")
        raise FileNotFoundError(f"Synthetic data file not found: {synthetic_data_path}")
    
    # Load datasets
    print(f"📂 Loading original data: {original_data_path}")
    original_data = pd.read_csv(original_data_path)
    
    print(f"📂 Loading synthetic data: {synthetic_data_path}")
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    print(f"📊 Original data shape: {original_data.shape}")
    print(f"📊 Synthetic data shape: {synthetic_data.shape}")
    
    # Get target column from configuration
    target_column = cfg.evaluation.downstream_tasks.target_column
    
    # Validate target column exists
    if target_column not in original_data.columns:
        available_columns = list(original_data.columns)
        print(f"❌ Target column '{target_column}' not found in original data")
        print(f"💡 Available columns: {available_columns}")
        raise ValueError(f"Target column '{target_column}' not found. Available: {available_columns}")
    
    if target_column not in synthetic_data.columns:
        available_columns = list(synthetic_data.columns)
        print(f"❌ Target column '{target_column}' not found in synthetic data")
        print(f"💡 Available columns: {available_columns}")
        raise ValueError(f"Target column '{target_column}' not found. Available: {available_columns}")
    
    print(f"🎯 Target column: {target_column}")
    
    # Get evaluation configuration
    task_type = cfg.evaluation.downstream_tasks.get("task_type", None)  # None = auto-detect
    random_state = cfg.evaluation.downstream_tasks.get("random_state", cfg.experiment.seed)
    
    # Create experiment name
    experiment_name = f"{cfg.experiment.get('name', 'downstream_eval')}_seed_{cfg.experiment.seed}"
    
    # Run downstream evaluation
    print(f"🚀 Starting downstream task evaluation...")
    
    try:
        results = evaluate_downstream_tasks(
            original_data=original_data,
            synthetic_data=synthetic_data,
            target_column=target_column,
            task_type=task_type,
            experiment_name=experiment_name,
            random_state=random_state
        )
        
        print(f"✅ Downstream evaluation completed successfully")
        
    except Exception as e:
        print(f"❌ Error during downstream evaluation: {e}")
        raise
    
    # Ensure output directories exist
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_filename = f"experiments/metrics/downstream_performance_{cfg.experiment.name}_{cfg.experiment.seed}.json"
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved: {results_filename}")
    
    # Generate and save human-readable report
    report = generate_downstream_report(results)
    report_filename = f"experiments/metrics/downstream_report_{cfg.experiment.name}_{cfg.experiment.seed}.txt"
    with open(report_filename, "w") as f:
        f.write(report)
    
    print(f"📋 Report saved: {report_filename}")
    
    # Print summary
    overall_utility = results["overall_utility_score"]
    task_type = results["metadata"]["task_type"]
    models_count = len(results["metadata"]["models_evaluated"])
    
    print(f"\n🎯 Downstream Task Evaluation Summary:")
    print(f"Task type: {task_type}")
    print(f"Models evaluated: {models_count}")
    print(f"Overall utility score: {overall_utility:.3f}")

    # Provide interpretation
    if overall_utility >= 0.9:
        print("🎉 Excellent! Synthetic data preserves ML performance very well")
    elif overall_utility >= 0.8:
        print("✅ Good! Synthetic data preserves most ML performance")
    elif overall_utility >= 0.7:
        print("✅ Moderate synthetic data quality for ML tasks")
    elif overall_utility >= 0.5:
        print("⚠️  Poor synthetic data quality - limited ML utility")
    else:
        print("❌ Very poor synthetic data quality - not suitable for ML")

    # Show best performing model
    if results["utility_scores"]:
        best_model = max(results["utility_scores"].items(), key=lambda x: x[1])
        print(f"🏆 Best model: {best_model[0]} (utility: {best_model[1]:.3f})")

    # Recommendations
    print(f"\n💡 Recommendations:")
    if overall_utility < 0.7:
        print("  • Consider different SDG model parameters")
        print("  • Try different SDG algorithms (e.g., CTGAN → TVAE)")
        print("  • Increase training data size or training epochs")
        print("  • Check data preprocessing steps")
    elif overall_utility < 0.9:
        print("  • Fine-tune SDG model hyperparameters")
        print("  • Consider ensemble of multiple SDG models")
    else:
        print("  • Great results! Synthetic data is ready for downstream tasks")

    print("\n✅ Downstream task evaluation completed")


if __name__ == "__main__":
    main()
