"""
Main evaluation script for SDPype - integrates with DVC pipeline
"""

import json
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

from sdpype.evaluation import evaluate_data_quality, compare_quality_metrics


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run data quality evaluation"""
    
    # Get data type from Hydra config (default to "both")
    data_type = cfg.get("evaluation_data_type", "both")
    
    print(f"Starting evaluation for: {data_type}")
    print(f"Experiment seed: {cfg.experiment.seed}")
    
    results = {}
    
    # Evaluate original data
    if data_type in ["original", "both"]:
        original_data_path = f"experiments/data/processed/data_{cfg.experiment.seed}.csv"
        
        if Path(original_data_path).exists():
            print(f"Evaluating original data: {original_data_path}")
            original_data = pd.read_csv(original_data_path)
            
            original_results = evaluate_data_quality(
                original_data, 
                data_source=f"original_seed_{cfg.experiment.seed}"
            )
            results["original"] = original_results
            
            # Save original data quality metrics
            Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
            with open(f"experiments/metrics/quality_original_{cfg.experiment.seed}.json", "w") as f:
                json.dump(original_results, f, indent=2)
            
            print(f"Original data quality saved: experiments/metrics/quality_original_{cfg.experiment.seed}.json")
        else:
            print(f"Original data not found: {original_data_path}")
    
    # Evaluate synthetic data  
    if data_type in ["synthetic", "both"]:
        synthetic_data_path = f"experiments/data/synthetic/synthetic_data_{cfg.experiment.seed}.csv"
        
        if Path(synthetic_data_path).exists():
            print(f"Evaluating synthetic data: {synthetic_data_path}")
            synthetic_data = pd.read_csv(synthetic_data_path)
            
            synthetic_results = evaluate_data_quality(
                synthetic_data,
                data_source=f"synthetic_seed_{cfg.experiment.seed}"
            )
            results["synthetic"] = synthetic_results
            
            # Save synthetic data quality metrics
            with open(f"experiments/metrics/quality_synthetic_{cfg.experiment.seed}.json", "w") as f:
                json.dump(synthetic_results, f, indent=2)
            
            print(f"Synthetic data quality saved: experiments/metrics/quality_synthetic_{cfg.experiment.seed}.json")
        else:
            print(f"Synthetic data not found: {synthetic_data_path}")
    
    # Generate comparison if both datasets evaluated
    if "original" in results and "synthetic" in results:
        print("Generating quality comparison...")
        
        comparison = compare_quality_metrics(results["original"], results["synthetic"])
        
        # Save comparison results
        with open(f"experiments/metrics/quality_comparison_{cfg.experiment.seed}.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Quality comparison saved: experiments/metrics/quality_comparison_{cfg.experiment.seed}.json")
        
        # Print summary
        score_diff = comparison["overall_score_comparison"]["score_difference"]
        preservation_rate = comparison["overall_score_comparison"]["quality_preservation_rate"]
        
        print(f"\nQuality Comparison Summary:")
        print(f"Original quality score: {results['original']['overall_quality_score']:.3f}")
        print(f"Synthetic quality score: {results['synthetic']['overall_quality_score']:.3f}")
        print(f"Quality difference: {score_diff:+.3f}")
        print(f"Quality preservation: {preservation_rate:.1%}")
        
        if preservation_rate >= 0.95:
            print("✅ Excellent quality preservation")
        elif preservation_rate >= 0.85:
            print("✅ Good quality preservation")
        elif preservation_rate >= 0.75:
            print("⚠️  Moderate quality preservation")
        else:
            print("❌ Poor quality preservation")
    
    print("✅ Evaluation completed")


if __name__ == "__main__":
    main()
