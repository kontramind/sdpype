"""
Enhanced evaluation script for SDPype - integrates intrinsic quality + statistical similarity
"""

import json
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

from sdpype.evaluation import evaluate_data_quality, compare_quality_metrics
from sdpype.evaluation.statistical import evaluate_statistical_similarity, generate_statistical_report  # ✨ NEW


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run comprehensive data evaluation - intrinsic quality + statistical similarity"""
    
    # Get data type from Hydra config (default to "both")
    data_type = cfg.get("evaluation_data_type", "both")
    
    print(f"🔬 Starting evaluation for: {data_type}")
    print(f"Experiment seed: {cfg.experiment.seed}")
    
    results = {}
    
    # === INTRINSIC QUALITY EVALUATION (EXISTING FUNCTIONALITY) ===

    # Evaluate original data
    if data_type in ["original", "both"]:
        original_data_path = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
        
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
            with open(f"experiments/metrics/quality_original_{cfg.experiment.name}_{cfg.experiment.seed}.json", "w") as f:
                json.dump(original_results, f, indent=2)
            
            print(f"Original data quality saved: experiments/metrics/quality_original_{cfg.experiment.name}_{cfg.experiment.seed}.json")
        else:
            print(f"Original data not found: {original_data_path}")
    
    # Evaluate synthetic data  
    if data_type in ["synthetic", "both"]:
        synthetic_data_path = f"experiments/data/synthetic/synthetic_data_{cfg.experiment.name}_{cfg.experiment.seed}.csv" 

        if Path(synthetic_data_path).exists():
            print(f"Evaluating synthetic data: {synthetic_data_path}")
            synthetic_data = pd.read_csv(synthetic_data_path)
            
            synthetic_results = evaluate_data_quality(
                synthetic_data,
                data_source=f"synthetic_seed_{cfg.experiment.seed}"
            )
            results["synthetic"] = synthetic_results
            
            # Save synthetic data quality metrics
            with open(f"experiments/metrics/quality_synthetic_{cfg.experiment.name}_{cfg.experiment.seed}.json", "w") as f:
                json.dump(synthetic_results, f, indent=2)
            
            print(f"Synthetic data quality saved: experiments/metrics/quality_synthetic_{cfg.experiment.name}_{cfg.experiment.seed}.json")
        else:
            print(f"Synthetic data not found: {synthetic_data_path}")
    
    # Generate intrinsic quality comparison if both datasets evaluated
    if "original" in results and "synthetic" in results:
        print("Generating quality comparison...")
        
        comparison = compare_quality_metrics(results["original"], results["synthetic"])
        
        # Save comparison results
        with open(f"experiments/metrics/quality_comparison_{cfg.experiment.name}_{cfg.experiment.seed}.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Quality comparison saved: experiments/metrics/quality_comparison_{cfg.experiment.name}_{cfg.experiment.seed}.json")
        
        # Print summary
        score_diff = comparison["overall_score_comparison"]["score_difference"]
        preservation_rate = comparison["overall_score_comparison"]["quality_preservation_rate"]
        
        print(f"\n📊 Quality Comparison Summary:")
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

    # === STATISTICAL SIMILARITY EVALUATION (NEW FUNCTIONALITY) ===

    # Run statistical similarity evaluation if enabled and both datasets available
    if (data_type == "both" and
        "original" in results and
        "synthetic" in results and
        cfg.evaluation.statistical_similarity.enabled):

        print(f"\n📈 Running Statistical Similarity Evaluation...")

        # Load original and synthetic data for comparison
        original_data_path = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
        synthetic_data_path = f"experiments/data/synthetic/synthetic_data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
        original_data = pd.read_csv(original_data_path)
        synthetic_data = pd.read_csv(synthetic_data_path)

        # Run statistical similarity evaluation
        statistical_results = evaluate_statistical_similarity(
            original_data,
            synthetic_data,
            experiment_name=f"{cfg.experiment.name}_seed_{cfg.experiment.seed}"
        )

        # Save statistical similarity results
        with open(f"experiments/metrics/statistical_similarity_{cfg.experiment.name}_{cfg.experiment.seed}.json", "w") as f:
            json.dump(statistical_results, f, indent=2)

        print(f"Statistical similarity results saved: experiments/metrics/statistical_similarity_{cfg.experiment.name}_{cfg.experiment.seed}.json")

        # Generate and save human-readable report
        report = generate_statistical_report(statistical_results)
        with open(f"experiments/metrics/statistical_report_{cfg.experiment.name}_{cfg.experiment.seed}.txt", "w") as f:
            f.write(report)

        print(f"Statistical similarity report saved: experiments/metrics/statistical_report_{cfg.experiment.name}_{cfg.experiment.seed}.txt")

        # Print statistical summary
        similarity_score = statistical_results["overall_similarity_score"]
        print(f"\n📊 Statistical Similarity Summary:")
        print(f"Overall similarity score: {similarity_score:.3f}")

        if similarity_score >= 0.9:
            print("✅ Excellent statistical similarity")
        elif similarity_score >= 0.8:
            print("✅ Good statistical similarity")
        elif similarity_score >= 0.7:
            print("⚠️  Moderate statistical similarity")
        else:
            print("❌ Poor statistical similarity")

        # Combined evaluation summary
        if "original" in results and "synthetic" in results:
            quality_preservation = comparison["overall_score_comparison"]["quality_preservation_rate"]
            combined_score = (quality_preservation + similarity_score) / 2

            print(f"\n🎯 Combined Evaluation:")
            print(f"Quality preservation: {quality_preservation:.1%}")
            print(f"Statistical similarity: {similarity_score:.3f}")
            print(f"Combined score: {combined_score:.3f}")

            if combined_score >= 0.9:
                print("🎉 Outstanding synthetic data quality!")
            elif combined_score >= 0.8:
                print("✅ High-quality synthetic data")
            elif combined_score >= 0.7:
                print("✅ Good synthetic data quality")
            else:
                print("⚠️  Synthetic data may need improvement")

    elif cfg.evaluation.statistical_similarity.enabled and data_type == "both":
        print("\n⚠️  Statistical similarity evaluation enabled but data not available")
    
    elif cfg.evaluation.statistical_similarity.enabled and data_type != "both":
        print(f"\n💡 Statistical similarity evaluation requires data_type='both' (current: '{data_type}')")

    print("\n✅ Evaluation completed")


if __name__ == "__main__":
    main()
