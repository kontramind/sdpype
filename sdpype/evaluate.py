"""
Enhanced evaluation script for SDPype - integrates intrinsic quality + statistical similarity
"""

import json
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

from sdpype.evaluation.statistical import evaluate_statistical_similarity, generate_statistical_report


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run statistical similarity evaluation between original and synthetic data"""

    print("📊 Starting statistical similarity evaluation...")
    print(f"Experiment seed: {cfg.experiment.seed}")

    # Load datasets for statistical comparison
    original_data_path = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
    synthetic_data_path = f"experiments/data/synthetic/synthetic_data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"

    if not Path(original_data_path).exists():
        raise FileNotFoundError(f"Original data not found: {original_data_path}")
    if not Path(synthetic_data_path).exists():
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_data_path}")

    print(f"📊 Loading original data: {original_data_path}")
    print(f"📊 Loading synthetic data: {synthetic_data_path}")

    original_data = pd.read_csv(original_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    # Run statistical similarity evaluation
    print("🔄 Running statistical similarity analysis...")
    statistical_results = evaluate_statistical_similarity(
        original_data,
        synthetic_data,
        experiment_name=f"{cfg.experiment.name}_seed_{cfg.experiment.seed}"
    )

    # Save statistical similarity results
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    with open(f"experiments/metrics/statistical_similarity_{cfg.experiment.name}_{cfg.experiment.seed}.json", "w") as f:
        json.dump(statistical_results, f, indent=2)

    print(f"📊 Statistical similarity results saved: experiments/metrics/statistical_similarity_{cfg.experiment.name}_{cfg.experiment.seed}.json")

    # Generate and save human-readable report
    report = generate_statistical_report(statistical_results)
    with open(f"experiments/metrics/statistical_report_{cfg.experiment.name}_{cfg.experiment.seed}.txt", "w") as f:
        f.write(report)

    print(f"📋 Statistical similarity report saved: experiments/metrics/statistical_report_{cfg.experiment.name}_{cfg.experiment.seed}.txt")

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

    print("\n✅ Statistical similarity evaluation completed")


if __name__ == "__main__":
    main()
