"""
Statistical metrics evaluation script for SDPype - Alpha Precision and PRDC
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from sdpype.evaluation.statistical import evaluate_statistical_metrics, generate_statistical_report

console = Console()

@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run statistical metrics evaluation between original and synthetic data"""

    # Check if statistical metrics are configured
    metrics_config = cfg.get("evaluation", {}).get("statistical_similarity", {}).get("metrics", [])

    if not metrics_config:
        print("📊 No statistical metrics configured, skipping evaluation")
        return

    print(f"📊 Starting statistical evaluation with {len(metrics_config)} metrics...")
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
    # Run statistical metrics evaluation
    print("🔄 Running statistical metrics analysis...")
    statistical_results = evaluate_statistical_metrics(
        original_data,
        synthetic_data,
        metrics_config,
        experiment_name=f"{cfg.experiment.name}_seed_{cfg.experiment.seed}"
    )

    # Save statistical similarity results
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    with open(f"experiments/metrics/statistical_similarity_{cfg.experiment.name}_{cfg.experiment.seed}.json", "w") as f:
        json.dump(statistical_results, f, indent=2)

    print(f"📊 Statistical metrics results saved: experiments/metrics/statistical_similarity_{cfg.experiment.name}_{cfg.experiment.seed}.json")

    # Generate and save human-readable report
    report = generate_statistical_report(statistical_results)
    with open(f"experiments/metrics/statistical_report_{cfg.experiment.name}_{cfg.experiment.seed}.txt", "w") as f:
        f.write(report)

    print(f"📋 Statistical metrics report saved: experiments/metrics/statistical_report_{cfg.experiment.name}_{cfg.experiment.seed}.txt")

    # Print individual metrics summary
    console.print("\n📊 Statistical Metrics Summary:", style="bold cyan")

    metrics = statistical_results.get("metrics", {})
    if "alpha_precision" in metrics and metrics["alpha_precision"]["status"] == "success":
        scores = metrics["alpha_precision"]["scores"]

        # Create Alpha Precision results table
        table = Table(title="✅ Alpha Precision Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("OC Variant", style="green", justify="right")
        table.add_column("Naive Variant", style="yellow", justify="right")

        table.add_row(
            "Delta Precision Alpha",
            f"{scores['delta_precision_alpha_OC']:.3f}",
            f"{scores['delta_precision_alpha_naive']:.3f}"
        )
        table.add_row(
            "Delta Coverage Beta",
            f"{scores['delta_coverage_beta_OC']:.3f}",
            f"{scores['delta_coverage_beta_naive']:.3f}"
        )
        table.add_row(
            "Authenticity",
            f"{scores['authenticity_OC']:.3f}",
            f"{scores['authenticity_naive']:.3f}"
        )

        console.print(table)
    else:
        console.print("❌ Alpha Precision failed", style="bold red")

    console.print("\n✅ Statistical metrics evaluation completed", style="bold green")


if __name__ == "__main__":
    main()
