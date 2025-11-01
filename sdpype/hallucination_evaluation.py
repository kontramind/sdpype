"""
Hallucination evaluation script for SDPype - DDR and Plausibility Metrics
"""

import json
from pathlib import Path
from typing import Dict, Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from sdv.metadata import SingleTableMetadata
from sdpype.evaluation.hallucination import evaluate_hallucination_metrics, generate_hallucination_report
from sdpype.metadata import load_csv_with_metadata

console = Console()


def _get_config_hash() -> str:
    """Get config hash from temporary file created during pipeline execution"""
    try:
        if Path('.sdpype_config_hash').exists():
            with open('.sdpype_config_hash', 'r') as f:
                return f.read().strip()
        return "nohash"
    except Exception:
        return "nohash"


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run hallucination evaluation between population, training, and synthetic data"""

    # Validate hallucination configuration exists
    halluc_config = cfg.get("evaluation", {}).get("hallucination", {})

    if not halluc_config:
        raise ValueError(
            "Hallucination evaluation configuration not found in params.yaml.\n"
            "Please add 'evaluation.hallucination' section with 'query_file' parameter."
        )

    print("🔍 Starting hallucination evaluation (DDR + Plausibility)...")
    print(f"Experiment seed: {cfg.experiment.seed}")

    config_hash = _get_config_hash()
    base_name = f"{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}"

    # Validate required configuration
    query_file = halluc_config.get("query_file")
    if not query_file:
        raise ValueError(
            "Query file not specified in params.yaml.\n"
            "Please set 'evaluation.hallucination.query_file' to your SQL validation file."
        )

    query_file_path = Path(query_file)
    if not query_file_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_file_path}")

    # Get dataset paths
    metadata_path = cfg.data.metadata_file
    population_file = cfg.data.get("population_file")
    training_file = cfg.data.training_file
    synthetic_decoded_path = f"experiments/data/synthetic/synthetic_data_{base_name}_decoded.csv"

    # Validate population file is configured
    if not population_file:
        raise ValueError(
            "Population file not specified in params.yaml.\n"
            "Please set 'data.population_file' to your population dataset."
        )

    # Validate all required files exist
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not Path(population_file).exists():
        raise FileNotFoundError(f"Population file not found: {population_file}")
    if not Path(training_file).exists():
        raise FileNotFoundError(f"Training file not found: {training_file}")
    if not Path(synthetic_decoded_path).exists():
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_decoded_path}")

    # Load metadata
    print(f"📋 Loading metadata: {metadata_path}")
    metadata = SingleTableMetadata.load_from_json(metadata_path)

    # Load all datasets with metadata for type consistency
    print(f"📊 Loading population data: {population_file}")
    population_data = load_csv_with_metadata(Path(population_file), Path(metadata_path), low_memory=False)
    print(f"   Loaded {len(population_data):,} rows")

    print(f"📊 Loading training data: {training_file}")
    training_data = load_csv_with_metadata(Path(training_file), Path(metadata_path), low_memory=False)
    print(f"   Loaded {len(training_data):,} rows")

    print(f"📊 Loading synthetic data: {synthetic_decoded_path}")
    synthetic_data = load_csv_with_metadata(Path(synthetic_decoded_path), Path(metadata_path), low_memory=False)
    print(f"   Loaded {len(synthetic_data):,} rows")

    print(f"📋 Using validation queries: {query_file_path}")
    print()

    # Run hallucination evaluation
    print("🔄 Running hallucination metrics analysis...")

    hallucination_results = evaluate_hallucination_metrics(
        population=population_data,
        training=training_data,
        synthetic=synthetic_data,
        metadata=metadata,
        query_file=query_file_path,
        experiment_name=cfg.experiment.name
    )

    # Save JSON results
    results_file = f"experiments/metrics/hallucination_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json"

    print(f"💾 Saving hallucination results to: {results_file}")
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(hallucination_results, f, indent=2)

    # Generate and save human-readable report
    report_file = f"experiments/metrics/hallucination_report_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.txt"
    report_content = generate_hallucination_report(hallucination_results)

    print(f"📋 Generating hallucination report: {report_file}")
    with open(report_file, 'w') as f:
        f.write(report_content)

    # Display results in rich tables
    _display_hallucination_tables(hallucination_results)

    print("✅ Hallucination evaluation completed successfully!")


def _display_hallucination_tables(results: Dict[str, Any]):
    """Display hallucination results in rich formatted tables"""

    console.print()
    console.print("=" * 80, style="blue")
    console.print("  HALLUCINATION SCORE RESULTS", style="bold blue")
    console.print("  DDR + Plausibility Validation", style="blue")
    console.print("=" * 80, style="blue")
    console.print()

    # Extract metrics
    pop_stats = results.get("population_statistics", {})
    train_stats = results.get("training_statistics", {})
    synth_stats = results.get("synthetic_statistics", {})
    matrix_2x2 = results.get("quality_matrix_2x2", {})

    # Create 2×2 Quality Matrix
    matrix_table = Table(
        title="📊 Synthetic Data Quality Matrix (2×2)",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )

    matrix_table.add_column("", style="cyan", justify="left")
    matrix_table.add_column("Factual (P)", justify="center", style="green")
    matrix_table.add_column("Plausible (V)", justify="center", style="blue")

    # Total row
    total_factual = matrix_2x2.get("total_factual", {})
    total_plausible = matrix_2x2.get("total_plausible", {})
    matrix_table.add_row(
        "Total",
        f"{total_factual.get('rate_pct', 0):.2f}%\n{total_factual.get('count', 0):,} records\nS ∩ P",
        f"{total_plausible.get('rate_pct', 0):.2f}%\n{total_plausible.get('count', 0):,} records\nS ∩ V"
    )

    # Novel row (not in training)
    novel_factual = matrix_2x2.get("novel_factual", {})
    novel_plausible = matrix_2x2.get("novel_plausible", {})
    matrix_table.add_row(
        "Novel (T̄)",
        f"{novel_factual.get('rate_pct', 0):.2f}%\n{novel_factual.get('count', 0):,} records\nS ∩ P ∩ T̄",
        f"{novel_plausible.get('rate_pct', 0):.2f}%\n{novel_plausible.get('count', 0):,} records\nS ∩ V ∩ T̄",
        style="bold"
    )

    console.print(matrix_table)
    console.print()

    # Add set notation legend
    legend_panel = Panel.fit(
        """Set Notation:
  S = Synthetic (SDG output, current generation)
  P = Population (ground truth real dataset)
  T = Training (SDG input, current generation)
  V = Valid (passes all validation rules)

  ∩ = Intersection (records in both sets)
  T̄ = NOT in Training (novel, not memorized)""",
        title="📐 Set Theory Reference",
        border_style="blue"
    )

    console.print(legend_panel)
    console.print()

    # Add interpretation guide
    guide_panel = Panel.fit(
        """🎯 Quality Matrix Interpretation:

Factual (S ∩ P):
  • Records that exist in the real population dataset
  • Higher is BETTER - indicates grounding in reality
  • Total: Includes both memorized and novel records
  • Novel (S ∩ P ∩ T̄): Factual AND not memorized - THE IDEAL!

Plausible (S ∩ V):
  • Records passing all validation rules (semantic validity)
  • Higher is BETTER - indicates well-formed data
  • Total: Includes both memorized and novel records
  • Novel (S ∩ V ∩ T̄): Plausible AND not memorized (creative generation)

Key Insights:
  • Novel Factual = DDR (Desirable Diverse Records) - Best outcome
  • High Novel % indicates low memorization (good for privacy/diversity)
  • Factual vs Plausible shows gap between "real" and "looks real"
  • If P is a sample (not complete): Factual underestimates quality""",
        title="📖 Metric Guide",
        border_style="blue"
    )

    console.print(guide_panel)
    console.print()

    # Summary info
    console.print("=" * 80, style="blue")
    console.print("✓ Validation complete!", style="bold green")
    console.print("=" * 80, style="blue")
    console.print()


if __name__ == "__main__":
    main()
