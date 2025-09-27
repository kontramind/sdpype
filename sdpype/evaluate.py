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

from sdv.metadata import SingleTableMetadata
from sdpype.evaluation.statistical import evaluate_statistical_metrics, generate_statistical_report

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
    """Run statistical metrics evaluation between original and synthetic data"""

    # Check if statistical metrics are configured
    metrics_config = cfg.get("evaluation", {}).get("statistical_similarity", {}).get("metrics", [])

    if not metrics_config:
        print("📊 No statistical metrics configured, skipping evaluation")
        return

    print(f"📊 Starting statistical evaluation with {len(metrics_config)} metrics...")
    print(f"Experiment seed: {cfg.experiment.seed}")

    config_hash = _get_config_hash()
    # Load datasets for statistical comparison
    original_data_path = f"experiments/data/processed/data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.csv"
    metadata_path = f"experiments/data/processed/data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}_metadata.json"
    synthetic_data_path = f"experiments/data/synthetic/synthetic_data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.csv"

    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not Path(original_data_path).exists():
        raise FileNotFoundError(f"Original data not found: {original_data_path}")
    if not Path(synthetic_data_path).exists():
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_data_path}")

    print(f"📊 Loading original data: {original_data_path}")
    print(f"📊 Loading synthetic data: {synthetic_data_path}")

    original_data = pd.read_csv(original_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    metadata = SingleTableMetadata.load_from_json(metadata_path)

    # Run statistical similarity evaluation
    # Run statistical metrics evaluation
    print("🔄 Running statistical metrics analysis...")
    statistical_results = evaluate_statistical_metrics(
        original_data,
        synthetic_data,
        metrics_config,
        experiment_name=f"{cfg.experiment.name}_seed_{cfg.experiment.seed}",
        metadata=metadata
    )

    # Save statistical similarity results
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    with open(f"experiments/metrics/statistical_similarity_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json", "w") as f:
        json.dump(statistical_results, f, indent=2)

    print(f"📊 Statistical metrics results saved: experiments/metrics/statistical_similarity_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json")

    # Generate and save human-readable report
    report = generate_statistical_report(statistical_results)
    with open(f"experiments/metrics/statistical_report_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.txt", "w") as f:
        f.write(report)

    print(f"📋 Statistical metrics report saved: experiments/metrics/statistical_report_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.txt")

    # Print individual metrics summary
    console.print("\n📊 Statistical Metrics Summary:", style="bold cyan")

    metrics = statistical_results.get("metrics", {})

    # TableStructure results table
    if "table_structure" in metrics and metrics["table_structure"]["status"] == "success":
        ts_result = metrics["table_structure"]

        # Get parameters info for display
        params_info = ts_result["parameters"]
        params_display = str(params_info) if params_info else "no parameters"

        # Create TableStructure results table
        ts_table = Table(title=f"✅ TableStructure Results ({params_display})", show_header=True, header_style="bold blue")
        ts_table.add_column("Metric", style="cyan", no_wrap=True)
        ts_table.add_column("Score", style="bright_green", justify="right")

        ts_table.add_row("Table Structure Score", f"{ts_result['score']:.3f}")

        console.print(ts_table)
    else:
        console.print("❌ TableStructure failed", style="bold red")

    # NewRowSynthesis results table
    if "new_row_synthesis" in metrics and metrics["new_row_synthesis"]["status"] == "success":
        nrs_result = metrics["new_row_synthesis"]

        # Get parameters info for display
        params_info = nrs_result["parameters"]
        tolerance = params_info.get("numerical_match_tolerance", 0.01)
        sample_size = params_info.get("synthetic_sample_size", "all rows")
        params_display = f"tolerance={tolerance}, sample_size={sample_size}"

        # Create NewRowSynthesis results table
        nrs_table = Table(title=f"✅ NewRowSynthesis Results ({params_display})", show_header=True, header_style="bold blue")
        nrs_table.add_column("Metric", style="cyan", no_wrap=True)
        nrs_table.add_column("Value", style="bright_green", justify="right")

        nrs_table.add_row("New Row Score", f"{nrs_result['score']:.3f}")
        nrs_table.add_row("New Rows", f"{nrs_result['num_new_rows']:,}")
        nrs_table.add_row("Matched Rows", f"{nrs_result['num_matched_rows']:,}")

        console.print(nrs_table)
    else:
        console.print("❌ NewRowSynthesis failed", style="bold red")

    # BoundaryAdherence results table
    if "boundary_adherence" in metrics and metrics["boundary_adherence"]["status"] == "success":
        ba_result = metrics["boundary_adherence"]

        # Get parameters info for display
        params_info = ba_result["parameters"]
        target_cols = params_info.get("target_columns", "all numerical/datetime")
        params_display = f"target_columns={target_cols}"

        # Create BoundaryAdherence results table
        ba_table = Table(title=f"✅ BoundaryAdherence Results ({params_display})", show_header=True, header_style="bold blue")
        ba_table.add_column("Column", style="cyan", no_wrap=True)
        ba_table.add_column("Boundary Score", style="bright_green", justify="right")
        ba_table.add_column("Status", style="yellow", justify="center")

        # Add aggregate score as first row
        if ba_result['aggregate_score'] is not None:
            ba_table.add_row("AGGREGATE", f"{ba_result['aggregate_score']:.3f}", "✓")
        else:
            ba_table.add_row("AGGREGATE", "n/a", "ℹ️")

        ba_table.add_section()

        # Show all columns from the original dataset
        all_columns = list(original_data.columns)
        column_scores = ba_result['column_scores']
        compatible_columns = ba_result['compatible_columns']

        for col in sorted(all_columns):
            if col in column_scores:
                ba_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
            elif col in compatible_columns:
                ba_table.add_row(col, "error", "⚠️")
            else:
                ba_table.add_row(col, "n/a", "—")

        # Add message at the bottom if no compatible columns found
        if ba_result.get("message"):
            ba_table.add_section()
            ba_table.add_row("INFO", ba_result["message"], "ℹ️")

        console.print(ba_table)
    else:
        console.print("❌ BoundaryAdherence failed", style="bold red")

    # CategoryAdherence results table
    if "category_adherence" in metrics and metrics["category_adherence"]["status"] == "success":
        ca_result = metrics["category_adherence"]

        # Get parameters info for display
        params_info = ca_result["parameters"]
        target_cols = params_info.get("target_columns", None)
        params_display = f"target_columns={target_cols}"

        if ca_result.get("message"):
            # Handle case where no compatible columns found
            console.print(f"⚠️  CategoryAdherence Results ({params_display})", style="bold yellow")
            console.print(f"   Status: {ca_result['message']}", style="yellow")
        else:
            # Create CategoryAdherence results table
            ca_table = Table(title=f"✅ CategoryAdherence Results ({params_display})", show_header=True, header_style="bold blue")
            ca_table.add_column("Column", style="cyan", no_wrap=True)
            ca_table.add_column("Category Score", style="bright_green", justify="right")
            ca_table.add_column("Status", style="yellow", justify="center")

            # Add aggregate score first
            ca_table.add_row("AGGREGATE", f"{ca_result['aggregate_score']:.3f}", "✓")

            ca_table.add_section()

            # Show all columns from the original dataset
            all_columns = list(original_data.columns)
            column_scores = ca_result['column_scores']
            compatible_columns = ca_result['compatible_columns']

            for col in sorted(all_columns):
                if col in column_scores:
                    ca_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
                elif col in compatible_columns:
                    ca_table.add_row(col, "error", "⚠️")
                else:
                    ca_table.add_row(col, "n/a", "—")

            # Add message at the bottom if no compatible columns found
            if ca_result.get("message"):
                ca_table.add_section()
                ca_table.add_row("INFO", ca_result["message"], "ℹ️")

            console.print(ca_table)
    else:
        console.print("❌ CategoryAdherence failed", style="bold red")

    # KSComplement results table
    if "ks_complement" in metrics and metrics["ks_complement"]["status"] == "success":
        ks_result = metrics["ks_complement"]

        # Get parameters info for display
        params_info = ks_result["parameters"]
        target_cols = params_info.get("target_columns", "all numerical/datetime")
        params_display = f"target_columns={target_cols}"

        # Create KSComplement results table
        ks_table = Table(title=f"✅ KSComplement Results ({params_display})", show_header=True, header_style="bold blue")
        ks_table.add_column("Column", style="cyan", no_wrap=True)
        ks_table.add_column("KS Score", style="bright_green", justify="right")
        ks_table.add_column("Status", style="yellow", justify="center")

        # Add aggregate score as first row
        if ks_result['aggregate_score'] is not None:
            ks_table.add_row("AGGREGATE", f"{ks_result['aggregate_score']:.3f}", "✓")
        else:
            ks_table.add_row("AGGREGATE", "n/a", "ℹ️")

        ks_table.add_section()

        # Show all columns from the original dataset
        all_columns = list(original_data.columns)
        column_scores = ks_result['column_scores']
        compatible_columns = ks_result['compatible_columns']

        for col in sorted(all_columns):
            if col in column_scores:
                ks_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
            elif col in compatible_columns:
                ks_table.add_row(col, "error", "⚠️")
            else:
                ks_table.add_row(col, "n/a", "—")

        # Add message at the bottom if no compatible columns found
        if ks_result.get("message"):
            ks_table.add_section()
            ks_table.add_row("INFO", ks_result["message"], "ℹ️")

        console.print(ks_table)
    else:
        console.print("❌ KSComplement failed", style="bold red")

    # TVComplement results table
    if "tv_complement" in metrics and metrics["tv_complement"]["status"] == "success":
        tv_result = metrics["tv_complement"]

        # Get parameters info for display
        params_info = tv_result["parameters"]
        target_cols = params_info.get("target_columns", "all categorical/boolean")
        params_display = f"target_columns={target_cols}"

        # Create TVComplement results table
        tv_table = Table(title=f"✅ TVComplement Results ({params_display})", show_header=True, header_style="bold blue")
        tv_table.add_column("Column", style="cyan", no_wrap=True)
        tv_table.add_column("TV Score", style="bright_green", justify="right")
        tv_table.add_column("Status", style="yellow", justify="center")

        # Add aggregate score as first row
        if tv_result['aggregate_score'] is not None:
            tv_table.add_row("AGGREGATE", f"{tv_result['aggregate_score']:.3f}", "✓")
        else:
            tv_table.add_row("AGGREGATE", "n/a", "ℹ️")

        tv_table.add_section()

        # Show all columns from the original dataset
        all_columns = list(original_data.columns)
        column_scores = tv_result['column_scores']
        compatible_columns = tv_result['compatible_columns']

        for col in sorted(all_columns):
            if col in column_scores:
                tv_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
            elif col in compatible_columns:
                tv_table.add_row(col, "error", "⚠️")
            else:
                tv_table.add_row(col, "n/a", "—")

        # Add message at the bottom if no compatible columns found
        if tv_result.get("message"):
            tv_table.add_section()
            tv_table.add_row("INFO", tv_result["message"], "ℹ️")

        console.print(tv_table)
    else:
        console.print("❌ TVComplement failed", style="bold red")


    if "alpha_precision" in metrics and metrics["alpha_precision"]["status"] == "success":
        scores = metrics["alpha_precision"]["scores"]
        # Get parameters info for display
        params_info = metrics["alpha_precision"]["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        # Create Alpha Precision results table
        table = Table(title=f"✅ Alpha Precision Results ({params_display})", show_header=True, header_style="bold magenta")
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

    # PRDC Score results table
    if "prdc_score" in metrics and metrics["prdc_score"]["status"] == "success":
        prdc_result = metrics["prdc_score"]

        # Get parameters info for display
        params_info = prdc_result["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        # Create PRDC results table
        prdc_table = Table(title=f"✅ PRDC Score Results ({params_display})", show_header=True, header_style="bold blue")
        prdc_table.add_column("Metric", style="cyan", no_wrap=True)
        prdc_table.add_column("Score", style="bright_green", justify="right")

        prdc_table.add_row("Precision", f"{prdc_result['precision']:.3f}")
        prdc_table.add_row("Recall", f"{prdc_result['recall']:.3f}")
        prdc_table.add_row("Density", f"{prdc_result['density']:.3f}")
        prdc_table.add_row("Coverage", f"{prdc_result['coverage']:.3f}")

        console.print(prdc_table)
    else:
        console.print("❌ PRDC Score failed", style="bold red")

    console.print("\n✅ Statistical metrics evaluation completed", style="bold green")


if __name__ == "__main__":
    main()
