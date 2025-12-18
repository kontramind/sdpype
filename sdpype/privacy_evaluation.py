"""
Privacy metrics evaluation script for SDPype - DCR and other privacy metrics
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sdv.metadata import SingleTableMetadata
from sdpype.evaluation.statistical import evaluate_privacy_metrics, generate_privacy_report
from sdpype.encoding import load_encoding_config
from sdpype.metadata import load_csv_with_metadata

console = Console()


def _get_config_hash() -> str:
    """Get config hash from temporary file created during pipeline execution"""
    try:
        if Path('.sdpype_config_hash').exists():
            with open('.sdpype_config_hash', 'r', encoding='utf-8') as f:
                return f.read().strip()
        return "nohash"
    except Exception:
        return "nohash"


def _display_privacy_tables(results: dict) -> None:
    """Display privacy metrics results in rich tables"""

    console.print("\n🔒 Privacy Metrics Evaluation:", style="bold cyan")

    metrics_data = results.get("metrics", {})

    # Create privacy results table
    privacy_table = Table(
        title="📊 DCR Baseline Protection Results",
        show_header=True,
        header_style="bold blue"
    )
    privacy_table.add_column("Metric", style="cyan", no_wrap=True)
    privacy_table.add_column("Score", style="bright_green", justify="right")
    privacy_table.add_column("Interpretation", style="yellow")

    # Process DCR baseline protection results
    dcr_result = metrics_data.get("dcr_baseline_protection", {})

    if dcr_result.get("status") == "success":
        score = dcr_result.get("score", 0.0)
        median_dcr_synthetic = dcr_result.get("median_dcr_synthetic", 0.0)
        median_dcr_random = dcr_result.get("median_dcr_random", 0.0)

        # Interpret privacy score
        if score > 0.8:
            interpretation = "Excellent"
        elif score > 0.6:
            interpretation = "Good"
        elif score > 0.4:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"

        privacy_table.add_row(
            "Privacy Score",
            f"{score:.3f}",
            interpretation
        )
        privacy_table.add_row(
            "Median DCR (Synthetic)",
            f"{median_dcr_synthetic:.6f}",
            ""
        )
        privacy_table.add_row(
            "Median DCR (Random)",
            f"{median_dcr_random:.6f}",
            "Higher is better"
        )
    else:
        error_msg = dcr_result.get("error_message", "Unknown error")
        privacy_table.add_row(
            "DCR Baseline Protection",
            "N/A",
            f"❌ Error: {error_msg[:40]}..."
        )

    console.print(privacy_table)

    # k-Anonymization table
    k_anon_result = metrics_data.get("k_anonymization", {})

    if k_anon_result and k_anon_result.get("status") == "success":
        k_ratio = k_anon_result.get("k_ratio", 0.0)
        k_real = k_anon_result.get("k_real", 0.0)
        k_synthetic = k_anon_result.get("k_synthetic", 0.0)

        # Interpret k-anonymization score
        if k_ratio > 1.5:
            interpretation = "Excellent"
        elif k_ratio > 1.0:
            interpretation = "Good"
        elif k_ratio > 0.7:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"

        k_table = Table(
            title="📊 k-Anonymization Results",
            show_header=True,
            header_style="bold blue"
        )
        k_table.add_column("Metric", style="cyan", no_wrap=True)
        k_table.add_column("Value", style="bright_green", justify="right")
        k_table.add_column("Interpretation", style="yellow")

        k_table.add_row(
            "k-Anonymization Ratio",
            f"{k_ratio:.3f}",
            interpretation
        )
        k_table.add_row(
            "k (Real Data)",
            f"{k_real:.0f}",
            ""
        )
        k_table.add_row(
            "k (Synthetic Data)",
            f"{k_synthetic:.0f}",
            "Higher is better"
        )

        # Add distribution statistics if available
        dist_real = k_anon_result.get("distribution_real", {})
        dist_syn = k_anon_result.get("distribution_synthetic", {})

        if dist_real and dist_syn:
            k_table.add_section()
            k_table.add_row(
                "Mean Group Size (Real)",
                f"{dist_real.get('mean', 0):.1f}",
                ""
            )
            k_table.add_row(
                "Mean Group Size (Synthetic)",
                f"{dist_syn.get('mean', 0):.1f}",
                ""
            )
            k_table.add_row(
                "Num Groups (Real)",
                f"{dist_real.get('num_unique_groups', 0):,}",
                ""
            )
            k_table.add_row(
                "Num Groups (Synthetic)",
                f"{dist_syn.get('num_unique_groups', 0):,}",
                ""
            )

        console.print("\n")
        console.print(k_table)
    elif k_anon_result and k_anon_result.get("status") == "error":
        error_msg = k_anon_result.get("error_message", "Unknown error")
        console.print(f"\n[red]❌ k-Anonymization Error: {error_msg}[/red]")

    # Add interpretation guide
    guide_text = """🔒 Privacy Metrics Guide:

DCR Baseline Protection:
• Privacy Score = How well synthetic data protects individual record privacy
• Score > 0.8 = Excellent privacy protection
• Score > 0.6 = Good privacy protection
• Score > 0.4 = Moderate privacy protection
• Median DCR (Synthetic) = Distance from synthetic records to nearest real record
• Median DCR (Random) = Distance from random baseline to nearest real record
• Higher distances indicate better privacy protection"""

    # Add k-Anonymization guide if metric was computed
    if k_anon_result and k_anon_result.get("status") == "success":
        guide_text += """

k-Anonymization:
• k-Ratio = k_synthetic / k_real (comparison of anonymity levels)
• k-Ratio > 1.5 = Excellent (synthetic is much more private)
• k-Ratio > 1.0 = Good (synthetic is at least as private)
• k-Ratio > 0.7 = Moderate (synthetic is somewhat less private)
• k (Real/Synthetic) = Minimum group size in quasi-identifier groups
• Higher k values = Better privacy (records are hidden in larger groups)"""

    guide_panel = Panel.fit(
        guide_text,
        title="📖 Privacy Metrics Guide",
        border_style="blue"
    )
    console.print(guide_panel)


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run privacy metrics evaluation between reference and synthetic data"""

    config_hash = _get_config_hash()
    experiment_name = cfg.experiment.name
    seed = cfg.experiment.seed

    # Setup output paths
    metrics_dir = Path("experiments/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / f"privacy_{experiment_name}_{config_hash}_{seed}.json"
    report_file = metrics_dir / f"privacy_report_{experiment_name}_{config_hash}_{seed}.txt"

    # Check if privacy metrics are configured
    metrics_config = cfg.get("evaluation", {}).get("privacy", {}).get("metrics", [])

    if not metrics_config:
        print("🔒 No privacy metrics configured, skipping evaluation")

        # Create empty output files to satisfy DVC
        empty_results = {
            "metadata": {
                "experiment_name": f"{experiment_name}_seed_{seed}",
                "evaluation_timestamp": "",
                "original_shape": [0, 0],
                "synthetic_shape": [0, 0],
                "evaluation_type": "privacy_metrics",
                "status": "skipped",
                "message": "No privacy metrics configured in params.yaml"
            },
            "metrics": {}
        }

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(empty_results, f, indent=2, ensure_ascii=False)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Privacy Metrics Evaluation Report\n")
            f.write("===================================\n\n")
            f.write("Status: Skipped\n")
            f.write("Reason: No privacy metrics configured in params.yaml\n\n")
            f.write("To enable privacy metrics, add to params.yaml:\n")
            f.write("  evaluation:\n")
            f.write("    privacy:\n")
            f.write("      metrics:\n")
            f.write("        - name: dcr_baseline_protection\n")
            f.write("          parameters: {}\n")

        print(f"✅ Created placeholder files: {metrics_file.name}, {report_file.name}")
        return

    print(f"🔒 Starting privacy evaluation with {len(metrics_config)} metrics...")
    print(f"Experiment seed: {seed}")

    # Define which privacy metrics need encoded vs decoded data
    # All synthcity privacy metrics need encoded data (numerical format required)
    ENCODED_PRIVACY_METRICS = {'dcr_baseline_protection', 'k_anonymization'}
    DECODED_PRIVACY_METRICS = set()

    # Define data paths
    encoded_reference = Path(f"experiments/data/encoded/reference_{experiment_name}_{config_hash}_{seed}.csv")
    decoded_reference = Path(f"experiments/data/decoded/reference_{experiment_name}_{config_hash}_{seed}.csv")
    encoded_synthetic = Path(f"experiments/data/encoded/synthetic_{experiment_name}_{config_hash}_{seed}.csv")
    decoded_synthetic = Path(f"experiments/data/synthetic/synthetic_data_{experiment_name}_{config_hash}_{seed}_decoded.csv")

    metadata_file = Path(cfg.data.metadata_file)

    # Load metadata
    metadata = SingleTableMetadata.load_from_json(str(metadata_file))

    # Check if any metrics need encoding
    needs_encoding = any(m.get('name') in ENCODED_PRIVACY_METRICS for m in metrics_config)

    # Load data based on metric requirements
    if needs_encoding:
        print(f"📊 Loading encoded data for distance-based privacy metrics...")
        try:
            if not encoded_reference.exists():
                raise FileNotFoundError(f"Encoded reference file not found: {encoded_reference}")
            if not encoded_synthetic.exists():
                raise FileNotFoundError(f"Encoded synthetic file not found: {encoded_synthetic}")

            reference_encoded = pd.read_csv(encoded_reference)
            synthetic_encoded = pd.read_csv(encoded_synthetic)
            print(f"   ✓ Loaded encoded reference: {reference_encoded.shape}")
            print(f"   ✓ Loaded encoded synthetic: {synthetic_encoded.shape}")
        except Exception as e:
            print(f"❌ ERROR: Failed to load encoded data: {e}")
            print(f"   Metrics requiring encoded data (k_anonymization, dcr_baseline_protection) will fail.")
            print(f"   Please ensure the encode_evaluation DVC stage has been run.")
            reference_encoded = None
            synthetic_encoded = None
    else:
        reference_encoded = None
        synthetic_encoded = None

    # Always load decoded data
    print(f"📊 Loading decoded data...")
    reference_decoded = load_csv_with_metadata(decoded_reference, metadata_file)
    synthetic_decoded = load_csv_with_metadata(decoded_synthetic, metadata_file)

    # Load encoding config if available
    encoding_config = None
    if cfg.encoding.get('config_file'):
        encoding_config_path = Path(cfg.encoding.config_file)
        if encoding_config_path.exists():
            print(f"📊 Loading encoding config from {encoding_config_path}")
            encoding_config = load_encoding_config(encoding_config_path)

    # Run privacy metrics evaluation
    print(f"🔒 Computing privacy metrics...")
    results = evaluate_privacy_metrics(
        reference_encoded if reference_encoded is not None else reference_decoded,
        synthetic_encoded if synthetic_encoded is not None else synthetic_decoded,
        metrics_config,
        experiment_name=f"{experiment_name}_seed_{seed}",
        metadata=metadata,
        reference_data_decoded=reference_decoded,
        synthetic_data_decoded=synthetic_decoded,
        reference_data_encoded=reference_encoded,
        synthetic_data_encoded=synthetic_encoded,
        encoding_config=encoding_config
    )

    # Display results in rich tables
    _display_privacy_tables(results)

    # Save results (paths already defined at top of function)
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Privacy metrics saved to {metrics_file}")

    # Generate and save report
    report = generate_privacy_report(results)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ Privacy report saved to {report_file}")
    print(f"🔒 Privacy evaluation complete!")


if __name__ == "__main__":
    main()
