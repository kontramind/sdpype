"""
Privacy metrics evaluation script for SDPype - DCR and other privacy metrics
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console

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


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run privacy metrics evaluation between reference and synthetic data"""

    # Check if privacy metrics are configured
    metrics_config = cfg.get("evaluation", {}).get("privacy", {}).get("metrics", [])

    if not metrics_config:
        print("🔒 No privacy metrics configured, skipping evaluation")
        return

    print(f"🔒 Starting privacy evaluation with {len(metrics_config)} metrics...")
    print(f"Experiment seed: {cfg.experiment.seed}")

    config_hash = _get_config_hash()

    # Define which privacy metrics need encoded vs decoded data
    # DCR and other distance-based privacy metrics need encoded data
    ENCODED_PRIVACY_METRICS = {'dcr_baseline_protection'}
    DECODED_PRIVACY_METRICS = set()

    # Build file paths
    experiment_name = cfg.experiment.name
    seed = cfg.experiment.seed

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
        reference_encoded = pd.read_csv(encoded_reference)
        synthetic_encoded = pd.read_csv(encoded_synthetic)
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

    # Save results
    metrics_dir = Path("experiments/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / f"privacy_{experiment_name}_{config_hash}_{seed}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Privacy metrics saved to {metrics_file}")

    # Generate and save report
    report_file = metrics_dir / f"privacy_report_{experiment_name}_{config_hash}_{seed}.txt"
    report = generate_privacy_report(results)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ Privacy report saved to {report_file}")
    print(f"🔒 Privacy evaluation complete!")


if __name__ == "__main__":
    main()
