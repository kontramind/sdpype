"""
Evaluation data encoding for SDPype pipeline.

Fits encoder on REFERENCE data (real) for fair evaluation comparison.
Transforms both reference and synthetic data using the same encoder.

CLI Usage:
    python -m sdpype.encode_evaluation
"""

import logging
from pathlib import Path
import json
import time
from datetime import datetime

import pandas as pd
import hydra
from omegaconf import DictConfig

from sdpype.encoding import RDTDatasetEncoder, load_encoding_config
from sdpype.metadata import load_csv_with_metadata

logger = logging.getLogger(__name__)


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
    """
    Encode reference and synthetic data for evaluation metrics.

    Fits encoder on REFERENCE data (real) to capture all categories.
    Transforms both datasets for fair comparison in metrics/detection.
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"🔧 EVALUATION DATA ENCODING")
    print(f"{'='*70}")
    print(f"Purpose: Encode data for evaluation metrics and detection")
    print(f"Fits on: REFERENCE data (captures all real categories)")

    # Get experiment metadata
    seed = cfg.experiment.seed
    config_hash = _get_config_hash()
    experiment_name = cfg.experiment.name

    # 1. Load encoding configuration
    encoding_config_path = Path(cfg.encoding.config_file)
    if not encoding_config_path.exists():
        raise FileNotFoundError(f"Encoding config not found: {encoding_config_path}")

    print(f"\n📋 Loading encoding config: {encoding_config_path}")
    config = load_encoding_config(encoding_config_path)
    print(f"✓ Config loaded - {len(config['sdtypes'])} columns")

    # 2. Load datasets
    print(f"\n📊 Loading datasets...")
    reference_file = Path(cfg.data.reference_file)
    metadata_path = Path(cfg.data.metadata_file)

    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Find synthetic file - check both decoded and original locations
    base_name = f"{experiment_name}_{config_hash}_{seed}"
    synthetic_decoded_path = Path(f"experiments/data/synthetic/synthetic_data_{base_name}_decoded.csv")

    # Fallback to checking data.synthetic_file if configured
    if not synthetic_decoded_path.exists() and hasattr(cfg.data, 'synthetic_file'):
        synthetic_file = Path(cfg.data.synthetic_file)
        if synthetic_file.exists():
            synthetic_decoded_path = synthetic_file

    if not synthetic_decoded_path.exists():
        raise FileNotFoundError(
            f"Synthetic decoded data not found: {synthetic_decoded_path}\n"
            f"Make sure generation stage completed successfully."
        )

    # Use metadata-based loading for type consistency
    reference_data = load_csv_with_metadata(reference_file, metadata_path)
    synthetic_data = load_csv_with_metadata(synthetic_decoded_path, metadata_path)

    print(f"\n📊 Loading datasets...")
    print(f"✓ Reference data: {reference_data.shape}")
    print(f"✓ Synthetic data: {synthetic_data.shape}")

    # 3. Create and fit encoder on REFERENCE data
    print(f"\n🔧 Fitting encoders on REFERENCE data...")
    print(f"   (This captures ALL real categories - no 'new category' warnings!)")
    encoder = RDTDatasetEncoder(config)
    encoder.fit(reference_data)

    # 4. Transform both datasets
    print(f"\n🔄 Transforming datasets...")
    print(f"   Transforming reference data...")
    encoded_reference = encoder.transform(reference_data)
    print(f"   ✓ Transformed to {encoded_reference.shape[1]} columns")

    print(f"   Transforming synthetic data...")
    encoded_synthetic = encoder.transform(synthetic_data)
    print(f"   ✓ Transformed to {encoded_synthetic.shape[1]} columns")

    # 5. Reverse transform for dual pipeline (decoded versions)
    print(f"\n🔄 Creating decoded versions for dual pipeline...")
    decoded_reference = encoder.reverse_transform(encoded_reference)
    decoded_synthetic = encoder.reverse_transform(encoded_synthetic)

    # 6. Save outputs
    print(f"\n💾 Saving outputs...")

    # Create output directories
    encoded_dir = Path(f"experiments/data/encoded")
    decoded_dir = Path(f"experiments/data/decoded")
    synthetic_dir = Path(f"experiments/data/synthetic")
    models_dir = Path(f"experiments/models")
    metrics_dir = Path(f"experiments/metrics")

    encoded_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Generate filenames
    base_name = f"{experiment_name}_{config_hash}_{seed}"

    # Save encoded data
    encoded_reference_path = encoded_dir / f"reference_{base_name}.csv"
    encoded_reference.to_csv(encoded_reference_path, index=False)
    print(f"✓ Encoded reference → {encoded_reference_path}")

    # Save encoded synthetic (for detection + encoded metrics)
    encoded_synthetic_path = encoded_dir / f"synthetic_{base_name}.csv"
    encoded_synthetic.to_csv(encoded_synthetic_path, index=False)
    print(f"✓ Encoded synthetic → {encoded_synthetic_path}")

    # Save decoded data
    decoded_reference_path = decoded_dir / f"reference_{base_name}.csv"
    decoded_reference.to_csv(decoded_reference_path, index=False)
    print(f"✓ Decoded reference → {decoded_reference_path}")

    decoded_synthetic_path = decoded_dir / f"synthetic_{base_name}_decoded.csv"
    decoded_synthetic.to_csv(decoded_synthetic_path, index=False)
    print(f"✓ Decoded synthetic → {decoded_synthetic_path}")

    # Save fitted EVALUATION encoder
    encoder_path = models_dir / f"evaluation_encoder_{base_name}.pkl"
    encoder.save(encoder_path)
    print(f"✓ Evaluation encoder → {encoder_path}")

    # 7. Record metrics
    elapsed_time = time.time() - start_time

    metrics = {
        "encoding_type": "evaluation",
        "encoding_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "experiment": {
            "name": experiment_name,
            "seed": seed,
            "config_hash": config_hash,
        },
        "config_file": str(encoding_config_path),
        "fitted_on": "reference_data",
        "input_shapes": {
            "reference": list(reference_data.shape),
            "synthetic": list(synthetic_data.shape),
        },
        "output_shapes": {
            "encoded_reference": list(encoded_reference.shape),
            "encoded_synthetic": list(encoded_synthetic.shape),
            "decoded_reference": list(decoded_reference.shape),
            "decoded_synthetic": list(decoded_synthetic.shape),
        },
        "transformers": {
            col: type(trans).__name__
            for col, trans in encoder.transformers.items()
        },
        "encoding_time_seconds": round(elapsed_time, 2),
        "outputs": {
            "encoded_reference": str(encoded_reference_path),
            "encoded_synthetic": str(encoded_synthetic_path),
            "decoded_reference": str(decoded_reference_path),
            "decoded_synthetic": str(decoded_synthetic_path),
            "evaluation_encoder": str(encoder_path),
        }
    }

    metrics_path = metrics_dir / f"encoding_evaluation_{base_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics → {metrics_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ EVALUATION ENCODING COMPLETE!")
    print(f"{'='*70}")
    print(f"⏱️  Time: {elapsed_time:.2f}s")
    print(f"📊 Columns: {reference_data.shape[1]} → {encoded_reference.shape[1]}")
    print(f"🎯 Purpose: Fair evaluation (fitted on real data)")
    print(f"💡 No 'new category' warnings - all real categories captured!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
