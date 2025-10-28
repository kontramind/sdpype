"""
Training data encoding for SDPype pipeline.

Fits encoder on TRAINING data (synthetic Gen N) for training next model (Gen N+1).
This prevents reference data leakage into the training encoding.

CLI Usage:
    python -m sdpype.encode_training
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
    Encode training data for next model generation.

    Fits encoder on training data (synthetic Gen N) to avoid reference leakage.
    Outputs encoded/decoded training data and fitted encoder for model training.
    """
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"🔧 TRAINING DATA ENCODING")
    print(f"{'='*70}")
    print(f"Purpose: Encode training data for next model generation")
    print(f"Fits on: TRAINING data (avoids reference leakage)")

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

    # 2. Load training dataset
    print(f"\n📊 Loading training dataset...")
    training_file = Path(cfg.data.training_file)

    if not training_file.exists():
        raise FileNotFoundError(f"Training file not found: {training_file}")

    training_data = pd.read_csv(training_file)
    print(f"✓ Training data: {training_data.shape}")

    # 3. Create and fit encoder on TRAINING data
    print(f"\n🔧 Fitting encoders on TRAINING data...")
    encoder = RDTDatasetEncoder(config)
    encoder.fit(training_data)

    # 4. Transform training data
    print(f"\n🔄 Transforming training data...")
    encoded_training = encoder.transform(training_data)
    print(f"✓ Encoded to {encoded_training.shape[1]} columns")

    # 5. Reverse transform for dual pipeline (decoded version)
    print(f"\n🔄 Creating decoded version for dual pipeline...")
    decoded_training = encoder.reverse_transform(encoded_training)

    # 6. Save outputs
    print(f"\n💾 Saving outputs...")

    # Create output directories
    encoded_dir = Path(f"experiments/data/encoded")
    decoded_dir = Path(f"experiments/data/decoded")
    models_dir = Path(f"experiments/models")
    metrics_dir = Path(f"experiments/metrics")

    encoded_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Generate filenames
    base_name = f"{experiment_name}_{config_hash}_{seed}"

    # Save encoded training data (for synthcity)
    encoded_training_path = encoded_dir / f"training_{base_name}.csv"
    encoded_training.to_csv(encoded_training_path, index=False)
    print(f"✓ Encoded training → {encoded_training_path}")

    # Save decoded training data (for SDV with injected transformer)
    decoded_training_path = decoded_dir / f"training_{base_name}.csv"
    decoded_training.to_csv(decoded_training_path, index=False)
    print(f"✓ Decoded training → {decoded_training_path}")

    # Save fitted TRAINING encoder
    encoder_path = models_dir / f"training_encoder_{base_name}.pkl"
    encoder.save(encoder_path)
    print(f"✓ Training encoder → {encoder_path}")

    # 7. Record metrics
    elapsed_time = time.time() - start_time

    metrics = {
        "encoding_type": "training",
        "encoding_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "experiment": {
            "name": experiment_name,
            "seed": seed,
            "config_hash": config_hash,
        },
        "config_file": str(encoding_config_path),
        "fitted_on": "training_data",
        "input_shapes": {
            "training": list(training_data.shape),
        },
        "output_shapes": {
            "encoded_training": list(encoded_training.shape),
            "decoded_training": list(decoded_training.shape),
        },
        "transformers": {
            col: type(trans).__name__
            for col, trans in encoder.transformers.items()
        },
        "encoding_time_seconds": round(elapsed_time, 2),
        "outputs": {
            "encoded_training": str(encoded_training_path),
            "decoded_training": str(decoded_training_path),
            "training_encoder": str(encoder_path),
        }
    }

    metrics_path = metrics_dir / f"encoding_training_{base_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics → {metrics_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ TRAINING ENCODING COMPLETE!")
    print(f"{'='*70}")
    print(f"⏱️  Time: {elapsed_time:.2f}s")
    print(f"📊 Columns: {training_data.shape[1]} → {encoded_training.shape[1]}")
    print(f"🎯 Purpose: Training next model generation (no reference leakage)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
