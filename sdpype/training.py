# Enhanced sdpype/training.py - Using new serialization module
"""
Enhanced SDG training module using centralized serialization
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

# SDV models
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer

# Synthcity models and serialization
from synthcity.plugins import Plugins

# Import new serialization module
from sdpype.serialization import save_model, create_model_metadata


def create_sdv_model(cfg: DictConfig, data: pd.DataFrame):
    """Create SDV model with metadata"""
    # SDV v1+ requires metadata
    from sdv.metadata import SingleTableMetadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    # Get model parameters from config, filtering for SDV-compatible params  
    all_params = dict(cfg.sdg.parameters) if cfg.sdg.parameters else {}
    
    # Filter out Synthcity-specific parameters that SDV doesn't understand
    synthcity_only_params = {'n_iter'}  # Add more Synthcity-only params as needed
    model_params = {k: v for k, v in all_params.items() if k not in synthcity_only_params}

    match cfg.sdg.model_type:
        case "gaussian_copula":
            return GaussianCopulaSynthesizer(metadata)

        case "ctgan" | "tvae" as model_type:
            # Common parameters for GAN-based models
            synthesizer_class = {
                "ctgan": CTGANSynthesizer,
                "tvae": TVAESynthesizer
            }[model_type]

            return synthesizer_class(
                metadata,
                epochs=model_params.get("epochs", 300),
                batch_size=model_params.get("batch_size", 500),
                verbose=False
            )

        case _:
            raise ValueError(f"Unknown SDV model: {cfg.sdg.model_type}")


def create_synthcity_model(cfg: DictConfig, data_shape):
    """Create Synthcity model"""
    model_name = cfg.sdg.model_type
    
    # Get model parameters from config, filtering for Synthcity-compatible params
    all_params = dict(cfg.sdg.parameters) if cfg.sdg.parameters else {}
    
    # Filter out SDV-specific parameters that Synthcity doesn't understand
    sdv_only_params = {'epochs', 'verbose'}  # Add more SDV-only params as needed
    model_params = {k: v for k, v in all_params.items() if k not in sdv_only_params}
    
    # Create synthcity plugin
    try:
        model = Plugins().get(model_name, **model_params)
        return model
    except Exception as e:
        raise ValueError(f"Failed to create Synthcity model '{model_name}': {e}")


def create_experiment_hash(cfg: DictConfig) -> str:
    """Create unique hash for experiment configuration"""
    # Include key config elements in hash
    hash_dict = {
        "sdg": OmegaConf.to_container(cfg.sdg, resolve=True),
        "preprocessing": OmegaConf.to_container(cfg.preprocessing, resolve=True),
        "seed": cfg.experiment.seed,
        "data_file": cfg.data.input_file
    }

    hash_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Train synthetic data generator with unified .pkl output"""
    
    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.seed)
    
    # Create experiment metadata
    experiment_hash = create_experiment_hash(cfg)
    experiment_id = f"{cfg.sdg.model_type}_{cfg.experiment.seed}_{experiment_hash}"

    print(f"🚀 Training {cfg.sdg.model_type} model (library: {cfg.sdg.library})")
    print(f"📋 Experiment: {experiment_id}")
    print(f"🎲 Seed: {cfg.experiment.seed}")

    # Load processed data (monolithic path + experiment versioning)
    data_file = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
    if not Path(data_file).exists():
        print(f"❌ Processed data not found: {data_file}")
        print("💡 Run preprocessing first: dvc repro -s preprocess")
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data = pd.read_csv(data_file)
    print(f"📊 Training data: {data.shape}")

    # Create model based on library
    library = cfg.sdg.library

    if library == "sdv":
        print(f"🔧 Creating SDV {cfg.sdg.model_type} model...")
        model = create_sdv_model(cfg, data)
    elif library == "synthcity":
        print(f"🔧 Creating Synthcity {cfg.sdg.model_type} model...")
        model = create_synthcity_model(cfg, data.shape)
    else:
        raise ValueError(f"Unknown library: {library}")

    # Train model
    print(f"⏳ Training {library} {cfg.sdg.model_type} model...")
    start_time = time.time()
    model.fit(data)
    training_time = time.time() - start_time

    print(f"⏱️  Training completed in {training_time:.1f}s")

    # Create standardized metadata using serialization module
    metadata = create_model_metadata(
        cfg, cfg.sdg.model_type, library, cfg.experiment.seed,
        training_time, data, experiment_id, experiment_hash
    )

    # Save model using new serialization module
    model_filename = save_model(
        model, metadata, library, cfg.experiment.seed, cfg.experiment.name
    )

    # Save training metrics (same as before)
    metrics = {
        "experiment_id": experiment_id,
        "experiment_hash": experiment_hash,
        "seed": cfg.experiment.seed,
        "library": library,
        "model_type": cfg.sdg.model_type,
        "training_time": training_time,
        "training_rows": len(data),
        "training_columns": len(data.columns),
        "timestamp": datetime.now().isoformat(),
        "data_source": data_file,
        "model_output": model_filename,
        "model_parameters": dict(cfg.sdg.parameters) if cfg.sdg.parameters else {}
    }

    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    metrics_filename = f"experiments/metrics/training_{cfg.experiment.name}_{cfg.experiment.seed}.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 Metrics saved: {metrics_filename}")
    print("✅ Model training completed")

    # Print training summary
    print(f"\n📈 Training Summary:")
    print(f"  Library: {library}")
    print(f"  Model: {cfg.sdg.model_type}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Training data: {len(data):,} rows × {len(data.columns)} columns")
    print(f"  Model file: {model_filename}")


if __name__ == "__main__":
    main()
