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
from sdv.metadata import SingleTableMetadata

# SDV models
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer

# Synthcity models and serialization
from synthcity.plugins import Plugins

# Import new serialization module
from sdpype.serialization import save_model, create_model_metadata


def create_sdv_model(cfg: DictConfig, metadata: SingleTableMetadata):
    """Create SDV model with metadata"""
    # SDV v1+ requires metadata
    # Get model parameters from config, filtering for SDV-compatible params  
    # Use OmegaConf.to_container to properly convert nested structures
    all_params = OmegaConf.to_container(cfg.sdg.parameters, resolve=True) if cfg.sdg.parameters else {}
    
    # Filter out Synthcity-specific parameters that SDV doesn't understand
    synthcity_only_params = {'n_iter'}  # Add more Synthcity-only params as needed
    model_params = {k: v for k, v in all_params.items() if k not in synthcity_only_params}

    match cfg.sdg.model_type:
        case "gaussian_copula":
            return GaussianCopulaSynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                locales=model_params.get("locales", ['en_US']),
                default_distribution=model_params.get("default_distribution", 'beta'),
                numerical_distributions=model_params.get("numerical_distributions", {})
            )
        case "ctgan":
            return CTGANSynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                locales=model_params.get("locales", ['en_US']),
                epochs=model_params.get("epochs", 300),
                batch_size=model_params.get("batch_size", 500),
                verbose=model_params.get("verbose", False),
                cuda=model_params.get("cuda", True),
                discriminator_dim=model_params.get("discriminator_dim", (256, 256)),
                discriminator_decay=model_params.get("discriminator_decay", 1e-06),
                discriminator_lr=model_params.get("discriminator_lr", 0.0002),
                discriminator_steps=model_params.get("discriminator_steps", 1),
                generator_lr=model_params.get("generator_lr", 0.0002),
                embedding_dim=model_params.get("embedding_dim", 128),
                generator_decay=model_params.get("generator_decay", 1e-06),
                generator_dim=model_params.get("generator_dim", (256, 256)),
                log_frequency=model_params.get("log_frequency", True),
                pac=model_params.get("pac", 10)
            )

        case "tvae":
            return TVAESynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                epochs=model_params.get("epochs", 300),
                batpch_size=model_params.get("batch_size", 500),
                verbose=model_params.get("verbose", False),
                cuda=model_params.get("cuda", True),
                compress_dims=model_params.get("compress_dims", (128, 128)),
                decompress_dims=model_params.get("decompress_dims", (128, 128)),
                embedding_dim=model_params.get("embedding_dim", 128),
                l2scale=model_params.get("l2scale", 1e-5),
                loss_factor=model_params.get("loss_factor", 2)
            )

        case "copula_gan":
            return CopulaGANSynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                locales=model_params.get("locales", ['en_US']),
                default_distribution=model_params.get("default_distribution", 'beta'),
                numerical_distributions=model_params.get("numerical_distributions", {}),
                epochs=model_params.get("epochs", 300),
                batch_size=model_params.get("batch_size", 500),
                verbose=model_params.get("verbose", False),
                cuda=model_params.get("cuda", True),
                discriminator_dim=model_params.get("discriminator_dim", (256, 256)),
                discriminator_decay=model_params.get("discriminator_decay", 1e-06),
                discriminator_lr=model_params.get("discriminator_lr", 0.0002),
                discriminator_steps=model_params.get("discriminator_steps", 1),
                generator_lr=model_params.get("generator_lr", 0.0002),
                embedding_dim=model_params.get("embedding_dim", 128),
                generator_decay=model_params.get("generator_decay", 1e-06),
                generator_dim=model_params.get("generator_dim", (256, 256)),
                log_frequency=model_params.get("log_frequency", True),
                pac=model_params.get("pac", 10)
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
        print("💡 Run preprocessing first!")
        raise FileNotFoundError(f"Data file not found: {data_file}")

    metadata_file = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}_metadata.json"
    if not Path(metadata_file).exists():
        print(f"❌ Metadata for processed data not found: {metadata_file}")
        print("💡 Run metadata creation first!")
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    data = pd.read_csv(data_file)
    print(f"📊 Training data: {data.shape}")

    metadata = SingleTableMetadata.load_from_json(metadata_file)

    # Create model based on library
    library = cfg.sdg.library

    if library == "sdv":
        print(f"🔧 Creating SDV {cfg.sdg.model_type} model...")
        model = create_sdv_model(cfg, metadata)
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
        "model_parameters": OmegaConf.to_container(cfg.sdg.parameters, resolve=True) if cfg.sdg.parameters else {}
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
