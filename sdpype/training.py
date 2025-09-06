# Enhanced sdpype/training.py - Unified model saving for SDV + Synthcity
"""
Enhanced SDG training module with unified model saving for all libraries
"""

import json
import pickle
import time
import hashlib
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

# SDV models
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

# Synthcity models and serialization
from synthcity.plugins import Plugins
from synthcity.utils.serialization import save as synthcity_save


def create_sdv_model(cfg: DictConfig, data: pd.DataFrame):
    """Create SDV model with metadata"""
    # SDV v1+ requires metadata
    from sdv.metadata import SingleTableMetadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    if cfg.sdg.model_type == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata)
    elif cfg.sdg.model_type == "ctgan":
        return CTGANSynthesizer(
            metadata,
            epochs=cfg.sdg.parameters.get("epochs", 300),
            batch_size=cfg.sdg.parameters.get("batch_size", 500),
            verbose=False
        )
    else:
        raise ValueError(f"Unknown SDV model: {cfg.sdg.model_type}")


def create_synthcity_model(cfg: DictConfig, data_shape):
    """Create Synthcity model"""
    model_name = cfg.sdg.model_type
    
    # Get model parameters from config
    model_params = dict(cfg.sdg.parameters) if cfg.sdg.parameters else {}
    
    # Create synthcity plugin
    try:
        model = Plugins().get(model_name, **model_params)
        return model
    except Exception as e:
        raise ValueError(f"Failed to create Synthcity model '{model_name}': {e}")


def save_model_unified(model, metadata: dict, library: str, experiment_seed: int):
    """Save any model to .pkl with library-specific internal handling"""

    Path("experiments/models").mkdir(parents=True, exist_ok=True)
    model_filename = f"experiments/models/sdg_model_{experiment_seed}.pkl"

    if library == "sdv":
        # SDV models can be stored directly
        model_data = {
            "model": model,        # Store model object directly
            "library": library,
            **metadata
        }

    elif library == "synthcity":
        # Synthcity models: serialize to bytes, then pack in .pkl
        model_bytes = synthcity_save(model)
        model_data = {
            "model_bytes": model_bytes,  # Store serialized bytes
            "library": library,
            **metadata
        }

    else:
        raise ValueError(f"Unknown library: {library}")

    # Always save as .pkl regardless of library
    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)

    print(f"📁 Model saved: {model_filename} ({library} format)")
    return model_filename


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
    data_file = f"experiments/data/processed/data_{cfg.experiment.seed}.csv"
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

    # Prepare metadata (without model object)
    metadata = {
        "model_type": cfg.sdg.model_type,
        "experiment": {
            "id": experiment_id,
            "seed": cfg.experiment.seed,
            "hash": experiment_hash,
            "timestamp": datetime.now().isoformat(),
            "name": cfg.experiment.name,
            "researcher": cfg.experiment.get("researcher", "anonymous")
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
        "training_data_shape": list(data.shape),  # Convert to list for JSON serialization
        "training_time": training_time,
        "training_data_columns": list(data.columns),
        "parameters": dict(cfg.sdg.parameters) if cfg.sdg.parameters else {}
    }

    # Save model using unified method
    model_filename = save_model_unified(
        model, metadata, library, cfg.experiment.seed
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
    metrics_filename = f"experiments/metrics/training_{cfg.experiment.seed}.json"
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
