# Enhanced sdpype/training.py - Monolithic structure with experiment versioning
"""
Enhanced SDG training module for monolithic SDPype with experiment tracking
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

# Synthcity models  
from synthcity.plugins import Plugins


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
    """Train synthetic data generator with experiment versioning"""
    
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
    library = cfg.sdg.get("library", "sdv")  # Default to SDV for backward compatibility

    if library == "sdv":
        model = create_sdv_model(cfg, data)
        # Train SDV model
        start_time = time.time()
        model.fit(data)
        training_time = time.time() - start_time

    elif library == "synthcity":
        model = create_synthcity_model(cfg, data.shape)
        # Train Synthcity model
        start_time = time.time()
        model.fit(data)
        training_time = time.time() - start_time

    else:
        raise ValueError(f"Unknown library: {library}")

    print(f"⏱️  Training completed in {training_time:.1f}s")

    # Save model with experiment metadata (monolithic path + versioning)
    model_data = {
        "model": model,
        "library": library,
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
        "training_data_shape": data.shape,
        "training_time": training_time
    }

    # Use monolithic path + seed-specific filename
    Path("experiments/models").mkdir(parents=True, exist_ok=True)
    model_filename = f"experiments/models/sdg_model_{cfg.experiment.seed}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)

    print(f"📁 Model saved: {model_filename}")

    # Save detailed metrics (monolithic path + experiment versioning)
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
        "model_output": model_filename
    }

    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    metrics_filename = f"experiments/metrics/training_{cfg.experiment.seed}.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 Metrics saved: {metrics_filename}")
    print("✅ Model training completed")


if __name__ == "__main__":
    main()
