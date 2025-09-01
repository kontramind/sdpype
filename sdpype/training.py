# Updated sdpype/training.py - Support both SDV and Synthcity
"""
Enhanced SDG training module supporting SDV and Synthcity models
"""

import json
import pickle
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# SDV models
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

# Synthcity models  
from synthcity.plugins import Plugins


def create_sdv_model(cfg: DictConfig):
    """Create SDV model"""
    if cfg.sdg.model_type == "gaussian_copula":
        return GaussianCopulaSynthesizer()
    elif cfg.sdg.model_type == "ctgan":
        return CTGANSynthesizer(
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train synthetic data generator"""
    
    print(f"🚀 Training {cfg.sdg.model_type} model (library: {cfg.sdg.library})...")
    
    # Load processed data
    data = pd.read_csv("data/processed/data.csv")
    print(f"Training data: {data.shape}")
    
    # Create model based on library
    library = cfg.sdg.get("library", "sdv")  # Default to SDV for backward compatibility
    
    if library == "sdv":
        model = create_sdv_model(cfg)
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
    
    print(f"Training completed in {training_time:.1f}s")
    
    # Save model with library info
    model_data = {
        "model": model,
        "library": library,
        "model_type": cfg.sdg.model_type,
        "config": dict(cfg.sdg)
    }
    
    Path("models").mkdir(exist_ok=True)
    with open("models/sdg_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # Save metrics
    metrics = {
        "library": library,
        "model_type": cfg.sdg.model_type,
        "training_time": training_time,
        "training_rows": len(data),
        "training_columns": len(data.columns)
    }
    
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/training.json", "w") as f:
        json.dump(metrics, f)
    
    print("✅ Model training completed")


if __name__ == "__main__":
    main()
