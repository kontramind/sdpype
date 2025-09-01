# Updated sdpype/generation.py - Support both libraries
"""
Enhanced synthetic data generation supporting SDV and Synthcity
"""

import json
import pickle
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate synthetic data"""
    
    print("🎯 Generating synthetic data...")
    
    # Load trained model
    with open("models/sdg_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    # Extract model and library info
    if isinstance(model_data, dict):
        model = model_data["model"]
        library = model_data["library"]
        model_type = model_data["model_type"]
    else:
        # Backward compatibility - assume SDV
        model = model_data
        library = "sdv"
        model_type = "unknown"
    
    # Generate synthetic data
    n_samples = cfg.generation.n_samples
    print(f"Generating {n_samples} samples using {library} {model_type}...")
    
    start_time = time.time()
    
    if library == "sdv":
        synthetic_data = model.sample(n_samples)
    elif library == "synthcity":
        synthetic_data = model.generate(count=n_samples).dataframe()
    else:
        raise ValueError(f"Unknown library: {library}")
    
    generation_time = time.time() - start_time
    
    print(f"Generated {len(synthetic_data)} samples in {generation_time:.1f}s")
    
    # Save synthetic data
    Path("data/synthetic").mkdir(exist_ok=True)
    synthetic_data.to_csv("data/synthetic/synthetic_data.csv", index=False)
    
    # Save metrics
    metrics = {
        "library": library,
        "model_type": model_type,
        "samples_generated": len(synthetic_data),
        "columns": len(synthetic_data.columns),
        "generation_time": generation_time,
        "samples_per_second": len(synthetic_data) / generation_time if generation_time > 0 else 0
    }
    
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/generation.json", "w") as f:
        json.dump(metrics, f)
    
    print("✅ Synthetic data generation completed")


if __name__ == "__main__":
    main()
