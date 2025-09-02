# Enhanced sdpype/generation.py - Monolithic structure with experiment versioning
"""
Enhanced synthetic data generation for monolithic SDPype with experiment tracking
"""

import json
import pickle
import time
from pathlib import Path
from datetime import datetime

import hydra
import numpy as np
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate synthetic data with experiment versioning"""

    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.seed)

    print("🎯 Generating synthetic data...")
    print(f"🎲 Experiment seed: {cfg.experiment.seed}")
    
    # Load trained model (monolithic path + seed-specific)
    model_filename = f"experiments/models/sdg_model.pkl"

    if not Path(model_filename).exists():
        print(f"❌ Model file not found: {model_filename}")
        print("💡 Run training first: dvc repro -s train_sdg")
        raise FileNotFoundError(f"Model not found: {model_filename}")
    
    with open(model_filename, "rb") as f:
        model_data = pickle.load(f)

    # Extract model and experiment info
    if isinstance(model_data, dict):
        model = model_data["model"]
        library = model_data["library"]
        model_type = model_data["model_type"]
        experiment_info = model_data.get("experiment", {})
        print(f"📋 Loaded experiment: {experiment_info.get('id', 'unknown')}")
    else:
        # Backward compatibility - assume SDV
        model = model_data
        library = "sdv"
        model_type = "unknown"
        experiment_info = {}
        print("⚠️  Loading legacy model format")

    # Generate synthetic data
    n_samples = cfg.generation.n_samples
    print(f"🔄 Generating {n_samples} samples using {library} {model_type}...")

    start_time = time.time()

    if library == "sdv":
        synthetic_data = model.sample(n_samples)
    elif library == "synthcity":
        synthetic_data = model.generate(count=n_samples).dataframe()
    else:
        raise ValueError(f"Unknown library: {library}")

    generation_time = time.time() - start_time

    print(f"📊 Generated {len(synthetic_data)} samples in {generation_time:.1f}s")

    # Save synthetic data (monolithic path + experiment versioning)
    Path("experiments/data/synthetic").mkdir(parents=True, exist_ok=True)
    synthetic_filename = f"experiments/data/synthetic/synthetic_data.csv"
    synthetic_data.to_csv(synthetic_filename, index=False)
    print(f"📁 Synthetic data saved: {synthetic_filename}")
    
    # Save detailed metrics (monolithic path + experiment versioning)
    metrics = {
        "experiment_id": experiment_info.get("id", f"gen_{cfg.experiment.seed}"),
        "experiment_seed": cfg.experiment.seed,
        "timestamp": datetime.now().isoformat(),
        "library": library,
        "model_type": model_type,
        "samples_generated": len(synthetic_data),
        "samples_requested": n_samples,
        "columns": len(synthetic_data.columns),
        "generation_time": generation_time,
        "samples_per_second": len(synthetic_data) / generation_time if generation_time > 0 else 0,
        "model_source": model_filename,
        "output_file": synthetic_filename
    }

    # Add basic data quality metrics
    numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metrics.update({
            "numeric_columns": len(numeric_cols),
            "mean_values": synthetic_data[numeric_cols].mean().to_dict(),
            "std_values": synthetic_data[numeric_cols].std().to_dict()
        })

    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    metrics_filename = f"experiments/metrics/generation.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 Metrics saved: {metrics_filename}")
    print("✅ Synthetic data generation completed")


if __name__ == "__main__":
    main()
