# Enhanced sdpype/generation.py - Using new serialization module
"""
Enhanced synthetic data generation using centralized serialization
"""

import json
import time
from pathlib import Path
from datetime import datetime

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Import new serialization module
from sdpype.serialization import load_model


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
    """Generate synthetic data with unified model loading"""

    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.seed)

    print("🎯 Generating synthetic data...")
    config_hash = _get_config_hash()
    print(f"🎲 Experiment seed: {cfg.experiment.seed}")
    
    # Load model using unified method
    try:
        model, model_data = load_model(cfg.experiment.seed, cfg.experiment.name)
        library = model_data.get("library", "sdv")
        model_type = model_data.get("model_type", "unknown")
        experiment_info = model_data.get("experiment", {})

        print(f"📋 Loaded {library} {model_type} model")
        print(f"📋 Experiment: {experiment_info.get('id', 'unknown')}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Run training first: dvc repro -s train_sdg")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Model file may be corrupted or incompatible")
        raise

    # Generate synthetic data
    n_samples = cfg.generation.n_samples

    # Handle null n_samples by using original dataset size
    if n_samples is None:
        print("🔍 Auto-determining sample count from original dataset...")
        # Load processed data to get original sample count
        processed_data_file = f"experiments/data/processed/data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.csv"
        if not Path(processed_data_file).exists():
            print(f"❌ Cannot determine dataset size: {processed_data_file} not found")
            print("💡 Run preprocessing first: dvc repro -s preprocess")
            raise FileNotFoundError(f"Processed data file not found: {processed_data_file}")

        original_data = pd.read_csv(processed_data_file)
        n_samples = len(original_data)

        # Validate that we got a reasonable sample count
        if n_samples <= 0:
            raise ValueError(f"Invalid original dataset size: {n_samples} samples")
        if n_samples > 1_000_000:
            print(f"⚠️  Large dataset detected: {n_samples:,} samples")
            print("💡 Generation may take significant time and memory")

        print(f"📊 Using original dataset size: {n_samples} samples")
        print(f"🔄 Generating {n_samples:,} samples using {library} {model_type}...")
    else:
        # Validate explicit n_samples value
        if n_samples <= 0:
            raise ValueError(f"Invalid n_samples configuration: {n_samples}")
        print(f"🔄 Generating {n_samples} samples using {library} {model_type}...")

    start_time = time.time()

    try:
        if library == "sdv":
            synthetic_data = model.sample(n_samples)
        elif library == "synthcity":
            synthetic_data = model.generate(count=n_samples).dataframe()
        elif library == "synthpop":
            # Synthpop generation
            print(f"🔄 Generating {n_samples} samples using synthpop {model_type}...")
            synthetic_data = model.sample(n_samples)
        else:
            raise ValueError(f"Unknown library: {library}")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("💡 Check if model and generation parameters are compatible")
        raise

    generation_time = time.time() - start_time
    print(f"📊 Generated {len(synthetic_data)} samples in {generation_time:.1f}s")

    # Validate generated data
    if len(synthetic_data) == 0:
        raise ValueError("Generated dataset is empty")

    # More nuanced validation for auto-determined vs explicit sample counts
    if len(synthetic_data) != n_samples:
        if cfg.generation.n_samples is None:
            print(f"⚠️  Generated {len(synthetic_data)} samples instead of original dataset size {n_samples}")
            print("💡 This may indicate model generation issues with auto-sizing")
        else:
            print(f"⚠️  Generated {len(synthetic_data)} samples instead of requested {n_samples}")
            print("💡 This may indicate model parameter issues")

    # Save synthetic data (monolithic path + experiment versioning)
    Path("experiments/data/synthetic").mkdir(parents=True, exist_ok=True)
    synthetic_filename = f"experiments/data/synthetic/synthetic_data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.csv"
    synthetic_data.to_csv(synthetic_filename, index=False)
    print(f"📁 Synthetic data saved: {synthetic_filename}")
    
    # Save detailed metrics (monolithic path + experiment versioning)
    metrics = {
        "experiment_id": experiment_info.get("id", f"gen_{cfg.experiment.seed}"),
        "experiment_seed": cfg.experiment.seed,
        "config_hash": config_hash,
        "timestamp": datetime.now().isoformat(),
        "library": library,
        "model_type": model_type,
        "n_samples_config": cfg.generation.n_samples,  # Original config value (may be null)
        "n_samples_auto_determined": cfg.generation.n_samples is None,  # Flag for auto-sizing
        "samples_generated": len(synthetic_data),
        "samples_requested": n_samples,
        "columns": len(synthetic_data.columns),
        "column_names": list(synthetic_data.columns),
        "generation_time": generation_time,
        "samples_per_second": len(synthetic_data) / generation_time if generation_time > 0 else 0,
        "output_file": synthetic_filename,
        "model_source": f"experiments/models/sdg_model_{cfg.experiment.name}_{cfg.experiment.seed}.pkl"
    }

    # Add basic data quality metrics
    numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metrics.update({
            "numeric_columns": len(numeric_cols),
            "numeric_column_names": list(numeric_cols),
            "mean_values": {col: float(synthetic_data[col].mean()) for col in numeric_cols},
            "std_values": {col: float(synthetic_data[col].std()) for col in numeric_cols},
            "min_values": {col: float(synthetic_data[col].min()) for col in numeric_cols},
            "max_values": {col: float(synthetic_data[col].max()) for col in numeric_cols}
        })

    # Add categorical data info
    categorical_cols = synthetic_data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        metrics.update({
            "categorical_columns": len(categorical_cols),
            "categorical_column_names": list(categorical_cols),
            "unique_values_per_column": {
                col: int(synthetic_data[col].nunique()) for col in categorical_cols
            }
        })

    # Check for missing values
    missing_values = synthetic_data.isnull().sum()
    if missing_values.sum() > 0:
        metrics["missing_values"] = {
            "total_missing": int(missing_values.sum()),
            "missing_by_column": {col: int(count) for col, count in missing_values.items() if count > 0}
        }
        print(f"⚠️  Generated data contains {missing_values.sum()} missing values")

    # Save metrics
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    metrics_filename = f"experiments/metrics/generation_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 Metrics saved: {metrics_filename}")
    print("✅ Synthetic data generation completed")

    # Print summary
    print(f"\n📈 Generation Summary:")
    print(f"  Library: {library}")
    print(f"  Model: {model_type}")
    if cfg.generation.n_samples is None:
        print(f"  Samples: {len(synthetic_data):,} (auto-matched to original dataset)")
    else:
        print(f"  Samples: {len(synthetic_data):,} (requested: {n_samples:,})")
    print(f"  Columns: {len(synthetic_data.columns)}")
    print(f"  Time: {generation_time:.1f}s")
    print(f"  Speed: {len(synthetic_data)/generation_time:.0f} samples/sec")


if __name__ == "__main__":
    main()
