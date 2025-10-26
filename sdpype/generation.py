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

# Import encoding modules for dual pipeline
from sdpype.encoding import RDTDatasetEncoder
from sdpype.label_encoding import SimpleLabelEncoder


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

    # Set all random seeds for reproducibility across libraries
    np.random.seed(cfg.experiment.seed)
    
    # Set PyTorch seed (SDV models use PyTorch)
    try:
        import torch
        torch.manual_seed(cfg.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.experiment.seed)
    except ImportError:
        pass
    
    # Set Python's built-in random seed
    import random
    random.seed(cfg.experiment.seed)

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

    # Load encoder (fitted HyperTransformer) for dual pipeline
    encoder_path = Path(f"experiments/models/encoders_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.pkl")
    if not encoder_path.exists():
        print(f"❌ Encoder not found: {encoder_path}")
        print("💡 Run encoding stage first: dvc repro -s encode_dataset")
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    print(f"📦 Loading fitted encoder: {encoder_path}")
    encoder = RDTDatasetEncoder.load(encoder_path)

    # Generate synthetic data
    n_samples = cfg.generation.n_samples

    # Handle null n_samples by using original dataset size
    if n_samples is None:
        print("🔍 Auto-determining sample count from reference dataset...")
        # Load reference data to get sample count
        reference_data_file = cfg.data.reference_file
        if not Path(reference_data_file).exists():
            print(f"❌ Cannot determine dataset size: {reference_data_file} not found")
            print("💡 Check your data.reference_file path in params.yaml!")
            raise FileNotFoundError(f"Reference data file not found: {reference_data_file}")

        reference_data = pd.read_csv(reference_data_file)
        n_samples = len(reference_data)

        # Validate that we got a reasonable sample count
        if n_samples <= 0:
            raise ValueError(f"Invalid original dataset size: {n_samples} samples")
        if n_samples > 1_000_000:
            print(f"⚠️  Large dataset detected: {n_samples:,} samples")
            print("💡 Generation may take significant time and memory")

        print(f"📊 Using reference dataset size: {n_samples} samples")
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
            print(f"⚠️  Generated {len(synthetic_data)} samples instead of reference dataset size {n_samples}")
            print("💡 This may indicate model generation issues with auto-sizing")
        else:
            print(f"⚠️  Generated {len(synthetic_data)} samples instead of requested {n_samples}")
            print("💡 This may indicate model parameter issues")

    # Dual pipeline: Create both encoded and decoded versions
    print(f"\n🔄 Creating dual pipeline outputs...")

    if library == "sdv":
        # SDV outputs decoded (native sdtypes) data
        synthetic_decoded = synthetic_data
        print(f"📊 SDV output (decoded/native): {synthetic_decoded.shape}")

        # Transform to encoded version for metrics
        print(f"🔄 Transforming to encoded version...")
        synthetic_encoded = encoder.transform(synthetic_decoded)
        print(f"📊 Encoded version: {synthetic_encoded.shape}")

    elif library == "synthpop":
        # Synthpop outputs label-encoded data (integers)
        synthetic_label_encoded = synthetic_data
        print(f"📊 Synthpop output (label-encoded integers): {synthetic_label_encoded.shape}")

        # Load label encoder to decode back to categories
        label_encoder_path = Path(f"experiments/models/label_encoder_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.pkl")
        print(f"📦 Loading label encoder: {label_encoder_path}")
        label_encoder = SimpleLabelEncoder.load(label_encoder_path)

        # Decode integers back to original categories
        print(f"🔄 Decoding label-encoded data to original format...")
        synthetic_decoded = label_encoder.inverse_transform(synthetic_label_encoded)
        print(f"📊 Decoded version: {synthetic_decoded.shape}")

        # Create RDT-encoded version for metrics (dual pipeline)
        print(f"🔄 Encoding with RDT for metrics compatibility...")
        synthetic_encoded = encoder.transform(synthetic_decoded)
        print(f"📊 RDT-encoded version: {synthetic_encoded.shape}")

    elif library == "synthcity":
        # Synthcity outputs encoded data (RDT preprocessing)
        synthetic_encoded = synthetic_data
        print(f"📊 Synthcity output (encoded): {synthetic_encoded.shape}")

        # Validate that all expected columns are present
        # Load encoded training data to get expected columns
        encoded_training_path = Path(f"experiments/data/encoded/training_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.csv")
        expected_columns = pd.read_csv(encoded_training_path, nrows=0).columns.tolist()

        missing_columns = set(expected_columns) - set(synthetic_encoded.columns)
        if missing_columns:
            print(f"\n❌ ERROR: Synthcity {model_type} failed to generate all columns!")
            print(f"📊 Expected {len(expected_columns)} columns, got {len(synthetic_encoded.columns)}")
            print(f"❌ Missing columns ({len(missing_columns)}): {sorted(missing_columns)}")
            print(f"\n💡 Try a different Synthcity model")
            raise ValueError(f"Synthcity {model_type} generated incomplete data: missing {len(missing_columns)} columns")

        # Reverse transform to decoded version
        print(f"🔄 Reverse transforming to decoded version...")
        synthetic_decoded = encoder.reverse_transform(synthetic_encoded)
        print(f"📊 Decoded version: {synthetic_decoded.shape}")

    # Save both versions
    Path("experiments/data/synthetic").mkdir(parents=True, exist_ok=True)

    base_filename = f"synthetic_data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}"
    encoded_filename = f"experiments/data/synthetic/{base_filename}_encoded.csv"
    decoded_filename = f"experiments/data/synthetic/{base_filename}_decoded.csv"

    synthetic_encoded.to_csv(encoded_filename, index=False)
    synthetic_decoded.to_csv(decoded_filename, index=False)

    print(f"📁 Encoded data saved: {encoded_filename}")
    print(f"📁 Decoded data saved: {decoded_filename}")
    
    # Save detailed metrics (monolithic path + experiment versioning)
    metrics = {
        "experiment_id": experiment_info.get("id", f"gen_{cfg.experiment.seed}"),
        "experiment_seed": cfg.experiment.seed,
        "config_hash": config_hash,
        "timestamp": datetime.now().isoformat(),
        "library": library,
        "model_type": model_type,
        "n_samples_config": cfg.generation.n_samples,  # Original config value (may be null)
        "n_samples_auto_determined": cfg.generation.n_samples is None,  # Flag for auto-sizing from reference
        "samples_generated": len(synthetic_decoded),
        "samples_requested": n_samples,
        "columns_decoded": len(synthetic_decoded.columns),
        "columns_encoded": len(synthetic_encoded.columns),
        "column_names_decoded": list(synthetic_decoded.columns),
        "column_names_encoded": list(synthetic_encoded.columns),
        "generation_time": generation_time,
        "samples_per_second": len(synthetic_decoded) / generation_time if generation_time > 0 else 0,
        "output_file_encoded": encoded_filename,
        "output_file_decoded": decoded_filename,
        "model_source": f"experiments/models/sdg_model_{cfg.experiment.name}_{cfg.experiment.seed}.pkl",
        "encoder_source": str(encoder_path)
    }

    # Add basic data quality metrics (using decoded version for interpretability)
    numeric_cols = synthetic_decoded.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        metrics.update({
            "numeric_columns": len(numeric_cols),
            "numeric_column_names": list(numeric_cols),
            "mean_values": {col: float(synthetic_decoded[col].mean()) for col in numeric_cols},
            "std_values": {col: float(synthetic_decoded[col].std()) for col in numeric_cols},
            "min_values": {col: float(synthetic_decoded[col].min()) for col in numeric_cols},
            "max_values": {col: float(synthetic_decoded[col].max()) for col in numeric_cols}
        })

    # Add categorical data info
    categorical_cols = synthetic_decoded.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        metrics.update({
            "categorical_columns": len(categorical_cols),
            "categorical_column_names": list(categorical_cols),
            "unique_values_per_column": {
                col: int(synthetic_decoded[col].nunique()) for col in categorical_cols
            }
        })

    # Check for missing values
    missing_values = synthetic_decoded.isnull().sum()
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
        print(f"  Samples: {len(synthetic_decoded):,} (auto-matched to reference dataset)")
    else:
        print(f"  Samples: {len(synthetic_decoded):,} (requested: {n_samples:,})")
    print(f"  Columns (decoded): {len(synthetic_decoded.columns)}")
    print(f"  Columns (encoded): {len(synthetic_encoded.columns)}")
    print(f"  Time: {generation_time:.1f}s")
    print(f"  Speed: {len(synthetic_decoded)/generation_time:.0f} samples/sec")
    print(f"  Outputs:")
    print(f"    - Encoded: {encoded_filename}")
    print(f"    - Decoded: {decoded_filename}")


if __name__ == "__main__":
    main()
