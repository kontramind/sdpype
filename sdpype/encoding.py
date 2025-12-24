"""
RDT-based dataset encoding for SDPype pipeline.

This module provides functionality to:
1. Load encoding configurations from YAML files
2. Instantiate RDT transformers from configuration specs
3. Fit transformers on training data
4. Transform/reverse-transform datasets (dual pipeline support)
5. Serialize fitted encoders for downstream use

CLI Usage:
    python -m sdpype.encoding
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pickle
import json
import time
from datetime import datetime

import yaml
import pandas as pd
import hydra
from omegaconf import DictConfig
from rdt import HyperTransformer
from sdpype.metadata import load_csv_with_metadata
from rdt.transformers import (
    UniformEncoder,
    OrderedUniformEncoder,
    LabelEncoder,
    OneHotEncoder,
    FrequencyEncoder,
    UnixTimestampEncoder,
    FloatFormatter,
)
from rdt.transformers.boolean import BinaryEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# TRANSFORMER REGISTRY
# =============================================================================

TRANSFORMER_REGISTRY = {
    'UniformEncoder': UniformEncoder,
    'OrderedUniformEncoder': OrderedUniformEncoder,
    'LabelEncoder': LabelEncoder,
    'OneHotEncoder': OneHotEncoder,
    'FrequencyEncoder': FrequencyEncoder,
    'UnixTimestampEncoder': UnixTimestampEncoder,
    'FloatFormatter': FloatFormatter,
    'BinaryEncoder': BinaryEncoder,
}


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_encoding_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse encoding configuration from YAML file.

    Args:
        config_path: Path to YAML encoding configuration file

    Returns:
        Dictionary with structure:
        {
            'sdtypes': {col_name: sdtype, ...},
            'transformers': {col_name: transformer_instance, ...}
        }

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config structure is invalid
        KeyError: If transformer type is not in registry
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Encoding config not found: {config_path}")

    logger.info(f"Loading encoding config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config structure
    if 'sdtypes' not in config:
        raise ValueError("Config must contain 'sdtypes' section")
    if 'transformers' not in config:
        raise ValueError("Config must contain 'transformers' section")

    # Extract sdtypes (no instantiation needed)
    sdtypes = config['sdtypes']

    # Instantiate transformers from specs
    transformers = {}
    for col_name, transformer_spec in config['transformers'].items():
        if not isinstance(transformer_spec, dict):
            raise ValueError(
                f"Transformer spec for '{col_name}' must be a dict with 'type' and 'params'"
            )

        transformer_type = transformer_spec.get('type')
        if not transformer_type:
            raise ValueError(f"Transformer spec for '{col_name}' missing 'type' field")

        if transformer_type not in TRANSFORMER_REGISTRY:
            available = ', '.join(TRANSFORMER_REGISTRY.keys())
            raise KeyError(
                f"Unknown transformer type '{transformer_type}' for column '{col_name}'. "
                f"Available types: {available}"
            )

        # Get transformer class
        transformer_class = TRANSFORMER_REGISTRY[transformer_type]

        # Get parameters (default to empty dict)
        params = transformer_spec.get('params', {})

        # Instantiate transformer with params
        try:
            transformer = transformer_class(**params)
            transformers[col_name] = transformer
            logger.debug(f"  {col_name}: {transformer_type}({params})")
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate {transformer_type} for '{col_name}' "
                f"with params {params}: {e}"
            )

    logger.info(f"Loaded {len(transformers)} transformer configurations")

    return {
        'sdtypes': sdtypes,
        'transformers': transformers,
        'config_path': str(config_path),
    }




# =============================================================================
# VALIDATION
# =============================================================================

def validate_config_against_data(
    config: Dict[str, Any],
    data: pd.DataFrame,
    require_all_columns: bool = True
) -> None:
    """
    Validate encoding configuration against actual dataset.

    Args:
        config: Config dictionary from load_encoding_config()
        data: DataFrame to validate against
        require_all_columns: If True, require all config columns exist in data

    Raises:
        ValueError: If validation fails
    """
    config_columns = set(config['sdtypes'].keys())
    data_columns = set(data.columns)

    # Check for columns in config but not in data
    missing_in_data = config_columns - data_columns
    if missing_in_data and require_all_columns:
        raise ValueError(
            f"Columns in encoding config not found in data: {missing_in_data}"
        )

    # Check for columns in data but not in config (warning only)
    missing_in_config = data_columns - config_columns
    if missing_in_config:
        logger.warning(
            f"Columns in data not found in encoding config (will not be encoded): "
            f"{missing_in_config}"
        )

    # Validate transformer config matches sdtype config
    transformer_columns = set(config['transformers'].keys())
    if transformer_columns != config_columns:
        sdtype_only = config_columns - transformer_columns
        transformer_only = transformer_columns - config_columns
        msg = []
        if sdtype_only:
            msg.append(f"sdtypes without transformers: {sdtype_only}")
        if transformer_only:
            msg.append(f"transformers without sdtypes: {transformer_only}")
        raise ValueError(
            f"Mismatch between sdtypes and transformers sections. {' | '.join(msg)}"
        )

    logger.info("✓ Config validation passed")


# =============================================================================
# RDT DATASET ENCODER
# =============================================================================

class RDTDatasetEncoder:
    """
    RDT-based dataset encoder for SDPype pipeline.

    Uses RDT's HyperTransformer under the hood for proper orchestration
    of multiple transformers.

    Supports:
    - Fitting transformers on training data
    - Transforming data to encoded (numeric) format
    - Reverse transforming encoded data back to original format (dual pipeline)
    - Serialization of fitted encoders for downstream use

    Usage:
        # Load config
        config = load_encoding_config('encoding_config.yaml')

        # Create encoder
        encoder = RDTDatasetEncoder(config)

        # Fit on training data
        encoder.fit(training_df)

        # Transform data
        encoded_train = encoder.transform(training_df)
        encoded_ref = encoder.transform(reference_df)

        # Reverse transform (for dual pipeline)
        decoded_synthetic = encoder.reverse_transform(synthetic_encoded_df)

        # Save fitted encoders
        encoder.save('fitted_encoders.pkl')
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize encoder with configuration.

        Args:
            config: Configuration dict from load_encoding_config()
        """
        self.config = config
        self.sdtypes = config['sdtypes']
        self.transformers = config['transformers']

        # Create HyperTransformer (will be configured during fit)
        self.ht = HyperTransformer()

        self._is_fitted = False
        logger.info(f"Initialized RDTDatasetEncoder with {len(self.sdtypes)} columns")

    def fit(self, training_data: pd.DataFrame) -> 'RDTDatasetEncoder':
        """
        Fit transformers on training data.

        Critical: Transformers are fitted ONLY on training data to prevent
        data leakage from reference/test sets.

        Args:
            training_data: Training DataFrame

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If validation fails
        """
        logger.info("Fitting encoders on training data...")

        # Validate config against data
        validate_config_against_data(self.config, training_data)

        # Configure HyperTransformer
        # Step 1: Detect initial config from data (required by RDT)
        self.ht.detect_initial_config(data=training_data)

        # Step 2: Update with custom sdtypes from config
        self.ht.update_sdtypes(column_name_to_sdtype=self.sdtypes)

        # Step 3: Update with custom transformers from config
        self.ht.update_transformers(column_name_to_transformer=self.transformers)

        # Step 4: Fit HyperTransformer on all data at once
        self.ht.fit(data=training_data)

        self._is_fitted = True
        logger.info(f"✓ Fitted {len(self.transformers)} transformers")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.

        Args:
            data: DataFrame to transform

        Returns:
            Transformed DataFrame (numeric format)

        Raises:
            RuntimeError: If encoder not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted. Call .fit() first.")

        logger.info(f"Transforming data ({len(data)} rows)...")

        # Use HyperTransformer to transform all columns at once
        transformed_data = self.ht.transform(data=data)

        logger.info(f"✓ Transformed to {transformed_data.shape[1]} columns")

        return transformed_data

    def reverse_transform(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse transform encoded data back to original format.

        This enables the dual pipeline approach where:
        - Metrics operate on encoded (numeric) data
        - Detection/privacy checks operate on native sdtype data

        Args:
            encoded_data: Encoded DataFrame from .transform()

        Returns:
            Decoded DataFrame with original sdtypes

        Raises:
            RuntimeError: If encoder not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted. Call .fit() first.")

        logger.info(f"Reverse transforming data ({len(encoded_data)} rows)...")

        # Use HyperTransformer to reverse transform all columns at once
        decoded_data = self.ht.reverse_transform(data=encoded_data)

        logger.info(f"✓ Reverse transformed to {decoded_data.shape[1]} columns")

        return decoded_data

    @property
    def is_fitted(self) -> bool:
        """Check if encoder has been fitted."""
        return self._is_fitted

    def save(self, filepath: Path) -> None:
        """
        Save fitted encoders to disk.

        Saves:
        - Fitted HyperTransformer (with learned state)
        - Original config (for reference)
        - Metadata about encoding process

        Args:
            filepath: Path to save pickle file

        Raises:
            RuntimeError: If encoder not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted encoder. Call .fit() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'hyper_transformer': self.ht,
            'sdtypes': self.sdtypes,
            'config': self.config,
            'encoding_version': '2.0',  # Updated version for HyperTransformer
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"✓ Saved fitted encoders to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'RDTDatasetEncoder':
        """
        Load fitted encoders from disk.

        Args:
            filepath: Path to pickle file

        Returns:
            Fitted RDTDatasetEncoder instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Encoder file not found: {filepath}")

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Create instance with config
        instance = cls(save_data['config'])

        # Restore fitted HyperTransformer
        instance.ht = save_data['hyper_transformer']
        instance._is_fitted = True

        logger.info(f"✓ Loaded fitted encoders from: {filepath}")

        return instance



# =============================================================================
# CLI ENTRY POINT
# =============================================================================

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
    Encode datasets using RDT transformers for SDPype pipeline.

    This stage:
    1. Loads encoding configuration from YAML
    2. Fits transformers on training data only
    3. Transforms both training and reference datasets
    4. Saves encoded data (for metrics)
    5. Saves decoded data (for detection/privacy - dual pipeline)
    6. Serializes fitted encoders for downstream use
    7. Records encoding metrics
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    print("🔧 Starting dataset encoding...")
    config_hash = _get_config_hash()
    experiment_name = cfg.experiment.name
    seed = cfg.experiment.seed

    start_time = time.time()

    # 1. Load encoding configuration
    print(f"\n📋 Loading encoding configuration...")

    if not hasattr(cfg, 'encoding') or not cfg.encoding.config_file:
        print("❌ No encoding.config_file specified in params.yaml")
        print("💡 Add 'encoding.config_file: experiments/configs/encoding/your_config.yaml'")
        raise ValueError("encoding.config_file not configured")

    encoding_config_path = Path(cfg.encoding.config_file)
    if not encoding_config_path.exists():
        print(f"❌ Encoding config not found: {encoding_config_path}")
        print("💡 Create a config file based on experiments/configs/encoding/default.yaml")
        raise FileNotFoundError(f"Encoding config not found: {encoding_config_path}")

    config = load_encoding_config(encoding_config_path)
    print(f"✓ Loaded config with {len(config['transformers'])} transformers")

    # 2. Load datasets with metadata for type consistency
    print(f"\n📊 Loading datasets...")
    training_file = Path(cfg.data.training_file)
    reference_file = Path(cfg.data.reference_file)
    metadata_file = Path(cfg.data.metadata_file)

    if not training_file.exists():
        raise FileNotFoundError(f"Training file not found: {training_file}")
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    training_data = load_csv_with_metadata(training_file, metadata_file)
    reference_data = load_csv_with_metadata(reference_file, metadata_file)

    print(f"✓ Training data: {training_data.shape}")
    print(f"✓ Reference data: {reference_data.shape}")

    # 3. Create and fit encoder
    print(f"\n🔧 Fitting encoders on training data...")
    encoder = RDTDatasetEncoder(config)
    encoder.fit(training_data)

    # 4. Transform datasets
    print(f"\n🔄 Transforming datasets...")
    encoded_training = encoder.transform(training_data)
    encoded_reference = encoder.transform(reference_data)

    print(f"✓ Encoded training: {encoded_training.shape}")
    print(f"✓ Encoded reference: {encoded_reference.shape}")

    # 5. Reverse transform for dual pipeline (decoded versions)
    print(f"\n🔄 Creating decoded versions for dual pipeline...")
    decoded_training = encoder.reverse_transform(encoded_training)
    decoded_reference = encoder.reverse_transform(encoded_reference)

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

    # Generate filenames with experiment naming convention
    base_name = f"{experiment_name}_{config_hash}_{seed}"

    # Save encoded data (numeric - for metrics)
    encoded_training_path = encoded_dir / f"training_{base_name}.csv"
    encoded_reference_path = encoded_dir / f"reference_{base_name}.csv"
    encoded_training.to_csv(encoded_training_path, index=False)
    encoded_reference.to_csv(encoded_reference_path, index=False)
    print(f"✓ Encoded data → {encoded_dir}")

    # Save decoded data (original sdtypes - for detection/privacy)
    decoded_training_path = decoded_dir / f"training_{base_name}.csv"
    decoded_reference_path = decoded_dir / f"reference_{base_name}.csv"
    decoded_training.to_csv(decoded_training_path, index=False)
    decoded_reference.to_csv(decoded_reference_path, index=False)
    print(f"✓ Decoded data → {decoded_dir}")

    # Save fitted encoders
    encoder_path = models_dir / f"encoders_{base_name}.pkl"
    encoder.save(encoder_path)
    print(f"✓ Fitted encoders → {encoder_path}")

    # 7. Record metrics
    elapsed_time = time.time() - start_time

    metrics = {
        "encoding_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "experiment": {
            "name": experiment_name,
            "seed": seed,
            "config_hash": config_hash,
        },
        "config_file": str(encoding_config_path),
        "input_shapes": {
            "training": list(training_data.shape),
            "reference": list(reference_data.shape),
        },
        "output_shapes": {
            "encoded_training": list(encoded_training.shape),
            "encoded_reference": list(encoded_reference.shape),
            "decoded_training": list(decoded_training.shape),
            "decoded_reference": list(decoded_reference.shape),
        },
        "transformers": {
            col: type(trans).__name__
            for col, trans in encoder.transformers.items()
        },
        "encoding_time_seconds": round(elapsed_time, 2),
        "outputs": {
            "encoded_training": str(encoded_training_path),
            "encoded_reference": str(encoded_reference_path),
            "decoded_training": str(decoded_training_path),
            "decoded_reference": str(decoded_reference_path),
            "fitted_encoders": str(encoder_path),
        }
    }

    metrics_path = metrics_dir / f"encoding_{base_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics → {metrics_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Encoding complete!")
    print(f"{'='*70}")
    print(f"⏱️  Time: {elapsed_time:.2f}s")
    print(f"📊 Columns: {training_data.shape[1]} → {encoded_training.shape[1]}")
    print(f"💾 Outputs: {encoded_dir}, {decoded_dir}, {encoder_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()