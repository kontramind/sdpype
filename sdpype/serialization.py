# sdpype/serialization.py
"""
Unified model serialization for all SDG libraries

This module handles saving and loading of synthetic data generation models
across different libraries (SDV, Synthcity, etc.) with a consistent interface.

SPECIAL HANDLING:
- DPGAN: Uses save_to_file/load_from_file (the working API)
"""

import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import tempfile

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Import serialization functions for different libraries
try:
    from synthcity.utils.serialization import (
        save as synthcity_save,
        load as synthcity_load,
        save_to_file as synthcity_save_to_file,
        load_from_file as synthcity_load_from_file
    )
    from synthcity.plugins import Plugins
    SYNTHCITY_AVAILABLE = True
except ImportError:
    SYNTHCITY_AVAILABLE = False
    synthcity_save = synthcity_load = Plugins = None
    synthcity_save_to_file = synthcity_load_from_file = None

# Check for synthpop availability
try:
    import synthpop
    SYNTHPOP_AVAILABLE = True
except ImportError:
    SYNTHPOP_AVAILABLE = False


# Model format version for compatibility
SERIALIZE_FORMAT_VERSION = "2.1"  # Bumped for new DPGAN handling

# Default paths
DEFAULT_MODEL_DIR = Path("experiments/models")
DEFAULT_METRICS_DIR = Path("experiments/metrics")


class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass


class ModelNotFoundError(SerializationError):
    """Raised when a model file is not found"""
    pass


class LibraryNotSupportedError(SerializationError):
    """Raised when trying to use an unsupported library"""
    pass


def _get_config_hash() -> str:
    """Get config hash from temporary file created during pipeline execution"""
    try:
        if Path('.sdpype_config_hash').exists():
            with open('.sdpype_config_hash', 'r') as f:
                return f.read().strip()
        return "nohash"
    except Exception:
        return "nohash"


def save_model(
    model: Any,
    metadata: Dict[str, Any],
    library: str,
    experiment_seed: int,
    experiment_name: str,
    model_dir: Optional[Path] = None
) -> str:
    """
    Save a trained model with unified interface across libraries

    Special handling: DPGAN uses save_to_file/load_from_file (the working API).

    Args:
        model: Trained model object
        metadata: Experiment metadata (training time, config, etc.)
        library: Library name ('sdv', 'synthcity', etc.)
        experiment_seed: Experiment seed for filename
        experiment_name: Experiment name for filename (required)
        model_dir: Custom model directory (default: experiments/models)
        
    Returns:
        str: Path to saved model file
        
    Raises:
        LibraryNotSupportedError: If library is not supported
        SerializationError: If saving fails
    """
    
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    model_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _get_config_hash()
    model_filename = model_dir / f"sdg_model_{experiment_name}_{config_hash}_{experiment_seed}.pkl"

    # Prepare base model data structure
    model_data = {
        "format_version": SERIALIZE_FORMAT_VERSION,
        "library": library,
        "saved_at": datetime.now().isoformat(),
        **metadata
    }
    
    try:
        if library == "sdv":
            # SDV models can be pickled directly
            model_data["model"] = model
            
        elif library == "synthcity":
            if not SYNTHCITY_AVAILABLE:
                raise LibraryNotSupportedError(
                    "Synthcity not available. Install with: pip install synthcity"
                )

            model_type = metadata.get("model_type", "unknown")

            # Special handling for DPGAN - use file-based API
            if model_type == "dpgan":
                print("⚠️  Using file-based serialization for DPGAN (save_to_file)")

                # Save to temporary file using Synthcity's working API
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp:
                    temp_path = tmp.name

                try:
                    # Use Synthcity's save_to_file (the one that works!)
                    synthcity_save_to_file(temp_path, model)

                    # Read the file as bytes to embed in our structure
                    with open(temp_path, 'rb') as f:
                        model_file_bytes = f.read()

                    model_data["dpgan_model_file"] = model_file_bytes
                    model_data["uses_file_based_serialization"] = True

                finally:
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)

            else:
                # Other Synthcity models: use native byte-based serialization
                model_bytes = synthcity_save(model)
                model_data["model_bytes"] = model_bytes

        elif library == "synthpop":
            if not SYNTHPOP_AVAILABLE:
                raise LibraryNotSupportedError(
                    "Synthpop not available. Install with: pip install python-synthpop"
                )
            # Synthpop models can be pickled directly like SDV
            model_data["model"] = model

        else:
            raise LibraryNotSupportedError(f"Library '{library}' not supported")
        
        # Save as pickle file
        with open(model_filename, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"📁 Model saved: {model_filename} ({library} format)")
        return str(model_filename)
        
    except Exception as e:
        raise SerializationError(f"Failed to save {library} model: {e}") from e


def load_model(experiment_seed: int, experiment_name: str, model_dir: Optional[Path] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained model with unified interface across libraries

    Special handling: DPGAN uses load_from_file (the working API).

    Args:
        experiment_seed: Experiment seed for filename
        experiment_name: Experiment name for filename (required)
        model_dir: Custom model directory (default: experiments/models)

    Returns:
        Tuple[model, metadata]: Loaded model object and metadata dict

    Raises:
        ModelNotFoundError: If model file doesn't exist
        SerializationError: If loading fails
        LibraryNotSupportedError: If library is not supported
    """

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    # Find model file with any config hash
    model_files = list(model_dir.glob(f"sdg_model_{experiment_name}_*_{experiment_seed}.pkl"))

    if not model_files:
        raise ModelNotFoundError(f"No model files found for: {experiment_name}_{experiment_seed}")
    
    # Use the most recent if multiple exist
    model_filename = max(model_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(model_filename, "rb") as f:
            model_data = pickle.load(f)
        
        # Handle different model data formats
        if isinstance(model_data, dict):
            # New format with metadata
            library = model_data.get("library", "sdv")
            format_version = model_data.get("format_version", "1.0")
            
            # Version compatibility handling
            if format_version != SERIALIZE_FORMAT_VERSION:
                warnings.warn(
                    f"Model format version {format_version} differs from current "
                    f"{SERIALIZE_FORMAT_VERSION}. Loading may fail."
                )
            
            if library == "sdv":
                model = model_data["model"]
                
            elif library == "synthcity":
                if not SYNTHCITY_AVAILABLE:
                    raise LibraryNotSupportedError(
                        "Synthcity not available. Install with: pip install synthcity"
                    )

                # Check if this uses file-based serialization (DPGAN)
                if model_data.get("uses_file_based_serialization", False):
                    print("⚠️  Loading DPGAN using file-based deserialization (load_from_file)")

                    # Extract the embedded file bytes
                    model_file_bytes = model_data["dpgan_model_file"]

                    # Write to temporary file
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp:
                        tmp.write(model_file_bytes)
                        temp_path = tmp.name

                    try:
                        # Load using Synthcity's working API
                        model = synthcity_load_from_file(temp_path)
                        print("✅ DPGAN loaded successfully")
                    finally:
                        # Clean up temp file
                        Path(temp_path).unlink(missing_ok=True)

                else:
                    # Normal Synthcity loading (non-DPGAN models)
                    model_bytes = model_data["model_bytes"]
                    model = synthcity_load(model_bytes)

            elif library == "synthpop":
                if not SYNTHPOP_AVAILABLE:
                    raise LibraryNotSupportedError(
                        "Synthpop not available. Install with: pip install python-synthpop"
                    )
                model = model_data["model"]

            else:
                raise LibraryNotSupportedError(f"Library '{library}' not supported")

            return model, model_data

        else:
            # Invalid format
            raise SerializationError("Invalid model file format - expected dict with metadata")

    except pickle.UnpicklingError as e:
        raise SerializationError(f"Failed to unpickle model file: {e}") from e
    except Exception as e:
        raise SerializationError(f"Failed to load model: {e}") from e


def get_model_info(experiment_seed: int, experiment_name: str, model_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get model metadata without loading the full model object
    
    Args:
        experiment_seed: Experiment seed for filename
        experiment_name: Experiment name for filename
        model_dir: Custom model directory (default: experiments/models)
        
    Returns:
        Dict with model metadata (library, type, training info, etc.)
        
    Raises:
        ModelNotFoundError: If model file doesn't exist
        SerializationError: If reading metadata fails
    """

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    # Find model file with any config hash for this experiment and seed
    model_files = list(model_dir.glob(f"sdg_model_{experiment_name}_*_{experiment_seed}.pkl"))

    if not model_files:
        raise ModelNotFoundError(f"No model files found for: {experiment_name}_{experiment_seed}")

    # Use the most recent if multiple exist
    model_filename = max(model_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(model_filename, "rb") as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict):
            # Return metadata without the actual model
            info = {k: v for k, v in model_data.items() 
                   if k not in ["model", "model_bytes", "dpgan_model_file"]}

            # Add file info
            info["file_size_mb"] = model_filename.stat().st_size / (1024 * 1024)
            info["file_path"] = str(model_filename)

            return info
        else:
            # Invalid format
            raise SerializationError("Invalid model file format - expected dict with metadata")

    except Exception as e:
        raise SerializationError(f"Failed to read model info: {e}") from e


def list_saved_models(model_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List all available saved models with their metadata
    
    Args:
        model_dir: Custom model directory (default: experiments/models)
        
    Returns:
        List of dicts with model information
    """
    
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
        
    if not model_dir.exists():
        return []
    
    models = []
    
    for model_file in model_dir.glob("sdg_model_*.pkl"):
        try:
            # Parse filename: sdg_model_experiment_name_config_hash_seed.pkl
            filename_parts = model_file.stem.split("_")
            
            if len(filename_parts) < 5:
                warnings.warn(f"Unexpected filename format: {model_file.name}")
                continue

            try:
                seed = int(filename_parts[-1])
                config_hash = filename_parts[-2]
            except ValueError:
                warnings.warn(f"Cannot extract seed/hash from filename: {model_file.name}")
                continue

            # Extract experiment name
            experiment_name = "_".join(filename_parts[2:-2])

            info = get_model_info(seed, experiment_name, model_dir)

            info["experiment_seed"] = seed
            info["experiment_name"] = experiment_name
            info["config_hash"] = config_hash

            models.append(info)

        except (ValueError, SerializationError) as e:
            warnings.warn(f"Skipping {model_file}: {e}")
            continue
    
    # Sort by timestamp if available
    models.sort(key=lambda x: x.get("saved_at", f"seed_{x['experiment_seed']}"))
    
    return models


def validate_model(experiment_seed: int, experiment_name: str, model_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Validate a saved model by attempting to load it
    
    Args:
        experiment_seed: Experiment seed for filename
        experiment_name: Experiment name for filename
        model_dir: Custom model directory (default: experiments/models)
        
    Returns:
        Dict with validation results
    """
    
    validation_result = {
        "valid": False,
        "error": None,
        "info": None,
        "loadable": False
    }

    try:
        # Check if file exists and get info
        info = get_model_info(experiment_seed, experiment_name, model_dir)
        validation_result["info"] = info
        validation_result["valid"] = True

        # Try to load the actual model
        model, metadata = load_model(experiment_seed, experiment_name, model_dir)
        validation_result["loadable"] = True

        # Add model-specific validation
        library = metadata.get("library", "unknown")
        if library == "sdv":
            validation_result["has_sample_method"] = hasattr(model, "sample")
        elif library == "synthcity":
            validation_result["has_generate_method"] = hasattr(model, "generate")

    except Exception as e:
        validation_result["error"] = str(e)
    
    return validation_result


def get_supported_libraries() -> Dict[str, bool]:
    """
    Get information about which libraries are available
    
    Returns:
        Dict mapping library names to availability status
    """
    
    libraries = {
        "sdv": True,
        "synthcity": SYNTHCITY_AVAILABLE,
        "synthpop": SYNTHPOP_AVAILABLE,
    }
    
    return libraries


def create_model_metadata(
    cfg: DictConfig,
    model_type: str,
    library: str,
    experiment_seed: int,
    training_time: float,
    data: pd.DataFrame,
    experiment_id: str,
    experiment_hash: str
) -> Dict[str, Any]:
    """
    Create standardized metadata for model serialization
    Now includes lineage tracking for recursive generations
    
    Args:
        cfg: Hydra configuration
        model_type: Type of model (e.g., 'ctgan', 'gaussiancopula')
        library: Library name ('sdv', 'synthcity')
        experiment_seed: Experiment seed
        training_time: Training time in seconds
        data: Training data
        experiment_id: Unique experiment identifier
        experiment_hash: Configuration hash
        
    Returns:
        Dict with standardized metadata structure
    """

    # Extract lineage information from experiment config
    generation = cfg.experiment.get("generation", 0)

    # Determine parent model if generation > 0
    parent_model_id = None
    if generation > 0:
        training_file = cfg.data.get("training_file", "")
        if "synthetic_data_" in training_file:
            # Extract parent model ID from synthetic data filename
            filename = Path(training_file).stem
            parent_model_id = filename.replace("synthetic_data_", "")

    return {
        "model_type": model_type,
        "experiment": {
            "id": experiment_id,
            "seed": experiment_seed,
            "hash": experiment_hash,
            "timestamp": datetime.now().isoformat(),
            "name": cfg.experiment.get("name", f"exp_{experiment_seed}"),
            "researcher": cfg.experiment.get("researcher", "anonymous")
        },
        "lineage": {
            "generation": generation,
            "parent_model_id": parent_model_id,
            "root_training_hash": cfg.data.get("root_hash", "unknown"),
            "reference_hash": cfg.data.get("reference_hash", "unknown"),
            "training_hash": cfg.data.get("training_hash", "unknown")
        },        
        "params": OmegaConf.to_container(cfg, resolve=True),
        "training_data_shape": list(data.shape),
        "training_time": training_time,
        "training_data_columns": list(data.columns),
        "parameters": dict(cfg.sdg.parameters) if cfg.sdg.parameters else {}
    }
