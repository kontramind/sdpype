# sdpype/serialization.py
"""
Unified model serialization for all SDG libraries

This module handles saving and loading of synthetic data generation models
across different libraries (SDV, Synthcity, etc.) with a consistent interface.
"""

import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Import serialization functions for different libraries
try:
    from synthcity.utils.serialization import save as synthcity_save, load as synthcity_load
    SYNTHCITY_AVAILABLE = True
except ImportError:
    SYNTHCITY_AVAILABLE = False
    synthcity_save = synthcity_load = None


# Model format version for compatibility
SERIALIZE_FORMAT_VERSION = "2.0"

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
    model_filename = model_dir / f"sdg_model_{experiment_name}_{experiment_seed}.pkl"

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
            # Synthcity models: serialize to bytes
            model_bytes = synthcity_save(model)
            model_data["model_bytes"] = model_bytes
            
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

    model_filename = model_dir / f"sdg_model_{experiment_name}_{experiment_seed}.pkl"

    if not model_filename.exists():
        raise ModelNotFoundError(f"Model file not found: {model_filename}")

    try:
        with open(model_filename, "rb") as f:
            model_data = pickle.load(f)
        
        # Handle different model data formats
        if isinstance(model_data, dict):
            # New format with metadata
            library = model_data.get("library", "sdv")  # Default to SDV for compatibility
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
                model_bytes = model_data["model_bytes"]
                model = synthcity_load(model_bytes)
                
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

    model_filename = model_dir / f"sdg_model_{experiment_name}_{experiment_seed}.pkl"

    if not model_filename.exists():
        raise ModelNotFoundError(f"Model file not found: {model_filename}")

    try:
        with open(model_filename, "rb") as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict):
            # Return metadata without the actual model
            info = {k: v for k, v in model_data.items() 
                   if k not in ["model", "model_bytes"]}

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
            # Parse filename: sdg_model_name_seed.pkl
            # Expected format: sdg_model_experiment_name_seed.pkl
            filename_parts = model_file.stem.split("_")
            
            if len(filename_parts) < 4:  # Less than sdg_model_name_seed
                warnings.warn(f"Unexpected filename format: {model_file.name}")
                continue

            try:
                seed = int(filename_parts[-1])  # Last part is always seed
            except ValueError:
                warnings.warn(f"Cannot extract seed from filename: {model_file.name}")
                continue

            # Extract experiment name: everything between "sdg_model_" and "_seed"
            experiment_name = "_".join(filename_parts[2:-1])

            info = get_model_info(seed, experiment_name, model_dir)

            info["experiment_seed"] = seed
            info["experiment_name"] = experiment_name

            models.append(info)

        except (ValueError, SerializationError) as e:
            # Skip files that don't match pattern or can't be read
            warnings.warn(f"Skipping {model_file}: {e}")
            continue
    
    # Sort by timestamp if available, otherwise by seed
    models.sort(key=lambda x: x.get("saved_at", f"seed_{x['experiment_seed']}"))
    
    return models


def delete_model(experiment_seed: int, experiment_name: str, model_dir: Optional[Path] = None) -> bool:
    """
    Delete a saved model file
    
    Args:
        experiment_seed: Experiment seed for filename
        experiment_name: Experiment name for filename
        model_dir: Custom model directory (default: experiments/models)
        
    Returns:
        bool: True if deleted successfully, False if file didn't exist
        
    Raises:
        SerializationError: If deletion fails
    """
    
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    model_filename = model_dir / f"sdg_model_{experiment_name}_{experiment_seed}.pkl"

    if not model_filename.exists():
        return False
    
    try:
        model_filename.unlink()
        print(f"🗑️ Deleted model: {model_filename}")
        return True
    except Exception as e:
        raise SerializationError(f"Failed to delete model: {e}") from e


def copy_model(
    source_seed: int,
    target_seed: int,
    source_name: str,
    target_name: str,    
    model_dir: Optional[Path] = None,
    update_metadata: bool = True
) -> str:
    """
    Copy a model from one experiment seed to another
    
    Args:
        source_seed: Source experiment seed
        target_seed: Target experiment seed
        source_name: Source experiment name
        target_name: Target experiment name
        model_dir: Custom model directory (default: experiments/models)
        update_metadata: Whether to update timestamp and seed in metadata
        
    Returns:
        str: Path to copied model file
        
    Raises:
        ModelNotFoundError: If source model doesn't exist
        SerializationError: If copying fails
    """
    
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    source_file = model_dir / f"sdg_model_{source_name}_{source_seed}.pkl"
    target_file = model_dir / f"sdg_model_{target_name}_{target_seed}.pkl"

    if not source_file.exists():
        raise ModelNotFoundError(f"Source model not found: {source_file}")
    
    try:
        if update_metadata:
            # Load, update metadata, and save
            with open(source_file, "rb") as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                model_data["copied_from_seed"] = source_seed
                model_data["copied_at"] = datetime.now().isoformat()
                
                # Update experiment seed in nested metadata
                if "experiment" in model_data and isinstance(model_data["experiment"], dict):
                    model_data["experiment"]["seed"] = target_seed
            
            with open(target_file, "wb") as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Simple file copy
            import shutil
            shutil.copy2(source_file, target_file)
        
        print(f"📋 Copied model: {source_file} → {target_file}")
        return str(target_file)
        
    except Exception as e:
        raise SerializationError(f"Failed to copy model: {e}") from e


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
            # Basic SDV model validation
            validation_result["has_sample_method"] = hasattr(model, "sample")
        elif library == "synthcity":
            # Basic Synthcity model validation
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
        "sdv": True,  # Always supported (pickle-based)
        "synthcity": SYNTHCITY_AVAILABLE
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
    
    Args:
        cfg: Hydra configuration
        model_type: Type of model (e.g., 'ctgan', 'gaussian_copula')
        library: Library name ('sdv', 'synthcity')
        experiment_seed: Experiment seed
        training_time: Training time in seconds
        data: Training data
        experiment_id: Unique experiment identifier
        experiment_hash: Configuration hash
        
    Returns:
        Dict with standardized metadata structure
    """
    
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
        "config": OmegaConf.to_container(cfg, resolve=True),
        "training_data_shape": list(data.shape),
        "training_time": training_time,
        "training_data_columns": list(data.columns),
        "parameters": dict(cfg.sdg.parameters) if cfg.sdg.parameters else {}
    }