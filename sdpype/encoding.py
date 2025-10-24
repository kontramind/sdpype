"""
RDT-based dataset encoding for SDPype pipeline.

This module provides functionality to:
1. Load encoding configurations from YAML files
2. Instantiate RDT transformers from configuration specs
3. Fit transformers on training data
4. Transform/reverse-transform datasets (dual pipeline support)
5. Serialize fitted encoders for downstream use
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pickle

import yaml
import pandas as pd
from rdt.transformers import (
    UniformEncoder,
    OrderedUniformEncoder,
    LabelEncoder,
    OneHotEncoder,
    FrequencyEncoder,
    UnixTimestampEncoder,
    FloatFormatter,
)

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
