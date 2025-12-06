# Synthcity Model Saving in sdpype

This document describes how synthcity models are serialized, saved, and loaded in the sdpype codebase.

## Overview

sdpype wraps synthcity's native serialization functions and adds metadata tracking for reproducibility and lineage. Models are saved as pickle files containing both the serialized model and comprehensive metadata.

## Key Files

| File | Purpose |
|------|---------|
| `sdpype/serialization.py` | Core save/load implementation |
| `sdpype/training.py` | Model creation and training |
| `sdpype/generation.py` | Model loading for synthetic data generation |

## Synthcity Imports

Located in `sdpype/serialization.py` (lines 24-35):

```python
from synthcity.utils.serialization import (
    save as synthcity_save,
    load as synthcity_load,
    save_to_file as synthcity_save_to_file,
    load_from_file as synthcity_load_from_file
)
from synthcity.plugins import Plugins
```

## Saving Models

### Function: `save_model()` in `sdpype/serialization.py` (lines 79-184)

### Standard Synthcity Models (ctgan, rtvae, nflow, etc.)

Uses byte-based serialization:

```python
model_bytes = synthcity_save(model)
model_data["model_bytes"] = model_bytes
```

### DPGAN Special Case

DPGAN requires file-based serialization due to its internal structure:

```python
if model_type == "dpgan":
    # Save to temporary file
    synthcity_save_to_file(temp_path, model)

    # Read bytes and embed in pickle
    with open(temp_path, 'rb') as f:
        model_file_bytes = f.read()

    model_data["dpgan_model_file"] = model_file_bytes
    model_data["uses_file_based_serialization"] = True
```

### Final Persistence

All models are wrapped in a pickle file:

```python
with open(model_filename, "wb") as f:
    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## File Format

- **Extension:** `.pkl`
- **Location:** `experiments/models/`
- **Naming:** `sdg_model_{experiment_name}_{config_hash}_{seed}.pkl`
- **Format Version:** `2.1`

## Saved Model Structure

```python
{
    # Header
    "format_version": "2.1",
    "library": "synthcity",
    "saved_at": "<ISO timestamp>",

    # Model data (one of these):
    "model_bytes": <bytes>,                    # Standard models
    "dpgan_model_file": <bytes>,               # DPGAN only
    "uses_file_based_serialization": True,     # DPGAN flag

    # Metadata
    "model_type": "ctgan|dpgan|rtvae|...",
    "experiment": {
        "id": "<experiment_id>",
        "seed": <int>,
        "hash": "<config_hash>",
        "timestamp": "<ISO>",
        "name": "<experiment_name>",
        "researcher": "<researcher>"
    },
    "lineage": {
        "generation": <int>,
        "parent_model_id": "<id>",
        "root_training_hash": "<hash>",
        "reference_hash": "<hash>",
        "training_hash": "<hash>"
    },
    "params": {<full_config>},
    "training_data_shape": [rows, cols],
    "training_time": <seconds>,
    "training_data_columns": ["col1", "col2", ...],
    "parameters": {<hyperparameters>}
}
```

## Loading Models

### Function: `load_model()` in `sdpype/serialization.py` (lines 186-289)

### Finding the Model File

```python
model_files = list(model_dir.glob(f"sdg_model_{experiment_name}_*_{seed}.pkl"))
model_filename = max(model_files, key=lambda f: f.stat().st_mtime)
```

### Unpickling

```python
with open(model_filename, "rb") as f:
    model_data = pickle.load(f)
```

### Standard Models

```python
model_bytes = model_data["model_bytes"]
model = synthcity_load(model_bytes)
```

### DPGAN Loading

```python
if model_data.get("uses_file_based_serialization", False):
    # Extract embedded bytes
    model_file_bytes = model_data["dpgan_model_file"]

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp:
        tmp.write(model_file_bytes)
        temp_path = tmp.name

    # Load using file-based API
    model = synthcity_load_from_file(temp_path)

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
```

## Generating Synthetic Data

Located in `sdpype/generation.py` (lines 220-224):

```python
synthetic_data = model.generate(count=n_samples).dataframe()
```

## Supported Synthcity Models

Defined in `sdpype/core/models.py`:

- ctgan, rtvae, marginaldistributions, nflow
- arf, adsgan, aim, decaf, pategan, privbayes
- dpgan, ddpm, bayesiannetwork

## Utility Functions

In `sdpype/serialization.py`:

| Function | Lines | Purpose |
|----------|-------|---------|
| `get_model_info()` | 291-340 | Get metadata without loading model |
| `list_saved_models()` | 342-396 | List all available models |
| `validate_model()` | 398-438 | Validate model integrity |

## CLI Commands

```bash
sdpype model list              # List all trained models
sdpype model info <model_id>   # Show model details
sdpype model info <id> --export # Export params to YAML
```

## Key Implementation Notes

1. **Pickle wrapping:** Synthcity's serialized bytes are wrapped in a pickle container with metadata
2. **DPGAN special handling:** Uses file-based API due to internal structure incompatibility with byte serialization
3. **Temp file cleanup:** DPGAN temp files are always cleaned up after save/load
4. **Version tracking:** Format version `2.1` ensures compatibility checks
5. **Highest protocol:** Uses `pickle.HIGHEST_PROTOCOL` for efficiency
