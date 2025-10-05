Here's the complete updated README.md:

```markdown
# SDPype - Synthetic Data Pipeline

Reproducible synthetic data generation pipeline with DVC and experiment tracking.

## Features

- Multiple SDG libraries (SDV, Synthcity, Synthpop)
- DVC pipeline for reproducibility
- Experiment versioning with metadata tracking
- Statistical similarity evaluation
- Detection-based quality assessment
- Recursive training for multi-generation analysis

## Quick Start

```bash
# Setup repository
uv run sdpype setup

# Configure experiment in params.yaml
vim params.yaml

# Run pipeline
uv run sdpype pipeline

# View available models
uv run sdpype models
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd sdpype

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Pipeline Stages

The pipeline consists of the following stages:

1. **train_sdg**: Train synthetic data generator
2. **generate_synthetic**: Generate synthetic data
3. **statistical_similarity**: Compare statistical similarity between original and synthetic data
4. **detection_evaluation**: Assess synthetic data quality using classifier-based detection methods

### Running Individual Stages

```bash
# Run specific stages (after editing params.yaml)
uv run sdpype stage train_sdg
uv run sdpype stage generate_synthetic
uv run sdpype stage statistical_similarity
uv run sdpype stage detection_evaluation
```

## Evaluation Framework

SDPype provides statistical similarity evaluation to assess the quality of synthetic data:

### Statistical Similarity Assessment

```bash
# Run statistical similarity evaluation (after setting up experiment in params.yaml)
uv run sdpype stage statistical_similarity
```

## Configuration

Edit `params.yaml` to configure experiments:

```yaml
experiment:
  seed: 42
  name: "baseline"
  description: "Baseline CTGAN experiment"

sdg:
  library: sdv
  model_type: ctgan
  parameters:
    epochs: 500
    batch_size: 100
```

## Available Models

```bash
# List all available SDG models
uv run sdpype models
```

**SDV Models:**
- `gaussiancopula` - Fast Gaussian copula model, good for tabular data
- `ctgan` - High quality generative adversarial network
- `tvae` - Tabular Variational Autoencoder
- `copulagan` - Mix of copula and GAN methods

**Synthcity Models:**
- `ctgan` - CTGAN implementation
- `tvae` - Tabular VAE
- `marginalcoupling` - Advanced copula method
- `nflow` - Normalizing flows
- `ddpm` - Diffusion model

## Project Structure

```
sdpype/
├── experiments/
│   ├── data/
│   │   ├── raw/               # Raw input data
│   │   ├── processed/         # Preprocessed data
│   │   └── synthetic/         # Generated synthetic data
│   ├── models/                # Trained models
│   └── metrics/               # Evaluation results
├── sdpype/                    # Source code
│   ├── cli/                   # CLI commands
│   ├── core/                  # Core functionality
│   ├── evaluation/            # Evaluation metrics
│   ├── training.py            # Model training
│   ├── generation.py          # Data generation
│   └── evaluate.py            # Evaluation pipeline
├── params.yaml                # Configuration
├── dvc.yaml                   # Pipeline definition
├── recursive_train.py         # Recursive training script
├── trace_chain.py             # Chain tracing utility
└── README.md
```

## Example Workflows

### Basic Experiment

```bash
# 1. Set up baseline experiment
vim params.yaml  # Set: experiment.name="baseline", experiment.seed=42
uv run sdpype pipeline

# 2. Try different model
vim params.yaml  # Set: sdg.model_type="gaussiancopula"
uv run sdpype pipeline

# 3. Compare results
uv run sdpype model list
```

### Recursive Training

Run multiple generations where each generation trains on synthetic data from the previous generation:

```bash
# Run 10 generations
python recursive_train.py --generations 10

# Resume from checkpoint
python recursive_train.py --generations 20 --resume

# Resume specific chain
python recursive_train.py --resume-from MODEL_ID --generations 20
```

### Chain Tracing

Analyze metric degradation across generations:

```bash
# Trace a generation chain
python trace_chain.py MODEL_ID

# Export to CSV
python trace_chain.py MODEL_ID --format csv --output chain.csv

# Generate plot
python trace_chain.py MODEL_ID --plot --plot-output degradation.png
```

## Model Management

```bash
# List all trained models
uv run sdpype model list

# Get detailed model info
uv run sdpype model info MODEL_ID
```

## CLI Commands

```bash
# Setup repository
uv run sdpype setup

# Run complete pipeline
uv run sdpype pipeline

# Run specific stage
uv run sdpype stage train_sdg

# Show repository status
uv run sdpype status

# List available models
uv run sdpype models

# Model management
uv run sdpype model list
uv run sdpype model info MODEL_ID

# Metrics inspection
uv run sdpype metrics list
uv run sdpype metrics compare MODEL_ID1 MODEL_ID2

# Purge experiments (destructive)
uv run sdpype purge --yes
```

## Model ID Format

Models are identified with the following format:
```
library_modeltype_refhash_roothash_trnhash_gen_N_cfghash_seed
```

Example:
```
sdv_gaussiancopula_0cf8e0f5_852ba944_852ba944_gen_0_9eadbd5d_51
```

Components:
- `library`: SDG library (sdv, synthcity, etc.)
- `modeltype`: Model type (gaussiancopula, ctgan, etc.)
- `refhash`: Reference data hash
- `roothash`: Root training data hash (generation 0)
- `trnhash`: Current training data hash
- `gen_N`: Generation number
- `cfghash`: Configuration hash
- `seed`: Random seed

## Development

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```

## License

MIT License - see LICENSE file for details

## Citation

If you use SDPype in your research, please cite:

```bibtex
@software{sdpype,
  title = {SDPype: Synthetic Data Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/sdpype}
}
```
```
