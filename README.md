# SDPype - Synthetic Data Pipeline

🚀 A pipeline for generating synthetic data with experiment tracking and comprehensive evaluation.

## Quick Start

### Install
```bash
# Clone and install
git clone <your-repo>
cd sdpype
uv sync
```

### Setup Repository
```bash
# Initialize SDPype repository with sample data
uv run sdpype setup
```

### Run Pipeline  
```bash
# SAFE WORKFLOW: Edit params.yaml first, then run pipeline
# This avoids DVC file overwriting issues

# 1. Edit params.yaml to set experiment.name and experiment.seed
vim params.yaml

# 2. Run pipeline safely
uv run sdpype pipeline

# Example: Multiple experiments
# Edit params.yaml: experiment.name="baseline", experiment.seed=42
uv run sdpype pipeline
# Edit params.yaml: experiment.name="variant", experiment.seed=123  
uv run sdpype pipeline
```

### Check Status
```bash
uv run sdpype status
uv run sdpype model list
```

## Experiment Management

**🎯 Key Concept**: Every model requires both an **experiment name** and **seed** for proper organization and isolation.

**⚠️ IMPORTANT**: Always edit `params.yaml` first, then run `uv run sdpype pipeline`. Never use `dvc exp run` or `sdpype exp run` commands as they overwrite previous experiment files.

### Safe Experiment Workflow
```bash
# 1. Edit params.yaml to configure your experiment
vim params.yaml  # Set experiment.name, experiment.seed, and model parameters

# 2. Run the pipeline
uv run sdpype pipeline

# 3. Check results
uv run sdpype model list
uv run sdpype status

# 4. For next experiment, repeat:
vim params.yaml  # Change experiment.name and/or experiment.seed
uv run sdpype pipeline
```

### Model Management
```bash
# List all models (experiment names will be shown)
uv run sdpype model list
uv run sdpype model list --verbose

# Get model information (both name and seed required)
uv run sdpype model info 42 --name baseline --config
uv run sdpype model info 123 --name variant

# Validate model integrity
uv run sdpype model validate 42 --name baseline
```

## Pipeline Stages

1. **preprocess**: Clean and prepare data (optional)
2. **train_sdg**: Train synthetic data generator
3. **generate_synthetic**: Generate synthetic data
4. **evaluate_original**: Evaluate original data quality
5. **evaluate_synthetic**: Evaluate synthetic data quality  
6. **compare_quality**: Compare data quality metrics
7. **evaluate_downstream**: ML performance comparison

### Running Individual Stages
```bash
# Run specific stages (after editing params.yaml)
uv run sdpype stage train_sdg
uv run sdpype stage generate_synthetic
uv run sdpype stage compare_quality
```

## Evaluation Framework

SDPype provides comprehensive evaluation with three approaches:

### 1. Intrinsic Quality Assessment
```bash
# Run evaluation stages (after setting up experiment in params.yaml)
uv run sdpype stage evaluate_original
uv run sdpype stage evaluate_synthetic
uv run sdpype stage compare_quality
```

### 2. Statistical Similarity  
```bash
# Statistical similarity is included in compare_quality stage
uv run sdpype stage compare_quality
```

### 3. Downstream Task Performance
```bash
# Run downstream evaluation (uses current params.yaml settings)
uv run sdpype eval downstream --target category

# Check results
uv run sdpype eval results --seed 42 --report
uv run sdpype eval status
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

evaluation:
  downstream_tasks:
    enabled: true
    target_column: category
    models:
    - RandomForest
    - LogisticRegression
```

## Available Models

```bash
# List all available SDG models
uv run sdpype models
```

**SDV Models:**
- `gaussian_copula` - Fast, good for tabular data
- `ctgan` - High quality, slower training

**Synthcity Models:**
- `synthcity_ctgan` - CTGAN implementation
- `synthcity_tvae` - Tabular VAE
- `synthcity_marginal_coupling` - Advanced copula method
- `synthcity_nflow` - Normalizing flows
- `synthcity_ddpm` - Diffusion model

## Project Structure

```
sdpype/
├── experiments/
│   ├── baseline/              # Experiment-specific directories
│   │   ├── data/
│   │   │   ├── processed/     # Preprocessed data
│   │   │   └── synthetic/     # Generated synthetic data
│   │   ├── models/            # Trained models (sdg_model_42.pkl)
│   │   └── metrics/           # Evaluation results
│   ├── variant/               # Another experiment
│   │   ├── data/
│   │   ├── models/
│   │   └── metrics/
│   └── data/
│       └── raw/               # Shared raw data
├── sdpype/                    # Source code
├── params.yaml                # Configuration
├── dvc.yaml                   # Pipeline definition
└── README.md
```

## Example Workflows

### Basic Experiment
```bash
# 1. Set up baseline experiment
vim params.yaml  # Set: experiment.name="baseline", experiment.seed=42
uv run sdpype pipeline

# 2. Try different model
vim params.yaml  # Set: experiment.name="ctgan_test", sdg.model_type="ctgan"
uv run sdpype pipeline

# 3. Test robustness with different seed
vim params.yaml  # Set: experiment.name="baseline", experiment.seed=123
uv run sdpype pipeline
```

### Model Comparison
```bash
# Compare multiple models for same data
vim params.yaml  # baseline: gaussian_copula, seed=42
uv run sdpype pipeline

vim params.yaml  # variant: ctgan, seed=42  
uv run sdpype pipeline

# Check results
uv run sdpype model list
uv run sdpype eval downstream --target category
```

## Best Practices

1. **Always edit params.yaml first**: Set meaningful experiment.name and experiment.seed
2. **Use descriptive experiment names**: `baseline`, `variant_ctgan`, `high_epochs`, etc.
3. **Test with multiple seeds**: Use different seeds (42, 123, 456) to test robustness
4. **Use the safe workflow**: `params.yaml` → `sdpype pipeline` → check results
5. **Never use exp commands**: Avoid `dvc exp run` or `sdpype exp run` - they overwrite files
6. **Organize experiments logically**: Group related experiments with similar names
7. **Regular cleanup**: Use `sdpype model clean` to remove old models
8. **Track results**: Use `sdpype eval status` to monitor evaluation completion

## Troubleshooting

```bash
# Check repository status
uv run sdpype status

# Validate specific model
uv run sdpype model validate 42 --name baseline

# See what models exist
uv run sdpype model list

# Clean and restart
uv run sdpype nuke --yes
uv run sdpype setup
```

## Why This Workflow?

**Problem**: Commands like `dvc exp run` overwrite previous experiment files, even with different experiment names.

**Solution**: Edit `params.yaml` manually and use `sdpype pipeline`, which creates isolated directories for each experiment.name, preventing file conflicts.

**Result**: Each experiment gets its own directory structure under `experiments/experiment_name/`, ensuring complete isolation and no data loss.