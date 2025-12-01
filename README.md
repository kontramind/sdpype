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

### Detecting Hallucinations in Synthetic Data

SDPype includes advanced capabilities for identifying "hallucinations" - synthetic records that contain unrealistic patterns not present in real data.

#### What Are Hallucinations?

When generating synthetic data, the generator may create records that look realistic but contain patterns that don't exist in real data. Training models on these hallucinations can lead to poor predictions on real patients/cases.

**Example**: A synthetic patient record might combine symptoms in ways that never occur in reality. A model trained on this data would learn false patterns.

#### How We Detect Them: Data Shapley

We use **Data Shapley** to measure each synthetic record's contribution to model performance on real test data:

- **Positive Shapley value**: Record improves predictions → valuable data ✓
- **Zero**: Record has negligible impact
- **Negative Shapley value**: Record harms predictions → hallucination! ⚠️

**Core concept**: Like measuring team members' contributions - add people one at a time and see if the team improves or gets worse.

#### Running Hallucination Detection

```bash
# Basic usage
uv run sdpype downstream mimic-iii-valuation \
  --train-data experiments/data/synthetic/train.csv \
  --test-data experiments/data/real/test.csv \
  --num-samples 20 \
  --max-coalition-size 1500

# Using optimized parameters from training (auto-loads encoding)
uv run sdpype downstream mimic-iii-valuation \
  --train-data experiments/data/synthetic/train.csv \
  --test-data experiments/data/real/test.csv \
  --lgbm-params-json experiments/models/downstream/lgbm_readmission_*.json \
  --num-samples 20 \
  --max-coalition-size 1500
```

#### Understanding the Output

**Console Output:**
```
Negative values (potential hallucinations): 1,234 (17.3%)
  95% CI: [16.4%, 18.2%]  (Wilson score interval)
```

**Interpretation**: 17.3% of synthetic records have negative Shapley values (hallucinations), with 95% confidence the true percentage is between 16.4-18.2%.

**CSV Output** (`experiments/data_valuation/data_valuation_*.csv`):

| Features... | target | shapley_value | sample_index |
|-------------|--------|---------------|--------------|
| [data]      | 1      | -0.00347      | 0            |
| [data]      | 0      | +0.00123      | 1            |

Sorted by Shapley value (most harmful first) for easy identification and removal.

#### Controlling Speed vs Accuracy

Two key parameters control the tradeoff:

**`--num-samples`** (Number of random shuffles)
- Controls **variance** (stability of estimates)
- Higher = smoother estimates, lower = noisier but faster
- **Recommended**: 10-50

**`--max-coalition-size`** (Early truncation)
- Controls **bias** (completeness of evaluation)
- How many records to test together per shuffle
- Larger = more thorough, smaller = faster but may miss patterns
- **Recommended**: 20-30% of dataset size

**Which matters more?** For hallucination detection, **coalition size is more important** than num samples. A record might look harmless in small groups but become obviously bad in larger coalitions.

**Recommended settings for 5,000-10,000 records:**

| Use Case | `--num-samples` | `--max-coalition-size` | Time | Quality |
|----------|-----------------|------------------------|------|---------|
| Quick test | 10 | 1000 | 1-3 hours | Good for initial screening |
| **Balanced (recommended)** | **20** | **1500** | **4-8 hours** | **Best speed/accuracy** |
| Thorough | 50 | 2000 | 12-20 hours | Highest accuracy |

**Rule of thumb**:
- Coalition size should be 20-30% of your dataset
- Num samples should be at least 10-20
- Prioritize larger coalition size over more samples

#### Using the Results

1. **Identify hallucinations**: Filter for `shapley_value < 0`
2. **Remove them**: Create cleaned dataset excluding bottom 10-20%
3. **Retrain**: Train final model on cleaned synthetic data
4. **Improved performance**: Better predictions on real data

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
- `rtvae` - Robust Tabular VAE with beta divergence
- `arf` - Adversarial Random Forests for density estimation
- `marginaldistributions` - Simple baseline sampling from marginal distributions independently
- `adsgan` - AdsGAN for anonymization through data synthesis with privacy guarantees
- `aim` - AIM for differentially private synthetic data with formal privacy guarantees
- `decaf` - DECAF for generating fair synthetic data using causally-aware networks
- `pategan` - PATEGAN for differentially private synthetic data using teacher ensemble aggregation
- `privbayes` - PrivBayes for differentially private data release via Bayesian networks
- `nflow` - Normalizing flows
- `ddpm` - Diffusion model
- `bayesiannetwork` - Bayesian Network using probabilistic graphical models

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
