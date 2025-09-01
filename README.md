# README.md
# SDPype - Synthetic Data Pipeline

🚀 A simple pipeline for generating synthetic data using machine learning.

## Quick Start

### Install
```bash
# Clone and install
git clone <your-repo>
cd sdpype
uv sync
```

### Initialize Project
```bash
# Create new project with sample data
sdpype init my-experiment --sample-data
cd my-experiment
```

### Run Pipeline  
```bash
# Run complete pipeline
sdpype pipeline

# Run with different model
sdpype pipeline --config sdg=ctgan

# Run specific stage
sdpype stage train_sdg
```

### Check Status
```bash
sdpype status
```

## Pipeline Stages

1. **preprocess**: Clean and prepare data (optional)
2. **train_sdg**: Train synthetic data generator
3. **generate_synthetic**: Generate synthetic data

## Configuration

- `config/preprocessing/`: `none`, `standard`  
- `config/sdg/`: `gaussian`, `ctgan`

## Project Structure

```
my-experiment/
├── data/
│   ├── raw/           # Original data
│   ├── processed/     # Preprocessed data
│   └── synthetic/     # Generated data
├── models/            # Trained models
├── metrics/           # Pipeline metrics
└── config/            # Configurations
```
