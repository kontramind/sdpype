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
4. **evaluate_original**: Evaluate original data quality
5. **evaluate_synthetic**: Evaluate synthetic data quality  
6. **compare_quality**: Compare data quality metrics
7. **evaluate_downstream**: ML performance comparison ⭐ NEW

## Evaluation Framework

SDPype provides comprehensive evaluation with three approaches:

### 1. Intrinsic Quality Assessment
```bash
# Evaluate data quality metrics
dvc repro -s evaluate_original
dvc repro -s evaluate_synthetic
dvc repro -s compare_quality
```

### 2. Statistical Similarity  
```bash
# Compare statistical distributions
dvc repro -s compare_quality  # includes statistical similarity
```

### 3. Downstream Task Performance ⭐ NEW
```bash
# Compare ML model performance on original vs synthetic data
sdpype eval downstream --target category

# With specific configuration
sdpype eval downstream --target income --task-type regression --models "RandomForest,SVM"

# Check results
sdpype eval results --seed 42 --report
sdpype eval status
```

## Evaluation Commands

```bash
# Run downstream evaluation
sdpype eval downstream              # Use default target from config
sdpype eval downstream --target age # Specify target column

# View results
sdpype eval results --seed 42       # Show summary
sdpype eval results --seed 42 --report  # Show full report
sdpype eval results --compare "42,43,44"  # Compare multiple experiments

# Check evaluation status
sdpype eval status                  # Pipeline status overview
```

## Project Structure

```
my-experiment/
├── data/
│   ├── raw/           # Original data
│   ├── processed/     # Preprocessed data
│   └── synthetic/     # Generated data
├── models/            # Trained models
├── metrics/           # Pipeline metrics
```
