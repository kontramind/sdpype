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

## Sampling Methodology

### Overview

The `transform` command in `prepare_mimic_iii_mini_cli.py` implements a **two-stage stratified sampling** strategy designed to:
1. Prevent data leakage at the episode level
2. Create balanced train/test splits for fair evaluation
3. Preserve a representative unsampled population for deployment testing

### Sampling Strategy

#### Stage 1: Random Study Cohort Selection

First, we randomly sample 20,000 ICU stay episodes (ICUSTAY_ID) from the full population to create a "study cohort":

```
All 61,532 episodes
    ↓
Random sample 20,000 episodes
    (Study Cohort)
```

**Why random?** This mimics realistic research practice where you collect a representative sample for your study.

#### Stage 2: Stratified Train/Test Split

The 20,000-episode cohort is then split 50/50 into training and test sets, **stratified by the target variable (READMIT)** to ensure identical class distributions:

```
Study Cohort (20,000)
    ↓
Stratified split on READMIT
    ↓
├─ Train: 10,000 episodes (balanced READMIT)
└─ Test:  10,000 episodes (balanced READMIT)
```

**Why stratified?** This ensures train and test have identical outcome distributions, eliminating sampling artifacts in model evaluation.

#### Stage 3: Unsampled Population

All remaining episodes (not in the study cohort) become the "unsampled" dataset:

```
Remaining ~41,000 episodes
    ↓
Unsampled (Population)
```

**Why keep all?** The unsampled dataset represents the full diverse population with natural class distribution, useful for testing deployment scenarios.

### Final Dataset Characteristics

| Dataset | Size | READMIT Rate | Purpose |
|---------|------|--------------|---------|
| **Training** | 10,000 | ~10.5% (stratified) | Model training with balanced outcomes |
| **Test** | 10,000 | ~10.5% (stratified) | Fair model evaluation |
| **Unsampled** | ~41,000 | ~10.3% (natural) | Deployment/population testing |

### Leakage Prevention

The sampling ensures **no episode-level leakage**:
- Each `ICUSTAY_ID` appears in exactly ONE split
- No ICU stay is shared between train/test/unsampled

**Note on patient-level overlap:** The same patient (`SUBJECT_ID`) may have different ICU episodes in different splits. This is:
- ✓ **Acceptable** for episode-level prediction (predicting readmission for a specific ICU stay)
- ❌ **Not acceptable** for patient-level prediction (predicting patient readmission risk)

Our use case is **episode-level prediction**, so patient overlap is expected and not considered leakage.

### Usage Examples

#### Basic Usage (with stratification)

```bash
uv run prepare_mimic_iii_mini_cli.py transform \
    MIMIC-III-mini-population.xlsx \
    --output MIMIC-III-mini-core \
    --sample 10000 \
    --seed 42
```

This creates:
- `dseed42/MIMIC-III-mini-core_sample10000_dseed42_training.csv` (10k, stratified)
- `dseed42/MIMIC-III-mini-core_sample10000_dseed42_test.csv` (10k, stratified)
- `dseed42/MIMIC-III-mini-core_sample10000_dseed42_unsampled.csv` (~41k, natural)

#### With Validation (Keep IDs)

```bash
uv run prepare_mimic_iii_mini_cli.py transform \
    MIMIC-III-mini-population.xlsx \
    --output MIMIC-III-mini-core \
    --sample 10000 \
    --seed 42 \
    --keep-ids
```

The `--keep-ids` flag:
- Preserves ICUSTAY_ID, SUBJECT_ID, HADM_ID in output files
- Runs comprehensive validation checks
- Displays leakage detection report
- Useful for debugging and quality assurance

#### Without Stratification (Random Split)

```bash
uv run prepare_mimic_iii_mini_cli.py transform \
    MIMIC-III-mini-population.xlsx \
    --output MIMIC-III-mini-core \
    --sample 10000 \
    --seed 42 \
    --no-stratify
```

Use `--no-stratify` if you want pure random sampling without balancing.

### Validation Output

When using `--keep-ids`, the command displays a validation report:

```
════════════════════════════════════════════════════════════
Data Split Validation
════════════════════════════════════════════════════════════

1. Episode-Level Leakage Check (ICUSTAY_ID):

  Train ∩ Test:           0 ✓ No leakage
  Train ∩ Unsampled:      0 ✓ No leakage
  Test ∩ Unsampled:       0 ✓ No leakage

2. Patient-Level Overlap (SUBJECT_ID):

  Train ∩ Test:         959 patients
  Train ∩ Unsampled:  2,639 patients
  Test ∩ Unsampled:   2,663 patients

  ℹ Patient-level overlap is EXPECTED for episode-level prediction.
    Same patient can have different ICU episodes in different splits.

3. Target Distribution (READMIT):

  Training:  0.1050 (1,050 / 10,000)
  Test:      0.1050 (1,050 / 10,000)
  Unsampled: 0.1036 (4,301 / 41,532)

  ✓ Train/Test distributions identical (diff: 0.0000)
    Stratified sampling is working correctly.

════════════════════════════════════════════════════════════
```

### Rationale

This two-stage approach provides:

1. **Realistic Research Scenario**
   - Stage 1 (random cohort) = "We collected 20k patients for our study"
   - Stage 2 (stratified split) = "We split them ensuring balanced outcomes"
   - Stage 3 (population) = "We test on broader population"

2. **Fair Evaluation**
   - Train/test have identical distributions → no sampling artifacts
   - Unsampled has natural distribution → realistic deployment testing

3. **Flexibility**
   - `--stratify`: For controlled experiments (default)
   - `--no-stratify`: For natural sampling variation
   - `--keep-ids`: For validation and debugging

### References

This methodology follows best practices for:
- Clinical machine learning (episode-level prediction)
- Stratified sampling (scikit-learn's `train_test_split`)
- Data leakage prevention (ICUSTAY_ID-level splitting)

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

## Standalone Evaluation Tools

SDPype provides standalone CLI tools for specific evaluation tasks:

### SDMetrics Quality Report

Comprehensive synthesis quality analysis using SDMetrics QualityReport with pairwise correlation matrices.

```bash
# Run quality analysis
python sdmetrics_quality_cli.py \
  --real experiments/data/real/test.csv \
  --synthetic experiments/data/synthetic/test.csv \
  --metadata experiments/data/metadata.json \
  --output quality_report.json
```

**What it does:**
- Generates three correlation matrices:
  1. Real data correlations (with pairwise deletion and confidence metrics)
  2. Synthetic data correlations (complete data analysis)
  3. Quality scores (SDMetrics similarity assessment)
- Computes distribution similarity for individual columns
- Analyzes correlation preservation across column pairs
- Provides diagnostic analysis with flags (✓/⚠/✗) for each pair

**Key features:**
- Uses pairwise deletion to handle missing values in real data
- Supports mixed data types (numerical, categorical, boolean)
- Confidence flags based on sample sizes (✓ ≥1000, ⚠ 500-999, ✗ 50-499, - <50)
- Diagnostic analysis identifies learning failures and spurious correlations
- JSON export for automated pipelines

**Interpreting results:**
- Quality scores ≥0.8: Excellent preservation
- Quality scores ≥0.6: Good preservation
- Quality scores <0.6: Poor preservation
- Diagnostic flags:
  - ✓ = Good preservation of correlations
  - ⚠ = Mixed quality or moderate issues
  - ✗ = Poor preservation or learning failure

### K-Anonymity Evaluation

Compute k-anonymity metrics for privacy assessment:

```bash
# Run k-anonymity analysis
python k_anonymity_cli.py \
  --population data/population.csv \
  --reference data/reference.csv \
  --synthetic data/synthetic.csv \
  --metadata data/metadata.json \
  --qi-cols "age,gender,zipcode" \
  --output k_anonymity_results.json
```

## Statistical Analysis

### Aggregating Metrics Across Experiments

When analyzing results from multiple experimental runs, SDPype provides two aggregation tools:

1. **`aggregate_metrics_cli.py`**: Simple bootstrap aggregation for independent runs
2. **`aggregate_hybrid_stratified_metrics_cli.py`**: Hierarchical aggregation for nested experimental designs

### Hybrid Stratified Aggregation

For experiments with a hierarchical structure (e.g., multiple data subsets × multiple model seeds), use the hybrid stratified approach to avoid pseudoreplication and obtain valid statistical inferences.

#### When to Use This Method

Use `aggregate_hybrid_stratified_metrics_cli.py` when your experimental design has:
- Multiple **data subsets** (dseeds) - different train/test splits or data samples
- Multiple **model training seeds** (mseeds) per data subset
- **Nested structure**: mseeds are nested within dseeds

Example: 6 data subsets × 6 model seeds = 36 total runs

#### The Pseudoreplication Problem

**Problem**: Treating all 36 runs as independent observations inflates statistical confidence because model seeds within the same data subset are correlated (they all train on the same data).

**Solution**: Use hierarchical aggregation that respects the nested structure.

#### Two-Step Aggregation Method

**Step 1: Average across model seeds (mseeds) within each data subset (dseed)**
- For each dseed, compute the mean across all mseeds
- This produces one representative value per data subset
- Smooths out model training variance

**Step 2: Compute t-based confidence intervals on dseed averages**
- Treat dseed averages as independent observations (n = number of dseeds)
- Use t-distribution for small sample sizes (critical for n < 30)
- Calculate both confidence intervals (CI) and prediction intervals (PI)

#### Statistical Formulas

**Confidence Interval (95%):**
```
CI = x̄ ± t(α/2, df) × (s / √n)
```
where:
- `x̄` = mean of dseed averages
- `s` = standard deviation of dseed averages
- `n` = number of data subsets (dseeds)
- `df` = n - 1 (degrees of freedom)
- `t(α/2, df)` = critical value from t-distribution (e.g., 2.571 for n=6, 95% CI)

**Prediction Interval (95%):**
```
PI = x̄ ± t(α/2, df) × s × √(1 + 1/n)
```

#### Interpretation

**Confidence Interval (Inner band, colored):**
- "Where the true population mean likely is"
- Quantifies uncertainty about the mean
- Narrows as sample size increases

**Prediction Interval (Outer band, gray):**
- "Where a NEW data subset would likely fall"
- Accounts for both estimation uncertainty AND natural variation
- Remains wider than CI even with large samples
- More relevant for predicting future experimental outcomes

**Example (n=6 dseeds):**
- CI: Mean ± 2.571 × (s / √6) ≈ Mean ± 1.05×s
- PI: Mean ± 2.571 × s × √(1 + 1/6) ≈ Mean ± 2.82×s

The prediction interval is ~2.7× wider, honestly representing the variation you'd expect in a new experiment.

#### Usage Example

```bash
# Aggregate results from hierarchical design
python aggregate_hybrid_stratified_metrics_cli.py \
  "./mimic_iii_baseline_dseed*_mseed*/" \
  -s results.html \
  -c aggregate_metrics.csv \
  -i dseed_averages.csv

# Output files:
# - results.html: Interactive visualizations with CI and PI bands
# - aggregate_metrics.csv: Summary statistics per generation
# - dseed_averages.csv: Intermediate dseed-level averages
```

#### Key Advantages

1. **Avoids pseudoreplication**: Properly handles nested structure (Hurlbert, 1984)
2. **Correct for small samples**: Uses t-distribution instead of normal approximation
3. **Honest uncertainty**: Prediction intervals show realistic variation
4. **Statistically valid**: Provides proper basis for scientific claims

#### When NOT to Use This Method

If your experimental runs are truly independent (e.g., completely different datasets, not just different seeds on the same data), use the simpler `aggregate_metrics_cli.py` instead.

## References

### Statistical Methods

1. **Hurlbert, S. H. (1984).** "Pseudoreplication and the design of ecological field experiments." *Ecological Monographs*, 54(2), 187-211.
   - Foundational paper on pseudoreplication in experimental design

2. **Lazic, S. E. (2010).** "The problem of pseudoreplication in neuroscientific studies: is it affecting your analysis?" *BMC Neuroscience*, 11(5).
   - Modern application of pseudoreplication concepts

3. **Student (Gosset, W. S.). (1908).** "The probable error of a mean." *Biometrika*, 6(1), 1-25.
   - Original t-distribution paper for small sample inference

4. **Meeker, W. Q., Hahn, G. J., & Escobar, L. A. (2017).** *Statistical Intervals: A Guide for Practitioners and Researchers* (2nd ed.). Wiley.
   - Comprehensive guide to confidence and prediction intervals

5. **Quinn, G. P., & Keough, M. J. (2002).** *Experimental Design and Data Analysis for Biologists*. Cambridge University Press.
   - Practical guide to nested experimental designs

6. **Cumming, G. (2014).** "The new statistics: Why and how." *Psychological Science*, 25(1), 7-29.
   - Modern perspective on estimation and confidence intervals

### Data Valuation & Shapley Values

7. **Ghorbani, A., & Zou, J. (2019).** "Data Shapley: Equitable valuation of data for machine learning." *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 97, 2242-2251.
   - Foundational paper on Data Shapley for valuing training examples

8. **Jia, R., et al. (2019).** "Towards efficient data valuation based on the Shapley value." *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)*.
   - Efficient algorithms for computing Data Shapley values

### BibTeX Format

```bibtex
@article{hurlbert1984pseudoreplication,
  title={Pseudoreplication and the design of ecological field experiments},
  author={Hurlbert, Stuart H},
  journal={Ecological monographs},
  volume={54},
  number={2},
  pages={187--211},
  year={1984}
}

@article{lazic2010pseudoreplication,
  title={The problem of pseudoreplication in neuroscientific studies: is it affecting your analysis?},
  author={Lazic, Stanley E},
  journal={BMC neuroscience},
  volume={11},
  number={5},
  year={2010}
}

@article{student1908probable,
  title={The probable error of a mean},
  author={Student},
  journal={Biometrika},
  volume={6},
  number={1},
  pages={1--25},
  year={1908}
}

@book{meeker2017statistical,
  title={Statistical Intervals: A Guide for Practitioners and Researchers},
  author={Meeker, William Q and Hahn, Gerald J and Escobar, Luis A},
  edition={2},
  year={2017},
  publisher={Wiley}
}

@book{quinn2002experimental,
  title={Experimental Design and Data Analysis for Biologists},
  author={Quinn, Gerry P and Keough, Michael J},
  year={2002},
  publisher={Cambridge University Press}
}

@article{cumming2014new,
  title={The new statistics: Why and how},
  author={Cumming, Geoff},
  journal={Psychological science},
  volume={25},
  number={1},
  pages={7--29},
  year={2014}
}

@inproceedings{ghorbani2019data,
  title={Data Shapley: Equitable valuation of data for machine learning},
  author={Ghorbani, Amirata and Zou, James},
  booktitle={International Conference on Machine Learning},
  pages={2242--2251},
  year={2019}
}

@inproceedings{jia2019towards,
  title={Towards efficient data valuation based on the Shapley value},
  author={Jia, Ruoxi and Dao, David and Wang, Boxin and Hubis, Frances Ann and Hynes, Nick and G{\"u}rel, Nezihe Merve and Li, Bo and Zhang, Ce and Song, Dawn and Spanos, Costas J},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2019}
}
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
