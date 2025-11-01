# DDR Metric Tool

**Desirable Diverse Records Rate (DDR)** - A metric for evaluating synthetic tabular data quality

## Overview

The DDR metric measures the "sweet spot" in synthetic data generation: records that are **both factual AND novel**.

Based on the paper: _"Magnitude and Impact of Hallucinations in Tabular Synthetic Health Data"_

## The Problem

Traditional hallucination rate (HR) doesn't distinguish between:
- ❌ Synthetic records that **copy training data** (privacy risk)
- ✅ Synthetic records that **match other population records** (good - diverse & factual)

## The Solution: DDR Metric

**DDR = |(S ∩ P) \ T| / |S|**

Where:
- **S** = Synthetic dataset (generated data)
- **P** = Population dataset (ground truth)
- **T** = Training dataset (subset of P used to train the generator)

DDR measures the proportion of synthetic records that are:
1. **Factual**: Exist in the population (S ∩ P)
2. **Novel**: NOT copied from training data (\ T)

## Key Metrics Computed

The tool computes metrics using **two perspectives**:

1. **Unique-based**: Metrics based on distinct records only (duplicates counted once)
2. **Total-based**: Metrics including all duplicates as they were generated

| Metric | Formula | Interpretation | Goal |
|--------|---------|----------------|------|
| **Duplicate Rate** | (Total - Unique) / Total | Proportion of duplicate rows | Track quality |
| **DDR** | \|(S ∩ P) \ T\| / \|S\| | Factual AND novel records | **MAXIMIZE** |
| **Training Copy Rate** | \|S ∩ T\| / \|S\| | Exact training copies | **MINIMIZE** |
| **Hallucination Rate** | \|S \ P\| / \|S\| | Fabricated records | **MINIMIZE** |
| **Population Match Rate** | \|S ∩ P\| / \|S\| | Factual (includes copies) | High is good |

**Key Relationship:**
```
100% = DDR Rate + Training Copy Rate + Hallucination Rate
```

### Why Dual Perspective?

Duplicates matter! If a generator creates 100 copies of the same training record, that's a bigger privacy risk than creating it once. The dual approach shows:
- **Unique metrics**: The diversity of generated patterns
- **Total metrics**: The actual distribution as generated (privacy & utility)

## Installation

Requires Python 3.8+ with:
```bash
pip install pandas typer rich
```

Or use the existing project environment:
```bash
# Already installed if you have sdpype dependencies
```

## Usage

### Basic Usage

```bash
python ddr_metric.py evaluate \
  --population experiments/data/processed/population.csv \
  --training experiments/data/processed/train.csv \
  --synthetic experiments/data/synthetic/synthetic_data.csv
```

### With All Options

```bash
python ddr_metric.py evaluate \
  --population path/to/population.csv \
  --training path/to/training.csv \
  --synthetic path/to/synthetic.csv \
  --samples 5 \
  --seed 42 \
  --formula
```

### Using Short Flags

```bash
python ddr_metric.py evaluate -p population.csv -t train.csv -s synthetic.csv
```

### Show Formulas Only

```bash
python ddr_metric.py formula
```

## Command Line Options

### `evaluate` command

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--population` | `-p` | Path to population CSV (ground truth) | `experiments/data/processed/canada_covid19_case_details_population.csv` |
| `--training` | `-t` | Path to training CSV (subset of population) | `experiments/data/processed/canada_covid19_case_details_train.csv` |
| `--synthetic` | `-s` | Path to synthetic CSV (generated data) | `experiments/data/synthetic/synthetic_data_*.csv` |
| `--samples` | `-n` | Number of sample records to show per category | `3` |
| `--seed` | | Random seed for sampling | `42` |
| `--formula` | `-f` | Show mathematical formula explanation | `False` |
| `--no-viz` | | Skip sample record visualization | `False` |

### `formula` command

Display mathematical formulas and explanations without running evaluation:

```bash
python ddr_metric.py formula
```

## Output Examples

### Duplicate Analysis

```
📦 Duplicate Analysis
  Total Generated Rows:     5,000
  Unique Records:           4,860 (97.20%)
  Duplicate Records:          140 (2.80%)

🔄 Duplicate Breakdown by Category
  ✓ DDR duplicates:          12 unique records → 24 duplicate rows
  ⚠ Training copy duplicates: 8 unique records → 16 duplicate rows
  ✗ Hallucination duplicates: 95 unique records → 100 duplicate rows
  Most duplicated: ✗ Hallucination record appears 5 times
```

### Dual Perspective Metrics Table

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│              📊 Synthetic Data Quality Metrics - Dual Perspective                  │
├──────────────────────┬──────────┬────────────┬──────────┬────────────┬────────────...│
│ Metric               │ Unique   │ Unique     │ Total    │ Total      │ Interpret...│
│                      │ Count    │ Rate       │ Count    │ Rate       │              │
├──────────────────────┼──────────┼────────────┼──────────┼────────────┼────────────...│
│ Total Synthetic      │    4,860 │    100.00% │    5,000 │    100.00% │ All records  │
├──────────────────────┼──────────┼────────────┼──────────┼────────────┼────────────...│
│ ✓ DDR                │       74 │      1.52% │       86 │      1.72% │ IDEAL        │
├──────────────────────┼──────────┼────────────┼──────────┼────────────┼────────────...│
│ ⚠ Training Copies    │       92 │      1.89% │      108 │      2.16% │ Privacy risk │
│ ✗ Hallucinations     │    4,694 │     96.58% │    4,806 │     96.12% │ Fabricated   │
├──────────────────────┼──────────┼────────────┼──────────┼────────────┼────────────...│
│ Population Matches   │      166 │      3.42% │      194 │      3.88% │ Factual      │
└──────────────────────┴──────────┴────────────┴──────────┴────────────┴────────────...┘

Interpretation:
  • Unique Count/Rate: Metrics based on distinct records only
  • Total Count/Rate:  Metrics including all duplicates (as generated)
```

### Quality Interpretation

- **DDR ≥ 70%**: 🟢 EXCELLENT - High factual diversity
- **DDR ≥ 50%**: 🟡 GOOD - Acceptable quality
- **DDR ≥ 30%**: 🟠 MODERATE - Needs improvement
- **DDR < 30%**: 🔴 POOR - Significant issues

## Sample Visualizations

The tool displays sample records from each category:

### ✓ DDR Records (Factual & Novel)
Shows synthetic records that exist in population but NOT in training data.

### ⚠ Training Copies
Shows synthetic records that exactly match training data (privacy concern).

### ✗ Hallucinations
Shows synthetic records that don't exist anywhere in the population (fabricated).

Each visualization shows:
- All column values
- Color-coded differences (for matches)
- Clear category labels

## Understanding the Results

### Ideal Scenario (Unique vs Total)
```
                        Unique    Total
DDR Rate:               85%       82%     ← High diversity, some good duplicates
Training Copy Rate:      5%        6%     ← Low privacy risk
Hallucination Rate:     10%       12%     ← Acceptable quality
Duplicate Rate:          -        3%      ← Low duplication
```

### Privacy Concern - Training Memorization
```
                        Unique    Total
DDR Rate:               30%       25%     ← Low
Training Copy Rate:     60%       70%     ← HIGH - memorizing AND duplicating!
Hallucination Rate:     10%        5%
Duplicate Rate:          -        15%     ← Duplicating training copies!
```
**Issue**: Generator is both memorizing training data AND creating duplicates of those copies.

### Quality Concern - Fabrication
```
                        Unique    Total
DDR Rate:               40%       38%     ← Moderate
Training Copy Rate:      5%        5%     ← Good
Hallucination Rate:     55%       57%     ← HIGH - fabricating too much!
Duplicate Rate:          -        5%
```
**Issue**: Generator creates many fabricated records, some appearing multiple times.

### Duplication Concern
```
                        Unique    Total
DDR Rate:               70%       50%     ← Drops significantly!
Training Copy Rate:      5%       10%     ← Doubles with duplicates
Hallucination Rate:     25%       40%     ← Increases
Duplicate Rate:          -        30%     ← Very HIGH duplication
```
**Issue**: High duplicate rate changes the picture dramatically. Many records appear multiple times, skewing the distribution.

## Implementation Details

### Hash-Based Comparison
- Uses `pd.util.hash_pandas_object()` for fast row-level comparison
- Converts all values to strings to avoid dtype issues
- Efficient set operations for large datasets

### Column Alignment
- Automatically checks for column mismatches
- Sorts columns alphabetically for consistent comparison
- Clear error messages if schemas don't match

### Sanity Checks
- Verifies that DDR + Copies + Hallucinations = Total
- Reports any computation inconsistencies

## Example Workflow

```bash
# 1. Check the formula explanation first
python ddr_metric.py formula

# 2. Run evaluation with defaults
python ddr_metric.py evaluate

# 3. Run with custom datasets and more samples
python ddr_metric.py evaluate \
  -p my_population.csv \
  -t my_training.csv \
  -s my_synthetic.csv \
  --samples 10 \
  --formula

# 4. Quick check without visualizations
python ddr_metric.py evaluate --no-viz
```

## File Requirements

### CSV Format
All three files (population, training, synthetic) must:
- Have the **same columns** (names and order don't matter)
- Be readable by pandas
- Contain the data types that can be converted to strings

### Training Subset Requirement
⚠️ **Important**: The training dataset MUST be a true subset of the population dataset. Otherwise, the metrics won't make sense.

Typical data split:
```
Population (100%)
├── Training (70-80%)  ← Used to train generator
└── Test (20-30%)      ← Held out
```

## Use Cases

### 1. Model Evaluation
Compare different synthetic data generators:
```bash
python ddr_metric.py evaluate -s model_a_output.csv > results_a.txt
python ddr_metric.py evaluate -s model_b_output.csv > results_b.txt
```

### 2. Hyperparameter Tuning
Track DDR across different model configurations to find the best settings.

### 3. Privacy Auditing
Monitor training copy rate to ensure synthetic data doesn't leak sensitive information.

### 4. Quality Assurance
Ensure hallucination rate stays within acceptable bounds for your domain.

## Troubleshooting

### "Column mismatch detected!"
- Ensure all three CSVs have identical column names
- Check for extra spaces or different capitalization

### "No records in this category"
- This is actually informative! e.g., "No hallucinations" is good news
- Indicates all synthetic records are factual

### Very low DDR
- Check if training set is too large (less room for novel records)
- Generator might be memorizing training data
- Consider data augmentation techniques

### High hallucination rate
- Generator is creating unrealistic combinations
- May need better conditioning or constraints
- Check if population data is representative

## Integration with SDPype

This tool is standalone but can be integrated into the SDPype evaluation pipeline:

```python
from ddr_metric import compute_ddr_metrics, display_metrics_table

# Use in your evaluation workflow
metrics = compute_ddr_metrics(pop_df, train_df, syn_df)
display_metrics_table(metrics)
```

## References

- Paper: "Magnitude and Impact of Hallucinations in Tabular Synthetic Health Data"
- Related work on privacy in synthetic data generation
- Hallucination metrics in generative models

## Contributing

To extend this tool:
1. Add new metrics to `compute_ddr_metrics()`
2. Update visualizations in `visualize_samples()`
3. Modify thresholds in quality interpretation

## License

Same license as the parent SDPype project.

---

**Questions?** Check the formula explanation with `python ddr_metric.py formula`
