# Post-Training Metrics Computation Guide

## Overview

The new `compute_metrics_cli.py` tool enables computing metrics on existing experiment folders **without re-running the DVC pipeline**. This decouples metrics computation from training, providing significant benefits:

### Benefits

1. **Faster Iteration** - Training is expensive, metrics are cheap
2. **Add New Metrics Anytime** - Apply new metrics to all past experiments
3. **Experiment with Configs** - Try different metric parameters
4. **Better Resource Usage** - Train on GPU cluster, compute metrics locally
5. **Easier Debugging** - Metric bugs don't require re-training
6. **Cleaner Pipeline** - DVC focuses on data generation, not evaluation

## Architecture Change

### Before (Metrics in DVC Pipeline):
```
encode → train → generate → [statistical] → [detection] → [hallucination]
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                             If metrics fail, must re-run everything!
```

### After (Post-Training Metrics):
```
DVC Pipeline: encode → train → generate ✓ (Save data/models)
Later:        python compute_metrics_cli.py --folder <exp>
```

## Installation

The tool is already in your repo:
```bash
cd ~/development/sdpype  # or wherever you cloned the repo
chmod +x compute_metrics_cli.py  # Make executable (already done)
```

## Usage

### Basic Examples

#### 1. Compute all metrics on one experiment folder:
```bash
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/
```

#### 2. Compute specific metrics only:
```bash
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --metrics statistical detection
```

#### 3. Compute metrics for a specific generation:
```bash
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --generation 5
```

#### 4. Batch process multiple folders:
```bash
python compute_metrics_cli.py \
    --folders-pattern "./mimic_iii_*dseed233*/"
```

#### 5. Force recompute existing metrics:
```bash
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --force
```

#### 6. Use custom config:
```bash
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --config custom_params.yaml
```

### Command-Line Options

```
Options:
  --folder PATH              Experiment folder path
  --folders-pattern PATTERN  Glob pattern to match multiple folders
  --generation N             Specific generation to compute (default: all)
  --metrics {statistical,detection,hallucination,all}
                            Which metrics to compute (default: all)
  --config PATH              Custom params.yaml config file (optional)
  --force                    Overwrite existing metrics
  --metadata PATH            Path to metadata.json (auto-discovered if not specified)
  -h, --help                Show help message
```

## How It Works

### 1. **Auto-Discovery**
   - Scans `data/synthetic/` for generation files
   - Extracts model IDs from filenames
   - Parses: `library_modeltype_refhash_roothash_trnhash_gen_N_cfghash_seed`

### 2. **Data Loading**
   - Loads data from standard locations:
     - `data/decoded/` - Decoded (original format) data
     - `data/encoded/` - Encoded (numeric) data
     - `data/binned/` - Binned data for hallucination metrics
   - Uses same metadata as pipeline

### 3. **Config Loading**
   - Automatically loads config from `checkpoints/params_*.yaml`
   - Or uses custom config with `--config`
   - Falls back to sensible defaults if not found

### 4. **Metrics Computation**
   - **Statistical**: Alpha Precision, PRDC, Wasserstein, MMD, KS, TV, etc.
   - **Detection**: GMM, XGBoost, MLP, Linear classifiers
   - **Hallucination**: DDR, Plausibility, Training Copies

### 5. **Output**
   - Saves metrics as JSON: `metrics/{metric_type}_{model_id}.json`
   - Generates reports as TXT: `metrics/{metric_type}_report_{model_id}.txt`
   - Same format as DVC pipeline output

## Expected Folder Structure

Your experiment folder should have this structure:
```
mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/
├── checkpoints/
│   ├── params_*.yaml          # Config for each generation
│   └── checkpoint_*.json
├── data/
│   ├── decoded/               # Original format data
│   │   ├── reference_*.csv
│   │   ├── synthetic_*.csv
│   │   └── training_*.csv
│   ├── encoded/               # Numeric encoded data
│   │   ├── reference_*.csv
│   │   ├── synthetic_*.csv
│   │   └── training_*.csv
│   ├── binned/                # Binned for hallucination
│   │   ├── population_data_for_hallucinations.csv
│   │   └── training_data_for_hallucinations.csv
│   └── synthetic/             # Main synthetic data
│       ├── synthetic_data_*_decoded.csv
│       └── synthetic_data_*_encoded.csv
├── metrics/                   # Metrics output (created by tool)
│   ├── statistical_similarity_*.json
│   ├── detection_evaluation_*.json
│   └── hallucination_*.json
└── models/                    # Trained models (not used by metrics)
    └── *.pkl
```

## Testing Your Installation

### Test on an existing experiment folder:

```bash
# 1. Navigate to your sdpype repo
cd ~/development/sdpype

# 2. Test on one generation
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --generation 0 \
    --metrics statistical

# 3. Check output
ls -la ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/metrics/
```

Expected output:
```
statistical_similarity_sdv_gaussiancopula_906d6c18_906d6c18_906d6c18_gen_0_mimic_iii_baseline_a7ca9863_24157817.json
statistical_report_sdv_gaussiancopula_906d6c18_906d6c18_906d6c18_gen_0_mimic_iii_baseline_a7ca9863_24157817.txt
```

## Integration with Existing Workflow

### Current DVC Pipeline
You can keep using the DVC pipeline as-is, or:

### Option 1: Remove Metrics from DVC (Recommended)
Edit `dvc.yaml` and comment out or remove:
```yaml
# Comment out these stages:
# statistical_similarity:
#   ...
# detection_evaluation:
#   ...
# hallucination_evaluation:
#   ...
```

This makes the pipeline faster (only training/generation).

### Option 2: Hybrid Approach
- Keep metrics in DVC for initial runs
- Use CLI tool for:
  - Adding new metrics later
  - Re-computing with different configs
  - Experimenting with metric parameters

## Advanced Usage

### Batch Processing with Custom Filters

Process all experiments with specific data seed:
```bash
python compute_metrics_cli.py \
    --folders-pattern "./mimic_iii_*dseed233*/" \
    --metrics hallucination
```

Process all experiments with specific model:
```bash
python compute_metrics_cli.py \
    --folders-pattern "./mimic_iii_*_sdv_gaussiancopula_*/" \
    --generation 0
```

### Custom Metric Configuration

Create `custom_metrics.yaml`:
```yaml
evaluation:
  statistical_similarity:
    metrics:
      - name: alpha_precision
        parameters:
          k: 5
      - name: prdc_score
        parameters:
          k: 10
      - name: wasserstein_distance

  detection_evaluation:
    methods:
      - name: detection_xgb
        parameters:
          n_estimators: 100
    common_params:
      n_folds: 10
      random_state: 42

  hallucination:
    query_file: queries/validation.sql
```

Then use:
```bash
python compute_metrics_cli.py \
    --folder <experiment_folder> \
    --config custom_metrics.yaml \
    --force
```

## Troubleshooting

### Issue: "Could not find metadata.json"
**Solution:** Specify metadata path explicitly:
```bash
python compute_metrics_cli.py \
    --folder <experiment_folder> \
    --metadata ./data/metadata.json
```

### Issue: "Missing required data files"
**Solution:** Ensure your experiment folder has all necessary CSV files. Check:
- `data/decoded/reference_*.csv`
- `data/synthetic/synthetic_data_*_decoded.csv`
- `data/encoded/` (for detection metrics)
- `data/binned/population_data_for_hallucinations.csv` (for hallucination)

### Issue: "Query file not found"
**Solution:** For hallucination metrics, ensure `queries/validation.sql` exists or specify in config:
```yaml
evaluation:
  hallucination:
    query_file: /path/to/your/validation.sql
```

### Issue: Metrics computation is slow
**Solution:**
- Compute specific metrics: `--metrics statistical`
- Compute specific generation: `--generation 0`
- Run in parallel for multiple folders (use separate terminals)

## Next Steps

1. **Test the tool** on your existing experiment folders
2. **Compare results** with DVC-generated metrics (should be identical)
3. **Consider removing metrics from DVC** to speed up pipeline
4. **Add new metrics** to the registry and apply to old experiments

## Support

If you encounter issues:
1. Check folder structure matches expected format
2. Verify all required CSV files exist
3. Check metadata.json is accessible
4. Look for detailed error messages in the output

For bugs or feature requests, please open a GitHub issue.
