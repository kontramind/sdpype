# Data Sources for Post-Training Metrics Tool

## Current Implementation - Data Source Map

### 1. **Metrics Configuration** (What metrics to compute + parameters)

**Source Priority:**
```
1. Custom config (--config flag)
   python compute_metrics_cli.py --config my_params.yaml
   ↓
2. Experiment folder checkpoint params
   <folder>/checkpoints/params_*_gen_{N}.yaml
   ↓
3. Hardcoded defaults in compute_metrics_cli.py
```

**What it contains:**
```yaml
evaluation:
  statistical_similarity:
    metrics:
      - name: alpha_precision
      - name: prdc_score
      # ... etc

  detection_evaluation:
    common_params: {n_folds: 5, random_state: 51}
    methods:
      - name: detection_gmm
      # ... etc

  hallucination:
    query_file: queries/validation.sql
```

---

### 2. **Encoding Configuration** (How data was encoded)

**Source:**
```
From config: config['encoding']['config_file']

Example: experiments/configs/encoding/canada_covid19_case_details_variant2.yaml
```

**⚠️ ISSUE:** This path is **relative to REPO ROOT**, not experiment folder!

**Current workaround:** The tool loads this from the config, but the path might be wrong if running from different directory.

---

### 3. **Data Files** (Reference, Training, Population, Synthetic)

#### A. **Reference Data** (evaluation dataset)

**Source (constructed from model_id):**
```python
# Decoded
<folder>/data/decoded/reference_{model_id}.csv

# Encoded
<folder>/data/encoded/reference_{model_id}.csv
```

**✓ This works** - Files are in experiment folder

---

#### B. **Training Data**

**Source (constructed from model_id):**
```python
# Decoded
<folder>/data/decoded/training_{model_id}.csv

# Encoded
<folder>/data/encoded/training_{model_id}.csv
```

**✓ This works** - Files are in experiment folder

---

#### C. **Population Data** (for hallucination metrics)

**Source:**
```python
<folder>/data/binned/population_data_for_hallucinations.csv
```

**✓ This works** - Based on your folder structure, this file exists!

**Note:** In params.yaml, original path is:
```yaml
data:
  population_file: experiments/data/processed/canada_covid19_case_details_population_variant2.csv
```

But the DVC pipeline copies/bins it to the experiment folder as `population_data_for_hallucinations.csv`.

---

#### D. **Synthetic Data**

**Source (constructed from model_id):**
```python
# Decoded (main file for discovery)
<folder>/data/synthetic/synthetic_data_{model_id}_decoded.csv

# Encoded
<folder>/data/synthetic/synthetic_data_{model_id}_encoded.csv
```

**✓ This works** - Files are in experiment folder

---

### 4. **Metadata.json** (SDV SingleTableMetadata)

**Auto-discovery order:**
```python
1. <folder>/metadata.json
2. <folder>/data/metadata.json
3. data/metadata.json           # Repo root
4. metadata.json                # Current directory
```

**Or specify explicitly:**
```bash
--metadata ./data/metadata.json
```

**In params.yaml:**
```yaml
data:
  metadata_file: experiments/data/processed/canada_covid19_case_details_variant2_encoded_metadata.json
```

**⚠️ ISSUE:** Params.yaml path is relative to repo root. Tool auto-discovery might find a different file!

---

### 5. **Query File** (for hallucination metrics)

**Source:**
```python
From config: config['evaluation']['hallucination']['query_file']

Example: queries/validation.sql
```

**⚠️ ISSUE:** This path is **relative to REPO ROOT**, not experiment folder!

---

## Summary Table

| Data Type | Expected Location | Path Type | Works? | Issue |
|-----------|------------------|-----------|---------|-------|
| **Metrics config** | checkpoints/params_*.yaml OR --config | Relative to folder | ✓ | Falls back to defaults if not found |
| **Encoding config** | From config path | Relative to REPO | ⚠️ | Path might be wrong if running from different dir |
| **Reference data** | data/decoded/ & data/encoded/ | Relative to folder | ✓ | Constructed from model_id |
| **Training data** | data/decoded/ & data/encoded/ | Relative to folder | ✓ | Constructed from model_id |
| **Population data** | data/binned/population_data_for_hallucinations.csv | Relative to folder | ✓ | Hardcoded filename |
| **Synthetic data** | data/synthetic/ | Relative to folder | ✓ | Constructed from model_id |
| **Metadata.json** | Auto-discovered OR --metadata | Various | ⚠️ | Auto-discovery might find wrong file |
| **Query file** | From config path | Relative to REPO | ⚠️ | Path might be wrong if running from different dir |

---

## Recommended Usage

### Option 1: Use Default Repo Config (Recommended)

**Run from repo root:**
```bash
cd ~/development/sdpype

python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --config params.yaml \
    --metadata data/metadata.json
```

This ensures all paths in params.yaml are resolved correctly.

---

### Option 2: Use Experiment Folder Config

**If checkpoints/params_*.yaml exists:**
```bash
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --metadata data/metadata.json
```

Tool will load config from checkpoints automatically.

---

### Option 3: Specify Everything Explicitly

```bash
python compute_metrics_cli.py \
    --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/ \
    --metadata ./data/your_metadata.json \
    --config custom_config.yaml
```

---

## Files That MUST Exist in Experiment Folder

Based on your folder structure, these should already be there:

```
<experiment_folder>/
├── data/
│   ├── binned/
│   │   └── population_data_for_hallucinations.csv  ← Required for hallucination
│   ├── decoded/
│   │   ├── reference_*.csv                          ← Required for all metrics
│   │   ├── training_*.csv                           ← Required for hallucination
│   │   └── synthetic_*.csv                          ← Required for all metrics
│   ├── encoded/
│   │   ├── reference_*.csv                          ← Required for detection
│   │   └── synthetic_*.csv                          ← Required for detection
│   └── synthetic/
│       ├── synthetic_data_*_decoded.csv             ← Used for discovery
│       └── synthetic_data_*_encoded.csv
└── checkpoints/
    └── params_*_gen_{N}.yaml                        ← Optional config source
```

---

## Files That Must Be Accessible from Repo Root

These are specified in params.yaml with repo-relative paths:

```
sdpype/  (repo root)
├── params.yaml                                      ← Metrics config
├── queries/
│   └── validation.sql                               ← Hallucination queries
├── data/
│   └── metadata.json                                ← Or specify path
└── experiments/
    └── configs/
        └── encoding/
            └── *.yaml                               ← Encoding config
```

---

## Quick Diagnostic

To check what the tool will find, run this from repo root:

```bash
cd ~/development/sdpype

# Check if all required files exist
ls -la ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/data/binned/population_data_for_hallucinations.csv
ls -la ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/data/decoded/reference_*.csv
ls -la ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/data/synthetic/synthetic_data_*_decoded.csv
ls -la ./queries/validation.sql
ls -la ./params.yaml
```

If any are missing, the tool will fail with a clear error message.
