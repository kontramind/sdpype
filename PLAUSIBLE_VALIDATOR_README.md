# Plausible Validator

**Rule-based validation for synthetic tabular data quality**

## Overview

The Plausible Validator checks whether synthetic data records are **plausible** (valid according to business rules), even if they don't exactly match population records.

This complements the DDR metric by identifying:
- Records that are **factually correct** (exact match in population) ✓
- Records that are **plausible** (follow validation rules) ✨
- Records that are **implausible** (violate validation rules) ✗

## Core Concepts

### Validation Rules

Three types of validation rules:

1. **Categorical Membership**: Value must be in allowed set
   - Example: `Province` must be one of `["ON", "QC", "BC", ...]`

2. **Range Checks**: Value must be within min/max bounds
   - Example: `Date Reported` must be between `2020-01-01` and `2023-12-31`
   - Works for dates and numeric values

3. **Combination Constraints**: Multi-column tuple must exist in population
   - Example: `(Health Region, Province, Province Name, Region ID)` must be a valid Canadian geography combination

### Rule Configuration

Rules are defined in YAML files for easy editing and reuse across datasets:

```yaml
metadata:
  dataset_name: canada_covid19
  version: "1.0"

rules:
  categorical:
    - column: Age Group
      allowed_values: ["0-19", "20-29", "30-39", ...]
      allow_null: false

  ranges:
    - column: Date Reported
      data_type: date
      min: "2020-01-01"
      max: "2023-12-31"

  combinations:
    - name: canadian_geography
      columns: [Health Region, Province Abbreviation, Province Name, Region ID]
      unique_tuples: [[tuple1], [tuple2], ...]
```

## Installation

Requires Python 3.8+ with dependencies:

```bash
pip install pandas pyyaml typer rich
```

Or use the existing sdpype environment (already has these).

## Usage

### 1. Generate Rules from Population

Auto-generate an initial rule file from your population data:

```bash
python plausible_validator.py generate-rules \
  --population experiments/data/processed/population.csv \
  --output validation_rules/my_dataset.yaml \
  --name my_dataset
```

**What it does:**
- Analyzes population data
- Detects categorical columns and extracts unique values
- Detects date/numeric columns and extracts min/max ranges
- Creates a YAML file with all auto-detected rules

**Output:**
```
🔍 Analyzing population data...
  ✓ Categorical: Age Group (10 unique values)
  ✓ Categorical: Case Status (3 unique values)
  ✓ Date range: Date Reported [2020-01-01 to 2023-12-31]
  ✓ Categorical: Province Abbreviation (13 unique values)
  ...
✅ Rules saved to: validation_rules/my_dataset.yaml

📝 Next steps:
   1. Review and edit the generated YAML file
   2. Add combination rules if needed
   3. Run validation
```

### 2. Add Combination Rules (Manual)

Edit the generated YAML to add multi-column constraints:

```yaml
rules:
  combinations:
    - name: canadian_geography
      description: "Valid Canadian province/region combinations"
      columns:
        - Health Region
        - Province Abbreviation
        - Province Name
        - Region ID
      unique_tuples: from_population  # Or manually list tuples
```

If using `from_population`, you'll need to provide the population CSV during validation.

### 3. Validate Synthetic Data

Run validation against your synthetic data:

```bash
python plausible_validator.py validate \
  --synthetic experiments/data/synthetic/synthetic_data.csv \
  --rules validation_rules/my_dataset.yaml \
  --population experiments/data/processed/population.csv \
  --output validation_results.json \
  --samples 5
```

**Options:**
- `--synthetic, -s`: Synthetic data CSV (required)
- `--rules, -r`: Rules YAML file (required)
- `--population, -p`: Population CSV (needed if rules use `from_population`)
- `--output, -o`: Save JSON report (optional)
- `--samples, -n`: Number of failing records to display (default: 5)

## Output

### Rich Console Output

```
════════════════════════════════════════════════════════════════════════════════
  PLAUSIBILITY VALIDATION
  Rule-based validation for synthetic data
════════════════════════════════════════════════════════════════════════════════

📂 Loading synthetic data from: synthetic_data.csv
   Loaded 5,000 rows × 9 columns

📋 Loading validation rules from: rules.yaml
   Loaded 5 categorical rules
   Loaded 1 range rules
   Loaded 1 combination rules

🔍 Validating synthetic data...
✓ Validation complete


                        📊 Validation Results Summary
╭─────────────────────┬───────┬────────────╮
│ Metric              │ Count │ Percentage │
├─────────────────────┼───────┼────────────┤
│ Total Records       │ 5,000 │    100.00% │
│ ✓ Passed Validation │ 4,200 │     84.00% │
│ ✗ Failed Validation │   800 │     16.00% │
╰─────────────────────┴───────┴────────────╯


                            📋 Failures by Rule
╭────────────────────────────────────┬──────────┬───────────╮
│ Rule                               │ Failures │ % of Total│
├────────────────────────────────────┼──────────┼───────────┤
│ combination:canadian_geography     │      650 │    13.00% │
│ categorical:Age Group              │      120 │     2.40% │
│ range:Date Reported                │       30 │     0.60% │
╰────────────────────────────────────┴──────────┴───────────╯


🔍 Sample Failing Records

Record #1 (Index: 42)
  Failed rules: combination:canadian_geography
    Age Group: 50-59
    Case Status: Active
    Province Abbreviation: XY
    Health Region: Invalid Region
    ...

Record #2 (Index: 103)
  Failed rules: categorical:Age Group
    Age Group: 150-200
    ...

  ... and 795 more failing records
```

### JSON Output

```json
{
  "summary": {
    "total_records": 5000,
    "passed_records": 4200,
    "failed_records": 800,
    "pass_rate": 84.0
  },
  "rule_failures": {
    "combination:canadian_geography": 650,
    "categorical:Age Group": 120,
    "range:Date Reported": 30
  },
  "failed_record_indices": [42, 103, 156, ...],
  "failure_details": {
    "42": ["combination:canadian_geography"],
    "103": ["categorical:Age Group"],
    "156": ["combination:canadian_geography", "range:Date Reported"]
  }
}
```

## Example Workflow

### Step 1: Generate Initial Rules

```bash
python plausible_validator.py generate-rules \
  -p experiments/data/processed/canada_covid19_population.csv \
  -o validation_rules/canada_covid19.yaml \
  --name canada_covid19
```

### Step 2: Edit YAML to Add Geography Rule

```yaml
# validation_rules/canada_covid19.yaml

rules:
  # ... auto-generated categorical and range rules ...

  combinations:
    - name: canadian_geography
      description: "Valid Canadian province/region/health region combinations"
      columns:
        - Health Region
        - Province Abbreviation
        - Province Name
        - Region ID
      unique_tuples: from_population
```

### Step 3: Run Validation

```bash
python plausible_validator.py validate \
  -s experiments/data/synthetic/synthetic_data.csv \
  -r validation_rules/canada_covid19.yaml \
  -p experiments/data/processed/canada_covid19_population.csv \
  -o validation_results.json \
  --samples 10
```

### Step 4: Use Results

The validation creates a boolean mask indicating which records are plausible:

```python
import json
import pandas as pd

# Load results
with open('validation_results.json') as f:
    results = json.load(f)

# Load synthetic data
synthetic = pd.read_csv('synthetic_data.csv')

# Filter to plausible records only
plausible_records = synthetic.drop(results['failed_record_indices'])

print(f"Plausible records: {len(plausible_records)} / {len(synthetic)}")
```

## Advanced Usage

### Manual Tuple Specification

Instead of `from_population`, you can manually list valid tuples:

```yaml
combinations:
  - name: valid_status_age_combos
    columns: [Case Status, Age Group]
    unique_tuples:
      - ["Active", "20-29"]
      - ["Active", "30-39"]
      - ["Resolved", "20-29"]
      - ["Resolved", "30-39"]
      - ["Fatal", "70-79"]
      - ["Fatal", "80+"]
```

### Multiple Rule Files

Create different rule sets for different validation levels:

```bash
# Strict validation
validation_rules/strict_rules.yaml

# Permissive validation (wider ranges, more values)
validation_rules/permissive_rules.yaml
```

### Programmatic Usage

Use as a library in your own code:

```python
from plausible_validator import RuleLoader, ValidationEngine
import pandas as pd

# Load rules
rules = RuleLoader.load('rules.yaml')

# Load data
synthetic = pd.read_csv('synthetic.csv')
population = pd.read_csv('population.csv')

# Validate
engine = ValidationEngine(rules, population)
result = engine.validate(synthetic)

# Access results
print(f"Pass rate: {result.pass_rate:.2f}%")
print(f"Failed records: {result.failed_records}")

# Get boolean mask
valid_mask = result.record_validation
plausible_data = synthetic[valid_mask]
```

## Integration with DDR Metric

The Plausible Validator complements the DDR metric:

```
Synthetic Record Categories:

┌─ Exact Match in Population ─────────────────────────┐
│  ✓ DDR (Desirable Diverse Records)                  │ ← Exact & novel & factual
│  ⚠ Training Copies                                  │ ← Exact training matches
└─────────────────────────────────────────────────────┘

┌─ Not Exact Match ───────────────────────────────────┐
│  ✨ PDR (Plausible Diverse Records)                  │ ← Valid by rules & novel
│     └─ Passes plausibility validation                │
│  ✗ Implausible Records                              │ ← Fails validation
│     └─ Violates one or more rules                   │
└─────────────────────────────────────────────────────┘

Combined Quality Metric:
  Good Records = DDR + PDR
  (Exact factual OR rule-based plausible)
```

**Future**: Create `extended_ddr_metric.py` that combines both tools for comprehensive evaluation.

## Rule Design Tips

### 1. Start Permissive, Then Tighten

Generate rules automatically, test validation, then tighten constraints:

```yaml
# First iteration: auto-generated
- column: Age Group
  allowed_values: [auto-detected from population]

# After review: add business constraint
- column: Age Group
  allowed_values: ["0-19", "20-29", ..., "80+"]  # Remove invalid values
```

### 2. Balance Strictness vs. Utility

Too strict = reject useful synthetic data
Too permissive = accept implausible data

Find the balance for your use case.

### 3. Document Your Rules

```yaml
rules:
  categorical:
    - column: Province Abbreviation
      allowed_values: ["ON", "QC", "BC", ...]
      allow_null: false
      # RATIONALE: Canadian provinces only, no international data
```

### 4. Version Your Rules

```yaml
metadata:
  version: "2.1"  # Increment when rules change
  changelog:
    - "2.1: Added geography combination constraint"
    - "2.0: Tightened date range to 2020-2023"
    - "1.0: Initial release"
```

## Troubleshooting

### "Column not found in synthetic data"

Check that your synthetic data has the same column names as the rules.

### Large combination rule files

If geography has 1000+ unique tuples, the YAML gets large. This is OK - YAML handles it.

```yaml
unique_tuples:  # Can have hundreds or thousands of tuples
  - ["Toronto", "ON", "Ontario", 3595]
  - ["Montreal", "QC", "Quebec", 2462]
  # ... 998 more ...
```

### Rule generation is slow

For very large population datasets (millions of rows), rule generation may take time. Consider sampling:

```python
population_sample = pd.read_csv('population.csv').sample(100000, random_state=42)
# Generate rules from sample, then manually verify
```

### Validation is slow

Combination rules with many columns can be slow. Optimize by:
- Using `from_population` extraction (done once)
- Indexing if using programmatically
- Running on smaller batches

## File Organization

Recommended structure:

```
project/
├── data/
│   ├── processed/
│   │   ├── population.csv
│   │   └── training.csv
│   └── synthetic/
│       └── synthetic_data.csv
├── validation_rules/
│   ├── dataset_v1.yaml
│   └── dataset_v2.yaml
├── validation_results/
│   ├── run_2024_10_30.json
│   └── run_2024_10_31.json
├── ddr_metric.py
└── plausible_validator.py
```

## FAQ

**Q: What's the difference between DDR and PDR?**

- **DDR**: Exact matches in population (hash-based equality)
- **PDR**: Plausible by rules but not exact (rule-based validation)

DDR is stricter, PDR is more flexible.

**Q: Should I use strict hallucination rate or plausibility validation?**

Both! They measure different things:
- Hallucination rate: factual correctness (exact)
- Plausibility: business rule compliance (approximate)

**Q: Can I use this for non-tabular data?**

Currently designed for tabular data (CSV/DataFrames). For images, text, etc., you'd need different validation approaches.

**Q: How do I handle dates with time components?**

The validator normalizes to date level. If you need timestamp precision:

```yaml
ranges:
  - column: timestamp
    data_type: numeric  # Treat as Unix timestamp
    min: 1577836800  # 2020-01-01 00:00:00
    max: 1704067199  # 2023-12-31 23:59:59
```

**Q: Can rules reference other columns?**

Not yet, but planned for future:

```yaml
conditional:
  - if: Case Status == "Fatal"
    then: Age Group in ["60-69", "70-79", "80+"]
```

## Contributing

To extend the validator:

1. Add new rule types in `ValidationEngine._validate_*` methods
2. Update `RuleGenerator` to detect new patterns
3. Update YAML schema documentation

## License

Same license as the parent SDPype project.

---

**Questions or issues?** Check the inline help: `python plausible_validator.py --help`
