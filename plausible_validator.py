#!/usr/bin/env python3
"""
Plausible Validator - Rule-based validation for synthetic tabular data

Validates synthetic data against configurable rules defined in YAML:
- Categorical membership (values must be in allowed set)
- Range checks (dates, numbers within population bounds)
- Combination constraints (multi-column tuples must exist in population)

Usage:
    # Generate rules from population data
    python plausible_validator.py generate-rules -p population.csv -o rules.yaml

    # Validate synthetic data
    python plausible_validator.py validate -s synthetic.csv -r rules.yaml
"""

import pandas as pd
import typer
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import json

# Import metadata utilities for type-safe loading
from sdpype.metadata import load_csv_with_metadata

console = Console()
app = typer.Typer(add_completion=False)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ValidationResult:
    """Results from validating synthetic data against rules."""
    total_records: int
    passed_records: int
    failed_records: int
    pass_rate: float
    record_validation: pd.Series  # Boolean series: True = valid, False = invalid
    failure_details: Dict[int, List[str]]  # {row_index: [failed_rule_names]}
    rule_failures: Dict[str, int]  # {rule_name: failure_count}


@dataclass
class ValidationRules:
    """Parsed validation rules from YAML."""
    metadata: Dict[str, Any]
    categorical_rules: List[Dict[str, Any]] = field(default_factory=list)
    range_rules: List[Dict[str, Any]] = field(default_factory=list)
    combination_rules: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Rule Generator
# ============================================================================

class RuleGenerator:
    """Generate validation rules from population data."""

    def __init__(self, population_df: pd.DataFrame, dataset_name: str = "dataset"):
        self.population = population_df
        self.dataset_name = dataset_name

    def generate_rules(self) -> Dict[str, Any]:
        """Generate complete rule set from population data."""

        console.print("🔍 Analyzing population data...", style="bold blue")

        rules = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "version": "1.0",
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_population_records": len(self.population),
                "description": f"Auto-generated validation rules for {self.dataset_name}"
            },
            "rules": {
                "categorical": [],
                "ranges": [],
                "combinations": []
            }
        }

        # Detect categorical columns (object/string types with reasonable cardinality)
        for col in self.population.columns:
            dtype = self.population[col].dtype
            n_unique = self.population[col].nunique()
            has_nulls = self.population[col].isna().any()

            # Categorical: object type or low cardinality
            if dtype == 'object' or (n_unique < 100 and dtype != 'datetime64[ns]'):
                unique_vals = self.population[col].dropna().unique().tolist()

                # Convert numpy types to Python native types for YAML serialization
                unique_vals = [str(v) if not pd.isna(v) else None for v in unique_vals]
                unique_vals = sorted([v for v in unique_vals if v is not None])

                rules["rules"]["categorical"].append({
                    "column": col,
                    "allowed_values": unique_vals,
                    "allow_null": bool(has_nulls),
                    "cardinality": len(unique_vals)
                })

                console.print(f"  ✓ Categorical: {col} ({len(unique_vals)} unique values)")

            # Date ranges
            elif pd.api.types.is_datetime64_any_dtype(self.population[col]):
                min_date = self.population[col].min()
                max_date = self.population[col].max()

                rules["rules"]["ranges"].append({
                    "column": col,
                    "data_type": "date",
                    "min": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
                    "max": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None,
                })

                console.print(f"  ✓ Date range: {col} [{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}]")

            # Numeric ranges
            elif pd.api.types.is_numeric_dtype(self.population[col]):
                min_val = float(self.population[col].min())
                max_val = float(self.population[col].max())

                rules["rules"]["ranges"].append({
                    "column": col,
                    "data_type": "numeric",
                    "min": min_val,
                    "max": max_val,
                })

                console.print(f"  ✓ Numeric range: {col} [{min_val} to {max_val}]")

        console.print("\n💡 To add combination rules, manually edit the YAML:", style="bold yellow")
        console.print("   See 'combinations' section in the generated file\n")

        return rules

    def add_combination_rule_template(self, rules: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        """Add a template for combination rules (to be manually configured)."""

        rules["rules"]["combinations"].append({
            "name": "example_combination",
            "description": "Example: multi-column combination constraint",
            "columns": columns,
            "unique_tuples": "TODO: Add unique tuples or set to 'from_population'",
            "comment": "Remove this template and add your actual combination rules"
        })

        return rules


# ============================================================================
# Rule Loader
# ============================================================================

class RuleLoader:
    """Load and parse validation rules from YAML."""

    @staticmethod
    def load(rules_path: Path) -> ValidationRules:
        """Load rules from YAML file."""

        with open(rules_path, 'r') as f:
            rules_dict = yaml.safe_load(f)

        return ValidationRules(
            metadata=rules_dict.get("metadata", {}),
            categorical_rules=rules_dict.get("rules", {}).get("categorical", []),
            range_rules=rules_dict.get("rules", {}).get("ranges", []),
            combination_rules=rules_dict.get("rules", {}).get("combinations", [])
        )


# ============================================================================
# Validation Engine
# ============================================================================

class ValidationEngine:
    """Execute validation rules on synthetic data."""

    def __init__(self, rules: ValidationRules, population_df: Optional[pd.DataFrame] = None):
        self.rules = rules
        self.population = population_df

    def validate(self, synthetic_df: pd.DataFrame) -> ValidationResult:
        """Validate synthetic data against all rules."""

        console.print("🔍 Validating synthetic data...", style="bold blue")

        total_records = len(synthetic_df)

        # Initialize tracking
        record_valid = pd.Series([True] * total_records, index=synthetic_df.index)
        failure_details: Dict[int, List[str]] = {}
        rule_failures: Dict[str, int] = {}

        # Validate categorical rules
        for rule in self.rules.categorical_rules:
            valid_mask = self._validate_categorical(synthetic_df, rule)
            invalid_indices = synthetic_df.index[~valid_mask]

            rule_name = f"categorical:{rule['column']}"
            rule_failures[rule_name] = len(invalid_indices)

            for idx in invalid_indices:
                record_valid.loc[idx] = False
                if idx not in failure_details:
                    failure_details[idx] = []
                failure_details[idx].append(rule_name)

        # Validate range rules
        for rule in self.rules.range_rules:
            valid_mask = self._validate_range(synthetic_df, rule)
            invalid_indices = synthetic_df.index[~valid_mask]

            rule_name = f"range:{rule['column']}"
            rule_failures[rule_name] = len(invalid_indices)

            for idx in invalid_indices:
                record_valid.loc[idx] = False
                if idx not in failure_details:
                    failure_details[idx] = []
                failure_details[idx].append(rule_name)

        # Validate combination rules
        for rule in self.rules.combination_rules:
            valid_mask = self._validate_combination(synthetic_df, rule)
            invalid_indices = synthetic_df.index[~valid_mask]

            rule_name = f"combination:{rule['name']}"
            rule_failures[rule_name] = len(invalid_indices)

            for idx in invalid_indices:
                record_valid.loc[idx] = False
                if idx not in failure_details:
                    failure_details[idx] = []
                failure_details[idx].append(rule_name)

        # Compute summary
        passed = int(record_valid.sum())
        failed = total_records - passed
        pass_rate = (passed / total_records * 100) if total_records > 0 else 0.0

        console.print(f"✓ Validation complete\n", style="green")

        return ValidationResult(
            total_records=total_records,
            passed_records=passed,
            failed_records=failed,
            pass_rate=pass_rate,
            record_validation=record_valid,
            failure_details=failure_details,
            rule_failures=rule_failures
        )

    def _validate_categorical(self, df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
        """Validate categorical membership rule."""
        col = rule["column"]
        allowed_values = rule["allowed_values"]
        allow_null = rule.get("allow_null", False)

        if col not in df.columns:
            console.print(f"⚠ Warning: Column '{col}' not found in synthetic data", style="yellow")
            return pd.Series([False] * len(df), index=df.index)

        # Convert both data and allowed values to string for consistent comparison
        # This handles cases where CSV has numbers but YAML has strings (e.g., 591 vs '591')
        col_data = df[col]

        # Check if "Missing" is in allowed values (accepts both string "Missing" and NaN)
        has_missing_value = "Missing" in allowed_values

        # Convert allowed values to strings (keep "Missing" in the set!)
        allowed_values_str = set(
            str(v) for v in allowed_values
            if v is not None  # Only exclude None/null, keep "Missing" string
        )

        # Convert column data to string
        # NaN becomes the string "nan", actual "Missing" stays "Missing"
        col_as_str = col_data.astype(str)

        # Check membership (comparing strings to strings)
        valid = col_as_str.isin(allowed_values_str)

        # Handle NaN/null values: if data has NaN and "Missing" or nulls are allowed
        is_null = col_data.isna()

        if allow_null or has_missing_value:
            # Mark NaN values as valid if nulls or "Missing" are allowed
            valid = valid | is_null

        return valid

    def _validate_range(self, df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
        """Validate range rule (numeric or date)."""
        col = rule["column"]
        data_type = rule["data_type"]
        min_val = rule.get("min")
        max_val = rule.get("max")

        if col not in df.columns:
            console.print(f"⚠ Warning: Column '{col}' not found in synthetic data", style="yellow")
            return pd.Series([False] * len(df), index=df.index)

        valid = pd.Series([True] * len(df), index=df.index)

        if data_type == "date":
            # Convert to datetime
            col_data = pd.to_datetime(df[col], errors='coerce')
            min_date = pd.to_datetime(min_val) if min_val else None
            max_date = pd.to_datetime(max_val) if max_val else None

            if min_date is not None:
                valid = valid & (col_data >= min_date)
            if max_date is not None:
                valid = valid & (col_data <= max_date)

        elif data_type == "numeric":
            col_data = pd.to_numeric(df[col], errors='coerce')

            if min_val is not None:
                valid = valid & (col_data >= min_val)
            if max_val is not None:
                valid = valid & (col_data <= max_val)

        return valid

    def _validate_combination(self, df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
        """Validate multi-column combination rule."""
        columns = rule["columns"]
        unique_tuples = rule.get("unique_tuples", [])

        # Check if we need to extract from population
        if unique_tuples == "from_population":
            if self.population is None:
                console.print(f"⚠ Warning: Population data needed for combination rule '{rule['name']}'", style="yellow")
                return pd.Series([False] * len(df), index=df.index)

            # Extract unique combinations from population
            unique_tuples = self._extract_unique_combinations(self.population, columns)

        # Check all columns exist
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            console.print(f"⚠ Warning: Columns {missing_cols} not found in synthetic data", style="yellow")
            return pd.Series([False] * len(df), index=df.index)

        # Convert tuples to set for fast lookup
        # IMPORTANT: Convert all values to strings to avoid type mismatches
        if isinstance(unique_tuples, list) and len(unique_tuples) > 0:
            # Convert to tuples with string values
            valid_tuples = set()
            for t in unique_tuples:
                if isinstance(t, list):
                    # Convert list elements to strings
                    valid_tuples.add(tuple(str(v) for v in t))
                else:
                    # Already a tuple, convert elements to strings
                    valid_tuples.add(tuple(str(v) for v in t))
        else:
            valid_tuples = set()

        # Check each row's combination (convert values to strings)
        def is_valid_combination(row):
            # Convert all values in the combination to strings
            combo = tuple(str(v) for v in row[columns].values)
            return combo in valid_tuples

        valid = df.apply(is_valid_combination, axis=1)

        return valid

    def _extract_unique_combinations(self, df: pd.DataFrame, columns: List[str]) -> List[Tuple]:
        """Extract unique combinations of columns from dataframe.

        Converts all values to strings to ensure consistent comparison.
        """
        # Get unique combinations and convert all values to strings
        unique_combos = df[columns].drop_duplicates()

        # Convert to list of tuples with string values
        result = []
        for _, row in unique_combos.iterrows():
            # Convert each value to string
            string_tuple = tuple(str(v) for v in row.values)
            result.append(string_tuple)

        return result


# ============================================================================
# Output Formatters
# ============================================================================

def _extract_failed_columns(rules: ValidationRules, failed_rule_names: List[str]) -> Set[str]:
    """Extract column names involved in failed rules."""
    failed_columns = set()

    for rule_name in failed_rule_names:
        # Parse rule name format: "type:column" or "type:name"
        if ":" in rule_name:
            rule_type, rule_identifier = rule_name.split(":", 1)

            if rule_type == "categorical":
                # categorical:Column Name
                failed_columns.add(rule_identifier)
            elif rule_type == "range":
                # range:Column Name
                failed_columns.add(rule_identifier)
            elif rule_type == "combination":
                # combination:rule_name - need to find the columns from rules
                for combo_rule in rules.combination_rules:
                    if combo_rule.get("name") == rule_identifier:
                        failed_columns.update(combo_rule.get("columns", []))

    return failed_columns


def display_validation_results(
    result: ValidationResult,
    synthetic_df: pd.DataFrame,
    rules: ValidationRules,
    n_samples: int = 5,
    no_visualization: bool = False
):
    """Display validation results in Rich format."""

    # Summary table
    summary_table = Table(
        title="📊 Validation Results Summary",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )

    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", justify="right", style="yellow")
    summary_table.add_column("Percentage", justify="right", style="green")

    summary_table.add_row(
        "Total Records",
        f"{result.total_records:,}",
        "100.00%"
    )
    summary_table.add_row(
        "✓ Passed Validation",
        f"{result.passed_records:,}",
        f"{result.pass_rate:.2f}%"
    )
    summary_table.add_row(
        "✗ Failed Validation",
        f"{result.failed_records:,}",
        f"{100 - result.pass_rate:.2f}%"
    )

    console.print()
    console.print(summary_table)
    console.print()

    # Rule failures table
    if result.rule_failures:
        rule_table = Table(
            title="📋 Failures by Rule",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )

        rule_table.add_column("Rule", style="cyan")
        rule_table.add_column("Failures", justify="right", style="red")
        rule_table.add_column("% of Total", justify="right", style="yellow")

        for rule_name, count in sorted(result.rule_failures.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = (count / result.total_records * 100)
                rule_table.add_row(rule_name, f"{count:,}", f"{pct:.2f}%")

        console.print(rule_table)
        console.print()

    # Sample failing records
    if result.failed_records > 0 and n_samples > 0 and not no_visualization:
        console.print("🔍 Sample Failing Records", style="bold blue")

        failed_indices = [idx for idx, valid in result.record_validation.items() if not valid]
        sample_indices = failed_indices[:n_samples]

        for i, idx in enumerate(sample_indices, start=1):
            failed_rules = result.failure_details.get(idx, [])

            # Extract which columns are involved in failures
            failed_columns = _extract_failed_columns(rules, failed_rules)

            console.print(f"\n[bold red]Record #{i}[/bold red] (Index: {idx})")
            console.print(f"  Failed rules: {', '.join(failed_rules)}")

            # Show record data with highlighting
            record_data = synthetic_df.loc[idx].to_dict()
            for col, val in record_data.items():  # Show all columns
                if col in failed_columns:
                    # Highlight failed fields in bold red with marker
                    console.print(f"    {col}: [bold red]{val} ← FAILED[/bold red]")
                else:
                    console.print(f"    {col}: [yellow]{val}[/yellow]")

        if len(failed_indices) > n_samples:
            console.print(f"\n  ... and {len(failed_indices) - n_samples} more failing records")

        console.print()


def save_validation_json(result: ValidationResult, output_path: Path):
    """Save validation results to JSON."""

    output = {
        "summary": {
            "total_records": result.total_records,
            "passed_records": result.passed_records,
            "failed_records": result.failed_records,
            "pass_rate": round(result.pass_rate, 2)
        },
        "rule_failures": result.rule_failures,
        "failed_record_indices": [int(idx) for idx, valid in result.record_validation.items() if not valid],
        "failure_details": {int(k): v for k, v in result.failure_details.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    console.print(f"💾 Saved JSON report to: {output_path}", style="green")


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def generate_rules(
    population_csv: Path = typer.Option(
        ...,
        "--population", "-p",
        help="Path to population CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    metadata_json: Path = typer.Option(
        ...,
        "--metadata", "-m",
        help="Path to SDV metadata JSON (REQUIRED for type consistency)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    output: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output path for generated YAML rules"
    ),
    dataset_name: str = typer.Option(
        "dataset",
        "--name", "-n",
        help="Name for this dataset"
    ),
):
    """
    Generate validation rules from population data.

    Analyzes population data and creates a YAML file with auto-detected rules
    for categorical membership, ranges, and templates for combination rules.
    """

    console.print()
    console.print("=" * 80, style="blue")
    console.print("  RULE GENERATION", style="bold blue")
    console.print("  Auto-generate validation rules from population data", style="blue")
    console.print("=" * 80, style="blue")
    console.print()

    try:
        # Load population data with metadata for type consistency
        console.print(f"📂 Loading population data from: {population_csv}", style="bold")
        console.print(f"   Using metadata: {metadata_json}")
        population = load_csv_with_metadata(population_csv, metadata_json, low_memory=False)
        console.print(f"   Loaded {len(population):,} rows × {len(population.columns)} columns\n")

        # Generate rules
        generator = RuleGenerator(population, dataset_name)
        rules = generator.generate_rules()

        # Save to YAML
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False, sort_keys=False)

        console.print(f"✅ Rules saved to: {output}", style="bold green")
        console.print()
        console.print("📝 Next steps:", style="bold yellow")
        console.print("   1. Review and edit the generated YAML file")
        console.print("   2. Add combination rules if needed (e.g., geography)")
        console.print("   3. Run validation: python plausible_validator.py validate -s synthetic.csv -r rules.yaml")
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    synthetic_csv: Path = typer.Option(
        ...,
        "--synthetic", "-s",
        help="Path to synthetic CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    rules_yaml: Path = typer.Option(
        ...,
        "--rules", "-r",
        help="Path to validation rules YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    metadata_json: Path = typer.Option(
        ...,
        "--metadata", "-m",
        help="Path to SDV metadata JSON (REQUIRED for type consistency)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    population_csv: Optional[Path] = typer.Option(
        None,
        "--population", "-p",
        help="Path to population CSV (required if rules use 'from_population')",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save validation results to JSON file"
    ),
    n_samples: int = typer.Option(
        5,
        "--samples", "-n",
        help="Number of failing records to display"
    ),
    no_visualization: bool = typer.Option(
        False,
        "--no-viz",
        help="Skip sample failing records visualization"
    ),
):
    """
    Validate synthetic data against rules.

    Checks synthetic data against all rules defined in the YAML file and
    reports which records pass/fail validation.
    """

    console.print()
    console.print("=" * 80, style="blue")
    console.print("  PLAUSIBILITY VALIDATION", style="bold blue")
    console.print("  Rule-based validation for synthetic data", style="blue")
    console.print("=" * 80, style="blue")
    console.print()

    try:
        # Load synthetic data with metadata for type consistency
        console.print(f"📂 Loading synthetic data from: {synthetic_csv}", style="bold")
        console.print(f"   Using metadata: {metadata_json}")
        synthetic = load_csv_with_metadata(synthetic_csv, metadata_json, low_memory=False)
        console.print(f"   Loaded {len(synthetic):,} rows × {len(synthetic.columns)} columns\n")

        # Load rules
        console.print(f"📋 Loading validation rules from: {rules_yaml}", style="bold")
        rules = RuleLoader.load(rules_yaml)
        console.print(f"   Loaded {len(rules.categorical_rules)} categorical rules")
        console.print(f"   Loaded {len(rules.range_rules)} range rules")
        console.print(f"   Loaded {len(rules.combination_rules)} combination rules\n")

        # Load population if needed (with metadata)
        population = None
        if population_csv:
            console.print(f"📂 Loading population data from: {population_csv}", style="bold")
            console.print(f"   Using metadata: {metadata_json}")
            population = load_csv_with_metadata(population_csv, metadata_json, low_memory=False)
            console.print(f"   Loaded {len(population):,} rows\n")

        # Run validation
        engine = ValidationEngine(rules, population)
        result = engine.validate(synthetic)

        # Display results
        display_validation_results(result, synthetic, rules, n_samples, no_visualization)

        # Save JSON if requested
        if output_json:
            save_validation_json(result, output_json)

        console.print("=" * 80, style="blue")
        console.print("✓ Validation complete!", style="bold green")
        console.print("=" * 80, style="blue")
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
