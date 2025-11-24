#!/usr/bin/env python3
"""
MIMIC-III ICU Stay Dataset Preparation CLI Tool

CLI tool to prepare MIMIC-III ICU stay data for synthetic data generation.
Handles the mimic_iii_icu_stay_with_orphans.csv derived table.
"""

import json
import typer
import yaml
import pandas as pd
from pathlib import Path
from rich.console import Console
from typing import Optional

console = Console()
app = typer.Typer(
    help="MIMIC-III ICU Stay Dataset Preparation Tool",
    rich_markup_mode="rich",
    no_args_is_help=True
)


@app.command()
def columns(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
):
    """
    Load a CSV file and display its column names.
    """
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading CSV file: {csv_path}[/blue]")
        df = pd.read_csv(csv_path)

        console.print(f"\n[green]Successfully loaded CSV file[/green]")
        console.print(f"[cyan]Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        console.print("[bold cyan]Column names:[/bold cyan]")
        for idx, col in enumerate(df.columns, 1):
            console.print(f"  {idx:3d}. {col}")

        console.print()

    except Exception as e:
        console.print(f"[red]Error loading CSV file: {str(e)}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Transformation functions
# =============================================================================

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop MIMIC-III ID columns if they exist."""
    id_columns = ['ICUSTAY_ID', 'SUBJECT_ID', 'HADM_ID']
    columns_to_drop = [col for col in id_columns if col in df.columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        console.print(f"  [green]>[/green] Dropped ID columns: {', '.join(columns_to_drop)}")
    else:
        console.print(f"  - No ID columns found to drop")

    return df


def uppercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to uppercase."""
    original_cols = df.columns.tolist()
    df.columns = [col.upper() for col in df.columns]
    console.print(f"  [green]>[/green] Converted all column names to UPPERCASE")
    return df


def transform_age(df: pd.DataFrame) -> pd.DataFrame:
    """Transform AGE: ensure numeric type."""
    if 'AGE' in df.columns:
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
        console.print(f"  [green]>[/green] AGE: converted to numeric")
    else:
        console.print(f"  - AGE column not found")
    return df


def transform_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Transform GENDER: string, NULL -> 'Missing'."""
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].fillna('Missing').astype(str)
        console.print(f"  [green]>[/green] GENDER: string, NULL -> 'Missing'")
    else:
        console.print(f"  - GENDER column not found")
    return df


def transform_ethnicity_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """Transform ETHNICITY_GROUPED: string, NULL -> 'Missing'."""
    if 'ETHNICITY_GROUPED' in df.columns:
        df['ETHNICITY_GROUPED'] = df['ETHNICITY_GROUPED'].fillna('Missing').astype(str)
        console.print(f"  [green]>[/green] ETHNICITY_GROUPED: string, NULL -> 'Missing'")
    else:
        console.print(f"  - ETHNICITY_GROUPED column not found")
    return df


def transform_admission_type(df: pd.DataFrame) -> pd.DataFrame:
    """Transform ADMISSION_TYPE: string, NULL -> 'Missing'."""
    if 'ADMISSION_TYPE' in df.columns:
        df['ADMISSION_TYPE'] = df['ADMISSION_TYPE'].fillna('Missing').astype(str)
        console.print(f"  [green]>[/green] ADMISSION_TYPE: string, NULL -> 'Missing'")
    else:
        console.print(f"  - ADMISSION_TYPE column not found")
    return df


def transform_boolean_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Transform a boolean column: 0->'0', 1->'1', NULL->'Missing' as categorical string."""
    if col_name in df.columns:
        df[col_name] = df[col_name].apply(
            lambda x: 'Missing' if pd.isna(x) else str(int(x))
        )
        console.print(f"  [green]>[/green] {col_name}: categorical string ('0'/'1'/'Missing')")
    else:
        console.print(f"  - {col_name} column not found")
    return df


def transform_numeric_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Transform a numeric column: ensure numeric type, allow NULLs."""
    if col_name in df.columns:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        console.print(f"  [green]>[/green] {col_name}: numeric, allow NULLs")
    else:
        console.print(f"  - {col_name} column not found")
    return df


def transform_datetime_column(df: pd.DataFrame, col_name: str, allow_nulls: bool = True) -> pd.DataFrame:
    """Transform a datetime column: keep as datetime string."""
    if col_name in df.columns:
        # Keep as string (already datetime strings in CSV)
        df[col_name] = df[col_name].astype(str)
        # Convert 'nan' strings back to actual NaN if nulls allowed
        if allow_nulls:
            df[col_name] = df[col_name].replace('nan', pd.NA)
            df[col_name] = df[col_name].replace('NaT', pd.NA)
        console.print(f"  [green]>[/green] {col_name}: datetime string{', allow NULLs' if allow_nulls else ''}")
    else:
        console.print(f"  - {col_name} column not found")
    return df


def generate_encoding_config() -> dict:
    """Generate RDT encoding configuration for MIMIC-III ICU stay dataset."""
    categorical_columns = [
        'GENDER',
        'ETHNICITY_GROUPED',
        'ADMISSION_TYPE',
    ]
    boolean_columns = [
        # 'LEADS_TO_READMISSION_30D',
        'IS_READMISSION_30D',
        # 'HOSPITAL_EXPIRE_FLAG',
        # 'ICU_MORTALITY_FLAG',
        # 'HAS_ADMISSION_RECORD',
        # 'HAS_ICUSTAY_DETAIL',
    ]
    numeric_columns = [
        'AGE',
        'HR_FIRST',
        'SYSBP_FIRST',
        'DIASBP_FIRST',
        'RESPRATE_FIRST',
        # 'HEIGHT_FIRST',
        # 'WEIGHT_FIRST',
        # 'BMI',
        # 'NTPROBNP_FIRST',
        'CREATININE_FIRST',
        # 'BUN_FIRST',
        # 'POTASSIUM_FIRST',
        # 'TOTAL_CHOLESTEROL_FIRST',
        # 'LOS_ICU',
    ]
    # datetime_columns = ['DOD', 'ICU_INTIME', 'ICU_OUTTIME']
    datetime_columns = []

    # Build sdtypes (boolean columns treated as categorical)
    sdtypes = {}
    for col in categorical_columns:
        sdtypes[col] = 'categorical'
    for col in boolean_columns:
        sdtypes[col] = 'categorical'
    for col in numeric_columns:
        sdtypes[col] = 'numerical'
    for col in datetime_columns:
        sdtypes[col] = 'datetime'

    # Build transformers
    transformers = {}
    for col in categorical_columns:
        transformers[col] = {'type': 'UniformEncoder', 'params': {}}
    for col in boolean_columns:
        transformers[col] = {'type': 'UniformEncoder', 'params': {}}
    for col in numeric_columns:
        # Use Int16 for HR_FIRST, SYSBP_FIRST, DIASBP_FIRST
        if col in ['AGE', 'HR_FIRST', 'SYSBP_FIRST', 'DIASBP_FIRST']:
            transformers[col] = {
                'type': 'FloatFormatter',
                'params': {
                    'computer_representation': 'Int16',
                    'missing_value_generation': None,
                    'enforce_min_max_values': True,
                    'learn_rounding_scheme': True
                }
            }
        elif col in ['RESPRATE_FIRST', 'CREATININE_FIRST']:
            transformers[col] = {
                'type': 'FloatFormatter',
                'params': {
                    'computer_representation': 'Float',
                    'missing_value_generation': None,
                    'enforce_min_max_values': True,
                    'learn_rounding_scheme': True
                }
            }
        else:
            transformers[col] = {'type': 'FloatFormatter', 'params': {}}
    for col in datetime_columns:
        transformers[col] = {
            'type': 'UnixTimestampEncoder',
            'params': {'datetime_format': '%Y-%m-%d %H:%M:%S'}
        }

    return {
        'sdtypes': sdtypes,
        'transformers': transformers,
    }


def generate_metadata() -> dict:
    """Generate SDV metadata for MIMIC-III ICU stay dataset."""
    categorical_columns = [
        'GENDER',
        'ETHNICITY_GROUPED',
        'ADMISSION_TYPE',
    ]
    boolean_columns = [
        # 'LEADS_TO_READMISSION_30D',
        'IS_READMISSION_30D',
        # 'HOSPITAL_EXPIRE_FLAG',
        # 'ICU_MORTALITY_FLAG',
        # 'HAS_ADMISSION_RECORD',
        # 'HAS_ICUSTAY_DETAIL',
    ]
    numeric_columns = [
        'AGE',
        'HR_FIRST',
        'SYSBP_FIRST',
        'DIASBP_FIRST',
        'RESPRATE_FIRST',
        # 'HEIGHT_FIRST',
        # 'WEIGHT_FIRST',
        # 'BMI',
        # 'NTPROBNP_FIRST',
        'CREATININE_FIRST',
        # 'BUN_FIRST',
        # 'POTASSIUM_FIRST',
        # 'TOTAL_CHOLESTEROL_FIRST',
        # 'LOS_ICU',
    ]
    # datetime_columns = ['DOD', 'ICU_INTIME', 'ICU_OUTTIME']
    datetime_columns = []

    # Boolean columns treated as categorical
    columns = {}
    for col in categorical_columns:
        columns[col] = {'sdtype': 'categorical'}
    for col in boolean_columns:
        columns[col] = {'sdtype': 'categorical'}
    for col in numeric_columns:
        # Use Int16 for HR_FIRST, SYSBP_FIRST, DIASBP_FIRST
        if col in ['AGE', 'HR_FIRST', 'SYSBP_FIRST', 'DIASBP_FIRST']:
            columns[col] = {
                'sdtype': 'numerical',
                'computer_representation': 'Int16'
            }
        elif col in ['RESPRATE_FIRST', 'CREATININE_FIRST']:
            columns[col] = {
                'sdtype': 'numerical',
                'computer_representation': 'Float'
            }
        else:
            columns[col] = {'sdtype': 'numerical'}
    for col in datetime_columns:
        columns[col] = {
            'sdtype': 'datetime',
            'datetime_format': '%Y-%m-%d %H:%M:%S'
        }

    return {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': columns,
    }


@app.command()
def transform(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
    output: Optional[Path] = typer.Argument(None, help="Output file path (default: <input>_transformed.csv)"),
    sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Randomly sample N records after transformation (requires --seed)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for sampling (required with --sample)"),
    encoding_config: bool = typer.Option(False, "--encoding-config", "-e", help="Generate RDT encoding config YAML file"),
):
    """
    Apply transformations to MIMIC-III ICU stay dataset.

    Transformations applied:
    - Drop ID columns (ICUSTAY_ID, SUBJECT_ID, HADM_ID)
    - Convert all column names to UPPERCASE
    - AGE: numeric
    - GENDER: string, NULL -> 'Missing'
    - ETHNICITY_GROUPED: string, NULL -> 'Missing'
    - ADMISSION_TYPE: string, NULL -> 'Missing'
    - Boolean columns (allow NULLs): LEADS_TO_READMISSION_30D, IS_READMISSION_30D,
      HOSPITAL_EXPIRE_FLAG, ICU_MORTALITY_FLAG, HAS_ADMISSION_RECORD, HAS_ICUSTAY_DETAIL
    - Numeric columns (allow NULLs): HR_FIRST, SYSBP_FIRST, DIASBP_FIRST, RESPRATE_FIRST,
      HEIGHT_FIRST, WEIGHT_FIRST, BMI, NTPROBNP_FIRST, CREATININE_FIRST, BUN_FIRST,
      POTASSIUM_FIRST, TOTAL_CHOLESTEROL_FIRST, LOS_ICU
    - Datetime columns: DOD (allow NULLs), ICU_INTIME, ICU_OUTTIME (allow NULLs)
    """
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
        raise typer.Exit(1)

    # Validate sample/seed must be used together
    if (sample is not None) != (seed is not None):
        console.print(f"[red]Error: --sample and --seed must be used together[/red]")
        raise typer.Exit(1)

    # Output is required when sampling
    if sample is not None and output is None:
        console.print(f"[red]Error: output path is required when using --sample[/red]")
        raise typer.Exit(1)

    if output is None:
        output = csv_path.parent / f"{csv_path.stem}_transformed.csv"

    try:
        console.print(f"[blue]Loading CSV file: {csv_path}[/blue]")
        df = pd.read_csv(csv_path, low_memory=False)

        console.print(f"[green]Successfully loaded CSV file[/green]")
        original_rows, original_cols = df.shape
        console.print(f"[cyan]Original dataset: {original_rows:,} rows x {original_cols} columns[/cyan]\n")

        console.print("[bold cyan]Applying transformations:[/bold cyan]")

        # Drop ID columns first
        df = drop_id_columns(df)

        # Uppercase all column names
        df = uppercase_columns(df)

        # Categorical/string columns
        df = transform_age(df)
        df = transform_gender(df)
        df = transform_ethnicity_grouped(df)
        df = transform_admission_type(df)

        # Boolean columns (as categorical strings: '0'/'1'/'Missing')
        boolean_columns = [
            # 'LEADS_TO_READMISSION_30D',
            'IS_READMISSION_30D',
            # 'HOSPITAL_EXPIRE_FLAG',
            # 'ICU_MORTALITY_FLAG',
            # 'HAS_ADMISSION_RECORD',
            # 'HAS_ICUSTAY_DETAIL',
        ]
        for col in boolean_columns:
            df = transform_boolean_column(df, col)

        # Numeric columns (allow NULLs)
        numeric_columns = [
            'HR_FIRST',
            'SYSBP_FIRST',
            'DIASBP_FIRST',
            'RESPRATE_FIRST',
            # 'HEIGHT_FIRST',
            # 'WEIGHT_FIRST',
            # 'BMI',
            # 'NTPROBNP_FIRST',
            'CREATININE_FIRST',
            # 'BUN_FIRST',
            # 'POTASSIUM_FIRST',
            # 'TOTAL_CHOLESTEROL_FIRST',
            # 'LOS_ICU',
        ]
        for col in numeric_columns:
            df = transform_numeric_column(df, col)

        # Datetime columns
        # df = transform_datetime_column(df, 'DOD', allow_nulls=True)
        # df = transform_datetime_column(df, 'ICU_INTIME', allow_nulls=False)
        # df = transform_datetime_column(df, 'ICU_OUTTIME', allow_nulls=True)

        # Keep only selected columns
        columns_to_keep = [
            'IS_READMISSION_30D',
            'AGE',
            'GENDER',
            'ETHNICITY_GROUPED',
            'ADMISSION_TYPE',
            'HR_FIRST',
            'SYSBP_FIRST',
            'DIASBP_FIRST',
            'RESPRATE_FIRST',
            # 'NTPROBNP_FIRST',
            'CREATININE_FIRST',
            # 'BUN_FIRST',
            # 'POTASSIUM_FIRST',
            # 'TOTAL_CHOLESTEROL_FIRST',
        ]
        df = df[[col for col in columns_to_keep if col in df.columns]]
        console.print(f"  [green]>[/green] Kept {len(df.columns)} columns")

        # Convert numeric columns to Int16 (nullable integer type)
        int16_columns = ['AGE', 'HR_FIRST', 'SYSBP_FIRST', 'DIASBP_FIRST']
        for col in int16_columns:
            if col in df.columns:
                df[col] = df[col].round().astype('Int16')
                console.print(f"  [green]>[/green] Converted {col} to Int16")

        # Round float columns to 2 decimal places
        float_columns = ['CREATININE_FIRST']
        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].round(2)
                console.print(f"  [green]>[/green] Rounded {col} to 2 decimal places")

        console.print()
        final_rows, final_cols = df.shape
        console.print(f"[cyan]Transformed dataset: {final_rows:,} rows x {final_cols} columns[/cyan]")

        # Apply sampling if requested
        if sample is not None:
            if sample > len(df):
                console.print(f"[yellow]Warning: sample size ({sample}) > dataset size ({len(df)}), using full dataset[/yellow]")
                sample = len(df)
            df = df.sample(n=sample, random_state=seed)
            console.print(f"[cyan]Sampled dataset: {len(df):,} rows (seed={seed})[/cyan]")

        # Save transformed data
        df.to_csv(output, index=False)
        console.print(f"[green]>[/green] Saved transformed data to: {output}")

        # Generate encoding config and metadata if requested
        if encoding_config:
            config = generate_encoding_config()
            encoding_path = output.parent / f"{output.stem}_encoding.yaml"
            with open(encoding_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            console.print(f"[green]>[/green] Saved encoding config to: {encoding_path}")

            metadata = generate_metadata()
            metadata_path = output.parent / f"{output.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            console.print(f"[green]>[/green] Saved metadata to: {metadata_path}")

        console.print()

    except Exception as e:
        console.print(f"[red]Error transforming CSV file: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
