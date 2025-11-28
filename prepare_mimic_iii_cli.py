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
import numpy as np
from pathlib import Path
from rich.console import Console
from typing import Optional

console = Console()
app = typer.Typer(
    help="MIMIC-III ICU Stay Dataset Preparation Tool",
    rich_markup_mode="rich",
    no_args_is_help=True
)


# =============================================================================
# Column Configuration
# =============================================================================
# Central configuration for all columns - comment out any column to exclude it
# from the transformation pipeline
COLUMN_CONFIG = [
    # Boolean column (treated as categorical: '0'/'1'/'Missing')
    {
        'name': 'IS_READMISSION_30D',
        'category': 'boolean',
        'sdtype': 'categorical',
        'transformer': {'type': 'UniformEncoder', 'params': {}},
        'order': 1,
    },

    # Categorical columns
    {
        'name': 'GENDER',
        'category': 'categorical',
        'sdtype': 'categorical',
        'transformer': {'type': 'UniformEncoder', 'params': {}},
        'order': 2,
    },
    {
        'name': 'ETHNICITY_GROUPED',
        'category': 'categorical',
        'sdtype': 'categorical',
        'transformer': {'type': 'UniformEncoder', 'params': {}},
        'order': 3,
    },
    {
        'name': 'ADMISSION_TYPE',
        'category': 'categorical',
        'sdtype': 'categorical',
        'transformer': {'type': 'UniformEncoder', 'params': {}},
        'order': 4,
    },

    # Numeric columns (Int16 representation)
    {
        'name': 'AGE',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int16',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 5,
    },
    {
        'name': 'HR_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int16',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 6,
    },
    {
        'name': 'SYSBP_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int16',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 7,
    },
    {
        'name': 'DIASBP_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int16',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 8,
    },
    {
        'name': 'RESPRATE_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int16',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 9,
    },

    # Numeric columns (Int32 representation)
    {
        'name': 'NTPROBNP_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int32',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int32',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 10,
    },

    # Numeric columns (Float representation)
    {
        'name': 'CREATININE_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Float',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Float',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 11,
    },
    {
        'name': 'BUN_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int16',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 12,
    },
    {
        'name': 'POTASSIUM_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Float',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Float',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 13,
    },
    {
        'name': 'TOTAL_CHOLESTEROL_FIRST',
        'category': 'numeric',
        'sdtype': 'numerical',
        'computer_representation': 'Int16',
        'transformer': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                'missing_value_replacement': 'mean',
                'enforce_min_max_values': True,
                'learn_rounding_scheme': True
            }
        },
        'order': 14,
    },

    # Commented out columns - uncomment to include them in the pipeline:
    # {
    #     'name': 'LEADS_TO_READMISSION_30D',
    #     'category': 'boolean',
    #     'sdtype': 'categorical',
    #     'transformer': {'type': 'UniformEncoder', 'params': {}},
    #     'order': 15,
    # },
    # {
    #     'name': 'HOSPITAL_EXPIRE_FLAG',
    #     'category': 'boolean',
    #     'sdtype': 'categorical',
    #     'transformer': {'type': 'UniformEncoder', 'params': {}},
    #     'order': 16,
    # },
    # {
    #     'name': 'HEIGHT_FIRST',
    #     'category': 'numeric',
    #     'sdtype': 'numerical',
    #     'computer_representation': 'Float',
    #     'transformer': {'type': 'FloatFormatter', 'params': {}},
    #     'order': 17,
    # },
    # {
    #     'name': 'WEIGHT_FIRST',
    #     'category': 'numeric',
    #     'sdtype': 'numerical',
    #     'computer_representation': 'Float',
    #     'transformer': {'type': 'FloatFormatter', 'params': {}},
    #     'order': 18,
    # },
]


def get_columns_by_category(category: str) -> list:
    """Get list of column names by category."""
    return [col['name'] for col in COLUMN_CONFIG if col['category'] == category]


def get_columns_by_representation(representation: str) -> list:
    """Get list of column names by computer representation."""
    return [
        col['name'] for col in COLUMN_CONFIG
        if col.get('computer_representation') == representation
    ]


def get_all_column_names() -> list:
    """Get all column names in order."""
    return [col['name'] for col in sorted(COLUMN_CONFIG, key=lambda x: x['order'])]


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
    """Generate RDT encoding configuration from COLUMN_CONFIG."""
    sdtypes = {}
    transformers = {}

    for col_config in COLUMN_CONFIG:
        col_name = col_config['name']
        sdtypes[col_name] = col_config['sdtype']
        transformers[col_name] = col_config['transformer']

    return {
        'sdtypes': sdtypes,
        'transformers': transformers,
    }


def generate_metadata() -> dict:
    """Generate SDV metadata from COLUMN_CONFIG."""
    columns = {}

    for col_config in COLUMN_CONFIG:
        col_name = col_config['name']
        col_metadata = {'sdtype': col_config['sdtype']}

        # Add computer_representation if present
        if 'computer_representation' in col_config:
            col_metadata['computer_representation'] = col_config['computer_representation']

        # Add datetime_format if present
        if 'datetime_format' in col_config:
            col_metadata['datetime_format'] = col_config['datetime_format']

        columns[col_name] = col_metadata

    return {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': columns,
    }


def apply_all_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all transformations to the dataframe."""
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
    boolean_columns = get_columns_by_category('boolean')
    for col in boolean_columns:
        df = transform_boolean_column(df, col)

    # Numeric columns (allow NULLs)
    numeric_columns = get_columns_by_category('numeric')
    for col in numeric_columns:
        df = transform_numeric_column(df, col)

    # Keep only selected columns (from COLUMN_CONFIG)
    columns_to_keep = get_all_column_names()
    df = df[[col for col in columns_to_keep if col in df.columns]]
    console.print(f"  [green]>[/green] Kept {len(df.columns)} columns")

    # Convert numeric columns to Int16 (nullable integer type)
    int16_columns = get_columns_by_representation('Int16')
    for col in int16_columns:
        if col in df.columns:
            df[col] = df[col].round().astype('Int16')
            console.print(f"  [green]>[/green] Converted {col} to Int16")

    # Convert numeric columns to Int32 (nullable integer type)
    int32_columns = get_columns_by_representation('Int32')
    for col in int32_columns:
        if col in df.columns:
            df[col] = df[col].round().astype('Int32')
            console.print(f"  [green]>[/green] Converted {col} to Int32")

    # Round float columns to 2 decimal places
    float_columns = get_columns_by_representation('Float')
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].round(2)
            console.print(f"  [green]>[/green] Rounded {col} to 2 decimal places")

    return df


def split_by_icustay_id(df: pd.DataFrame, train_size: int, test_size: int, seed: int) -> tuple:
    """
    Split dataset into train and test sets at ICUSTAY_ID level to avoid leakage.

    Groups by ICUSTAY_ID, shuffles groups randomly, and splits them 50/50.
    Then samples rows from each group to reach target sizes.

    Args:
        df: DataFrame with ICUSTAY_ID column
        train_size: Target number of rows for training set
        test_size: Target number of rows for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    if 'ICUSTAY_ID' not in df.columns:
        console.print("[red]Error: ICUSTAY_ID column not found for splitting[/red]")
        raise typer.Exit(1)

    # Get unique ICUSTAY_IDs and shuffle them
    unique_ids = df['ICUSTAY_ID'].unique()
    rng = np.random.RandomState(seed)
    shuffled_ids = rng.permutation(unique_ids)

    # Split IDs 50/50
    split_point = len(shuffled_ids) // 2
    train_ids = shuffled_ids[:split_point]
    test_ids = shuffled_ids[split_point:]

    # Get rows for each set
    train_df = df[df['ICUSTAY_ID'].isin(train_ids)].copy()
    test_df = df[df['ICUSTAY_ID'].isin(test_ids)].copy()

    # Sample to target sizes if needed
    if len(train_df) > train_size:
        train_df = train_df.sample(n=train_size, random_state=seed)
    if len(test_df) > test_size:
        test_df = test_df.sample(n=test_size, random_state=seed)

    console.print(f"  [green]>[/green] Split by ICUSTAY_ID (50/50 groups)")
    console.print(f"      Training: {len(train_df):,} rows from {len(train_ids):,} unique ICU stays")
    console.print(f"      Test: {len(test_df):,} rows from {len(test_ids):,} unique ICU stays")

    return train_df, test_df


@app.command()
def transform(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
    sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Total sample size to split into train/test (e.g., 10000 creates 5k train + 5k test, requires --seed)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for train/test split (required with --sample)"),
    encoding_config: bool = typer.Option(False, "--encoding-config", "-e", help="Generate RDT encoding config YAML file"),
):
    """
    Apply transformations to MIMIC-III ICU stay dataset.

    When --sample is used, creates train/test split at ICUSTAY_ID level to avoid leakage.
    Output files are named: <input>_training.csv and <input>_test.csv

    Transformations are defined in COLUMN_CONFIG at the top of this file.
    To exclude a column from export, simply comment it out in COLUMN_CONFIG.

    General transformations:
    - Drop ID columns (ICUSTAY_ID, SUBJECT_ID, HADM_ID)
    - Convert all column names to UPPERCASE
    - Categorical columns: string, NULL -> 'Missing'
    - Boolean columns: '0'/'1'/'Missing' as categorical strings
    - Numeric columns: various representations (Int16, Int32, Float) with NULLs allowed

    See COLUMN_CONFIG for the complete list of columns and their configurations.
    """
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
        raise typer.Exit(1)

    # Validate sample/seed must be used together
    if (sample is not None) != (seed is not None):
        console.print(f"[red]Error: --sample and --seed must be used together[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading CSV file: {csv_path}[/blue]")
        df = pd.read_csv(csv_path, low_memory=False)

        console.print(f"[green]Successfully loaded CSV file[/green]")
        original_rows, original_cols = df.shape
        console.print(f"[cyan]Original dataset: {original_rows:,} rows x {original_cols} columns[/cyan]\n")

        # Handle train/test split if sampling is requested
        if sample is not None:
            console.print("[bold cyan]Splitting into train/test sets:[/bold cyan]")
            target_size_per_set = sample // 2
            train_df, test_df = split_by_icustay_id(df, target_size_per_set, target_size_per_set, seed)
            console.print()

            # Apply transformations to training set
            console.print("[bold cyan]Applying transformations to training set:[/bold cyan]")
            train_df = apply_all_transformations(train_df)
            console.print()
            console.print(f"[cyan]Transformed training dataset: {len(train_df):,} rows x {train_df.shape[1]} columns[/cyan]\n")

            # Apply transformations to test set
            console.print("[bold cyan]Applying transformations to test set:[/bold cyan]")
            test_df = apply_all_transformations(test_df)
            console.print()
            console.print(f"[cyan]Transformed test dataset: {len(test_df):,} rows x {test_df.shape[1]} columns[/cyan]\n")

            # Save training data
            train_output = csv_path.parent / f"{csv_path.stem}_transformed_sample{sample}_seed{seed}_training.csv"
            train_df.to_csv(train_output, index=False)
            console.print(f"[green]>[/green] Saved training data to: {train_output}")

            # Save test data
            test_output = csv_path.parent / f"{csv_path.stem}_transformed_sample{sample}_seed{seed}_test.csv"
            test_df.to_csv(test_output, index=False)
            console.print(f"[green]>[/green] Saved test data to: {test_output}")

            # Generate encoding config and metadata if requested
            if encoding_config:
                config = generate_encoding_config()
                encoding_path = csv_path.parent / f"{csv_path.stem}_transformed_sample{sample}_seed{seed}_encoding.yaml"
                with open(encoding_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                console.print(f"[green]>[/green] Saved encoding config to: {encoding_path}")

                metadata = generate_metadata()
                metadata_path = csv_path.parent / f"{csv_path.stem}_transformed_sample{sample}_seed{seed}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                console.print(f"[green]>[/green] Saved metadata to: {metadata_path}")

        else:
            # Standard transformation without splitting
            console.print("[bold cyan]Applying transformations:[/bold cyan]")
            df = apply_all_transformations(df)
            console.print()
            console.print(f"[cyan]Transformed dataset: {len(df):,} rows x {df.shape[1]} columns[/cyan]\n")

            output = csv_path.parent / f"{csv_path.stem}_transformed.csv"
            df.to_csv(output, index=False)
            console.print(f"[green]>[/green] Saved transformed data to: {output}")

            # Generate encoding config and metadata if requested
            if encoding_config:
                config = generate_encoding_config()
                encoding_path = csv_path.parent / f"{csv_path.stem}_encoding.yaml"
                with open(encoding_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                console.print(f"[green]>[/green] Saved encoding config to: {encoding_path}")

                metadata = generate_metadata()
                metadata_path = csv_path.parent / f"{csv_path.stem}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                console.print(f"[green]>[/green] Saved metadata to: {metadata_path}")

        console.print()

    except Exception as e:
        console.print(f"[red]Error transforming CSV file: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
