#!/usr/bin/env python3
"""
MIMIC-III Mini CLI Tool

A simplified CLI tool for exploring and transforming MIMIC-III data from XLSX files.
"""

import typer
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional
from sklearn.model_selection import train_test_split

console = Console()
app = typer.Typer(
    help="MIMIC-III Mini Data Preparation Tool",
    rich_markup_mode="rich",
    no_args_is_help=True
)


def generate_encoding_config(impute: bool = False) -> dict:
    """Generate RDT encoding configuration.

    Args:
        impute: If True, use mean imputation for missing values.
                If False, use missing_value_generation: null (no imputation).
    """
    # Always enforce bounds
    enforce_bounds = True

    sdtypes = {
        'ADMTYPE': 'categorical',
        'AGE': 'numerical',
        'ETHGRP': 'categorical',
        'GENDER': 'categorical',
        'NTproBNP': 'numerical',
        'CREAT': 'numerical',
        'BUN': 'numerical',
        'POTASS': 'numerical',
        'CHOL': 'numerical',
        'HR': 'numerical',
        'SBP': 'numerical',
        'DBP': 'numerical',
        'RR': 'numerical',
        'READMIT': 'categorical',
    }

    transformers = {
        'ADMTYPE': {'type': 'UniformEncoder', 'params': {}},
        'AGE': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'ETHGRP': {'type': 'UniformEncoder', 'params': {}},
        'GENDER': {'type': 'UniformEncoder', 'params': {}},
        'NTproBNP': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int32',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'CREAT': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Float',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'BUN': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'POTASS': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Float',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'CHOL': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'HR': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'SBP': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'DBP': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'RR': {
            'type': 'FloatFormatter',
            'params': {
                'computer_representation': 'Int16',
                **({'missing_value_replacement': 'mean'} if impute else {'missing_value_generation': None}),
                'enforce_min_max_values': enforce_bounds,
                'learn_rounding_scheme': True
            }
        },
        'READMIT': {'type': 'UniformEncoder', 'params': {}},
    }

    return {
        'sdtypes': sdtypes,
        'transformers': transformers,
    }


def generate_metadata() -> dict:
    """Generate SDV metadata."""
    columns = {
        'ADMTYPE': {'sdtype': 'categorical'},
        'AGE': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
        'ETHGRP': {'sdtype': 'categorical'},
        'GENDER': {'sdtype': 'categorical'},
        'NTproBNP': {'sdtype': 'numerical', 'computer_representation': 'Int32'},
        'CREAT': {'sdtype': 'numerical', 'computer_representation': 'Float'},
        'BUN': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
        'POTASS': {'sdtype': 'numerical', 'computer_representation': 'Float'},
        'CHOL': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
        'HR': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
        'SBP': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
        'DBP': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
        'RR': {'sdtype': 'numerical', 'computer_representation': 'Int16'},
        'READMIT': {'sdtype': 'categorical'},
    }

    return {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': columns,
    }


def load_data_file(file_path: Path) -> pd.DataFrame:
    """Load data from XLSX or CSV file based on extension."""
    suffix = file_path.suffix.lower()

    if suffix == '.xlsx':
        return pd.read_excel(file_path)
    elif suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .xlsx or .csv")


def apply_transformations(df: pd.DataFrame, verbose: bool = True, keep_ids: bool = False) -> pd.DataFrame:
    """Apply all transformations to a dataframe."""

    # Drop unwanted columns (but NOT ICUSTAY_ID yet - it may have been dropped already for sampling)
    if verbose:
        console.print("[bold cyan]Dropping unwanted columns:[/bold cyan]")
    unwanted_columns = [
        # ID columns (ICUSTAY_ID handled separately for sampling)
        'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
        # Other unwanted columns
        'IS_NEWBORN', 'ICD9_CHAPTER', 'WEIGHT', 'HEIGHT',
        'INSURANCE', 'RELIGION_GROUP', 'TEMP',
        'EXPIRE_FLAG', 'HOSPITAL_EXPIRE_FLAG', 'SPO2',
        'ICUSTAY_EXPIRE', 'HEMOGLOBIN', 'ALBUMIN',
        'LANGUAGE_GROUP', 'MARITAL_GROUP', 'GLUCOSE_BLOOD'
    ]
    columns_to_drop = [col for col in unwanted_columns if col in df.columns]

    # Keep ID columns if requested
    if keep_ids:
        id_columns = ['ICUSTAY_ID', 'SUBJECT_ID', 'HADM_ID']
        columns_to_drop = [col for col in columns_to_drop if col not in id_columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        if verbose:
            for col in columns_to_drop:
                console.print(f"  [green]>[/green] Dropped: {col}")
    elif verbose:
        console.print(f"  [yellow]No unwanted columns found to drop[/yellow]")

    if verbose:
        console.print(f"\n[cyan]Dataset after dropping: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

    # Rename columns
    if verbose:
        console.print("[bold cyan]Renaming columns:[/bold cyan]")
    column_mapping = {
        'ADMISSION_TYPE': 'ADMTYPE',
        'ADMTYPE': 'ADMTYPE',
        'AGE': 'AGE',
        'ETHNICITY_GROUPED': 'ETHGRP',
        'ETHNICITY_GROUP': 'ETHGRP',
        'GENDER': 'GENDER',
        'NTPROBNP_FIRST': 'NTproBNP',
        'NT-proBNP': 'NTproBNP',
        'CREATININE_FIRST': 'CREAT',
        'Creatinine': 'CREAT',
        'BUN_FIRST': 'BUN',
        'BLOOD_UREA_NITRO': 'BUN',
        'POTASSIUM_FIRST': 'POTASS',
        'POTASSIUM': 'POTASS',
        'TOTAL_CHOLESTEROL_FIRST': 'CHOL',
        'CHOLESTEROL': 'CHOL',
        'HR_FIRST': 'HR',
        'HEARTRATE': 'HR',
        'SYSBP_FIRST': 'SBP',
        'SYSTOLIC': 'SBP',
        'DIASBP_FIRST': 'DBP',
        'DIASTOLIC': 'DBP',
        'RESPRATE_FIRST': 'RR',
        'RESP': 'RR',
        'IS_READMISSION_30D': 'READMIT',
        'READMISSION': 'READMIT',
    }

    columns_to_rename = {old: new for old, new in column_mapping.items() if old in df.columns}

    if columns_to_rename:
        if verbose:
            for old_name, new_name in columns_to_rename.items():
                if old_name != new_name:
                    console.print(f"  [green]>[/green] {old_name} → {new_name}")
                else:
                    console.print(f"  [dim]·[/dim] {old_name} (unchanged)")
        df = df.rename(columns=columns_to_rename)
    elif verbose:
        console.print(f"  [yellow]No columns found to rename[/yellow]")

    if verbose:
        console.print(f"\n[cyan]Dataset after renaming: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

    # Transform numeric columns
    if verbose:
        console.print("[bold cyan]Transforming numeric columns:[/bold cyan]")

    # AGE: convert to Int16
    if 'AGE' in df.columns:
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
        df['AGE'] = df['AGE'].round().astype('Int16')
        if verbose:
            console.print(f"  [green]>[/green] AGE → Int16 (whole numbers, allows NULL)")

    # Other Int16 columns
    int16_columns = ['HR', 'SBP', 'DBP', 'RR', 'BUN', 'CHOL']
    for col in int16_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].round().astype('Int16')
            if verbose:
                console.print(f"  [green]>[/green] {col} → Int16 (whole numbers, allows NULL)")

    # Int32 columns
    if 'NTproBNP' in df.columns:
        df['NTproBNP'] = pd.to_numeric(df['NTproBNP'], errors='coerce')
        df['NTproBNP'] = df['NTproBNP'].round().astype('Int32')
        if verbose:
            console.print(f"  [green]>[/green] NTproBNP → Int32 (whole numbers, allows NULL)")

    # Float columns
    float_columns = ['CREAT', 'POTASS']
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].round(2)
            if verbose:
                console.print(f"  [green]>[/green] {col} → Float (2 decimals, allows NULL)")

    # READMIT - treat as categorical (keep string representation)
    if 'READMIT' in df.columns:
        df['READMIT'] = df['READMIT'].astype(str)
        if verbose:
            console.print(f"  [green]>[/green] READMIT → String (categorical)")

    return df


def validate_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unsampled_df: pd.DataFrame,
    target_col: str = 'READMIT'
) -> None:
    """Validate data splits for leakage and distribution issues.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        unsampled_df: Unsampled dataframe
        target_col: Name of target column to check distribution
    """

    console.print("\n[bold cyan]" + "═" * 60 + "[/bold cyan]")
    console.print("[bold cyan]Data Split Validation[/bold cyan]")
    console.print("[bold cyan]" + "═" * 60 + "[/bold cyan]\n")

    # Check for ID columns
    has_icustay = 'ICUSTAY_ID' in train_df.columns
    has_subject = 'SUBJECT_ID' in train_df.columns

    if not (has_icustay or has_subject):
        console.print("[yellow]⚠ No ID columns found. Cannot validate for leakage.[/yellow]")
        console.print("[yellow]  Tip: ID columns were dropped during transformation.[/yellow]")
        console.print("[yellow]  This validation only works with --keep-ids flag.[/yellow]\n")
        return

    # 1. Check for episode-level leakage (ICUSTAY_ID)
    console.print("[bold]1. Episode-Level Leakage Check (ICUSTAY_ID):[/bold]\n")

    if has_icustay:
        train_icu = set(train_df['ICUSTAY_ID'])
        test_icu = set(test_df['ICUSTAY_ID'])
        unsampled_icu = set(unsampled_df['ICUSTAY_ID'])

        overlap_train_test = len(train_icu & test_icu)
        overlap_train_unsamp = len(train_icu & unsampled_icu)
        overlap_test_unsamp = len(test_icu & unsampled_icu)

        status_1 = "✓ No leakage" if overlap_train_test == 0 else "❌ LEAKAGE!"
        status_2 = "✓ No leakage" if overlap_train_unsamp == 0 else "❌ LEAKAGE!"
        status_3 = "✓ No leakage" if overlap_test_unsamp == 0 else "❌ LEAKAGE!"

        color_1 = "green" if overlap_train_test == 0 else "red"
        color_2 = "green" if overlap_train_unsamp == 0 else "red"
        color_3 = "green" if overlap_test_unsamp == 0 else "red"

        console.print(f"  Train ∩ Test:      {overlap_train_test:>6,} [{color_1}]{status_1}[/{color_1}]")
        console.print(f"  Train ∩ Unsampled: {overlap_train_unsamp:>6,} [{color_2}]{status_2}[/{color_2}]")
        console.print(f"  Test ∩ Unsampled:  {overlap_test_unsamp:>6,} [{color_3}]{status_3}[/{color_3}]")

        if any([overlap_train_test, overlap_train_unsamp, overlap_test_unsamp]):
            console.print(f"\n  [red]⚠ Episode-level leakage detected![/red]")
            console.print(f"  [red]  Same ICU stays appear in multiple splits.[/red]")

    # 2. Check for patient-level overlap (expected for episode-level prediction)
    console.print(f"\n[bold]2. Patient-Level Overlap (SUBJECT_ID):[/bold]\n")

    if has_subject:
        train_subj = set(train_df['SUBJECT_ID'])
        test_subj = set(test_df['SUBJECT_ID'])
        unsampled_subj = set(unsampled_df['SUBJECT_ID'])

        overlap_train_test = len(train_subj & test_subj)
        overlap_train_unsamp = len(train_subj & unsampled_subj)
        overlap_test_unsamp = len(test_subj & unsampled_subj)

        console.print(f"  Train ∩ Test:      {overlap_train_test:>6,} patients")
        console.print(f"  Train ∩ Unsampled: {overlap_train_unsamp:>6,} patients")
        console.print(f"  Test ∩ Unsampled:  {overlap_test_unsamp:>6,} patients")

        if any([overlap_train_test, overlap_train_unsamp, overlap_test_unsamp]):
            console.print(f"\n  [yellow]ℹ Patient-level overlap is EXPECTED for episode-level prediction.[/yellow]")
            console.print(f"  [yellow]  Same patient can have different ICU episodes in different splits.[/yellow]")
            console.print(f"  [yellow]  This is acceptable if prediction task is episode-level, not patient-level.[/yellow]")

    # 3. Check target distribution
    console.print(f"\n[bold]3. Target Distribution ({target_col}):[/bold]\n")

    # Convert string 'True'/'False' to numeric for calculation
    train_positive = (train_df[target_col].astype(str) == 'True').sum()
    test_positive = (test_df[target_col].astype(str) == 'True').sum()
    unsamp_positive = (unsampled_df[target_col].astype(str) == 'True').sum()

    train_rate = train_positive / len(train_df)
    test_rate = test_positive / len(test_df)
    unsamp_rate = unsamp_positive / len(unsampled_df)

    console.print(f"  Training:  {train_rate:.4f} ({train_positive:>5,} / {len(train_df):>6,})")
    console.print(f"  Test:      {test_rate:.4f} ({test_positive:>5,} / {len(test_df):>6,})")
    console.print(f"  Unsampled: {unsamp_rate:.4f} ({unsamp_positive:>5,} / {len(unsampled_df):>6,})")

    train_test_diff = abs(train_rate - test_rate)

    if train_test_diff < 0.001:
        console.print(f"\n  [green]✓ Train/Test distributions identical (diff: {train_test_diff:.4f})[/green]")
        console.print(f"  [green]  Stratified sampling is working correctly.[/green]")
    elif train_test_diff < 0.02:
        console.print(f"\n  [green]✓ Train/Test distributions similar (diff: {train_test_diff:.4f})[/green]")
    else:
        console.print(f"\n  [yellow]⚠ Train/Test distribution mismatch (diff: {train_test_diff:.4f})[/yellow]")
        console.print(f"  [yellow]  Consider using --stratify flag for balanced splits.[/yellow]")

    console.print("\n[bold cyan]" + "═" * 60 + "[/bold cyan]\n")


@app.command()
def show(
    file_path: Path = typer.Argument(..., help="Path to data file (XLSX or CSV)"),
    rows: int = typer.Option(10, "--rows", "-n", help="Number of rows to display"),
):
    """
    Load a data file (XLSX or CSV) and display the first N rows in a table.
    """
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading file: {file_path}[/blue]")

        # Read file based on extension
        df = load_data_file(file_path)
        console.print(f"[green]Successfully loaded data[/green]")
        console.print(f"[cyan]Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        # Limit rows to display
        display_rows = min(rows, len(df))
        df_display = df.head(display_rows)

        # Create rich table
        table = Table(title=f"First {display_rows} rows", show_lines=True)

        # Add columns
        for col in df_display.columns:
            table.add_column(str(col), style="cyan", no_wrap=False)

        # Add rows
        for _, row in df_display.iterrows():
            table.add_row(*[str(val) for val in row])

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error loading file: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    file_path: Path = typer.Argument(..., help="Path to data file (XLSX or CSV)"),
):
    """
    Display information about the data file (columns, types, null counts).
    """
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading file: {file_path}[/blue]")

        # Read file based on extension
        df = load_data_file(file_path)
        console.print(f"[green]Successfully loaded data[/green]")
        console.print(f"[cyan]Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        # Create info table
        table = Table(title="Column Information", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Column Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Unique", justify="right", style="blue")
        table.add_column("Non-Null Count", justify="right", style="magenta")
        table.add_column("Null Count", justify="right", style="yellow")
        table.add_column("Null %", justify="right", style="red")

        for idx, col in enumerate(df.columns, 1):
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()

            table.add_row(
                str(idx),
                str(col),
                str(df[col].dtype),
                f"{unique_count:,}",
                f"{non_null:,}",
                f"{null_count:,}",
                f"{null_pct:.1f}%"
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error loading file: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def unique(
    file_path: Path = typer.Argument(..., help="Path to data file (XLSX or CSV)"),
    column: str = typer.Argument(..., help="Column name to get unique values from"),
):
    """
    Display all unique values in a specified column.
    """
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading file: {file_path}[/blue]")

        # Read file based on extension
        df = load_data_file(file_path)
        console.print(f"[green]Successfully loaded data[/green]")
        console.print(f"[cyan]Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        # Check if column exists
        if column not in df.columns:
            console.print(f"[red]Error: Column '{column}' not found in dataset[/red]")
            console.print(f"\n[yellow]Available columns:[/yellow]")
            for col in df.columns:
                console.print(f"  - {col}")
            raise typer.Exit(1)

        # Get unique values
        unique_values = df[column].unique()
        null_count = df[column].isna().sum()

        console.print(f"[bold cyan]Column: {column}[/bold cyan]")
        console.print(f"[cyan]Total unique values: {len(unique_values):,}[/cyan]")
        console.print(f"[cyan]Null/NaN values: {null_count:,}[/cyan]\n")

        # Create table for unique values
        table = Table(title=f"Unique values in '{column}'", show_lines=False)
        table.add_column("#", style="dim", width=6, justify="right")
        table.add_column("Value", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Percentage", style="magenta", justify="right")

        # Get value counts
        value_counts = df[column].value_counts(dropna=False)
        total_rows = len(df)

        for idx, (value, count) in enumerate(value_counts.items(), 1):
            percentage = (count / total_rows) * 100
            value_str = str(value) if pd.notna(value) else "[dim]<null>[/dim]"
            table.add_row(
                str(idx),
                value_str,
                f"{count:,}",
                f"{percentage:.1f}%"
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def transform(
    xlsx_path: Path = typer.Argument(..., help="Path to XLSX file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output prefix for file names (with --sample) or full CSV path (without --sample)"),
    sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample size PER SET (creates N train + N test rows, requires --seed)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducible train/test split (required with --sample)"),
    subfolder: Optional[str] = typer.Option(None, "--subfolder", help="Subfolder name to save files in (defaults to dseed{seed} when using --sample)"),
    impute: bool = typer.Option(False, "--impute", "-i", help="Enable missing value imputation (sets missing_value_replacement='mean' in encoding config)"),
    keep_ids: bool = typer.Option(False, "--keep-ids", help="Keep ID columns (ICUSTAY_ID, SUBJECT_ID, HADM_ID) for validation and debugging"),
):
    """
    Transform XLSX file: drop unwanted columns, rename to abbreviated format, apply type transformations, and export to CSV.

    With --sample and --seed flags, creates stratified train/test/unsampled split at ICUSTAY_ID level to avoid data leakage.
    Uses two-stage sampling: (1) random cohort selection preserving natural distribution, (2) stratified train/test split for fair ML evaluation.
    Example: --sample 10000 --seed 42 creates 10k train + 10k test + remaining unsampled rows.

    The --output parameter works differently based on mode:
    - With --sample: used as a prefix (e.g., "MIMIC-III-mini-core" → "MIMIC-III-mini-core_sample1000_dseed42_training.csv")
    - Without --sample: used as the full output CSV path

    Subfolder behavior (with --sample):
    - By default, creates a subfolder named dseed{seed} (e.g., dseed2025)
    - Use --subfolder to specify a custom subfolder name
    - Fails if the subfolder already exists to prevent accidental overwrites

    Output files:
    - CSV data files (training/test/unsampled when sampling, or single file otherwise)
    - RDT encoding config YAML (for SDV/DVC pipeline)
    - SDV metadata JSON (for SDV/DVC pipeline)

    Steps:
    1. Drop unwanted columns (19 total including ICUSTAY_ID after splitting)
    2. Rename remaining columns to abbreviated uppercase format
    3. Transform column types (Int16, Int32, Float, Int8)
    4. Generate encoding config and metadata files

    All numeric types allow NULL values. Use --impute to enable mean imputation for missing values.
    """
    if not xlsx_path.exists():
        console.print(f"[red]Error: XLSX file not found: {xlsx_path}[/red]")
        raise typer.Exit(1)

    # Validate sample/seed must be used together
    if (sample is not None) != (seed is not None):
        console.print(f"[red]Error: --sample and --seed must be used together[/red]")
        raise typer.Exit(1)

    # Validate subfolder can only be used with sample
    if subfolder is not None and sample is None:
        console.print(f"[red]Error: --subfolder can only be used with --sample[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading XLSX file: {xlsx_path}[/blue]")
        df = pd.read_excel(xlsx_path)
        console.print(f"[green]Successfully loaded data[/green]")
        console.print(f"[cyan]Original dataset: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        # Handle train/test split if sampling is requested
        if sample is not None:
            console.print(f"[bold cyan]Two-Stage Sampling with Stratified Split (sample={sample:,}, dseed={seed}):[/bold cyan]\n")

            # Check for ICUSTAY_ID and READMIT columns
            if 'ICUSTAY_ID' not in df.columns:
                console.print(f"[red]Error: ICUSTAY_ID column not found (required for sampling)[/red]")
                raise typer.Exit(1)
            if 'READMISSION' not in df.columns and 'IS_READMISSION_30D' not in df.columns and 'READMIT' not in df.columns:
                console.print(f"[red]Error: READMIT column not found (required for stratified sampling)[/red]")
                raise typer.Exit(1)

            # Normalize READMIT column name
            readmit_col = None
            for col in ['READMISSION', 'IS_READMISSION_30D', 'READMIT']:
                if col in df.columns:
                    readmit_col = col
                    break

            # Step 1: Random sample of study cohort (2x the requested sample size)
            console.print(f"[bold cyan]Step 1: Randomly sampling {sample * 2:,} episodes for study cohort:[/bold cyan]")

            all_icustay_ids = df['ICUSTAY_ID'].unique()
            rng = np.random.RandomState(seed)

            # Sample 2x episodes for cohort
            n_study_cohort = sample * 2
            if len(all_icustay_ids) < n_study_cohort:
                console.print(f"[red]Error: Not enough unique ICUSTAY_IDs ({len(all_icustay_ids):,}) for cohort size {n_study_cohort:,}[/red]")
                raise typer.Exit(1)

            study_cohort_ids = rng.choice(all_icustay_ids, size=n_study_cohort, replace=False)
            study_cohort_df = df[df['ICUSTAY_ID'].isin(study_cohort_ids)].copy()

            # Calculate readmit rate for display
            readmit_rate = study_cohort_df[readmit_col].mean() if readmit_col else 0
            console.print(f"  [green]>[/green] Study cohort: {len(study_cohort_df):,} episodes")
            console.print(f"  [green]>[/green] READMIT rate: {readmit_rate:.4f}\n")

            # Step 2: Stratified split of study cohort into train/test
            console.print(f"[bold cyan]Step 2: Stratified split of cohort into train/test:[/bold cyan]")

            # Stratified 50/50 split to ensure identical READMIT distributions
            # This preserves the natural rate from Stage 1, but ensures fair ML evaluation
            train_df, test_df = train_test_split(
                study_cohort_df,
                test_size=0.5,
                stratify=study_cohort_df[readmit_col] if readmit_col else None,
                random_state=seed
            )

            if readmit_col:
                train_rate = train_df[readmit_col].mean()
                test_rate = test_df[readmit_col].mean()
                dist_diff = abs(train_rate - test_rate)
                console.print(f"  [green]>[/green] Using stratified split for fair binary classification evaluation")
                console.print(f"  [green]>[/green] Train: {len(train_df):,} episodes (READMIT: {train_rate:.4f})")
                console.print(f"  [green]>[/green] Test:  {len(test_df):,} episodes (READMIT: {test_rate:.4f})")
                console.print(f"  [green]>[/green] Distribution difference: {dist_diff:.6f}\n")
            else:
                console.print(f"  [green]>[/green] Random 50/50 split (no stratification without READMIT column)")
                console.print(f"  [green]>[/green] Train: {len(train_df):,} episodes")
                console.print(f"  [green]>[/green] Test:  {len(test_df):,} episodes\n")

            # Step 3: Unsampled = all episodes NOT in study cohort
            console.print(f"[bold cyan]Step 3: Creating unsampled population dataset:[/bold cyan]")

            unsampled_df = df[~df['ICUSTAY_ID'].isin(study_cohort_ids)].copy()

            if readmit_col:
                unsamp_rate = unsampled_df[readmit_col].mean()
                console.print(f"  [green]>[/green] Unsampled: {len(unsampled_df):,} episodes (READMIT: {unsamp_rate:.4f})")
            else:
                console.print(f"  [green]>[/green] Unsampled: {len(unsampled_df):,} episodes")
            console.print(f"  [green]>[/green] Represents: Full population not in study cohort\n")

            # Apply transformations to all 3 sets
            console.print("[bold cyan]Applying transformations to training set:[/bold cyan]")
            train_df = apply_transformations(train_df, verbose=True, keep_ids=keep_ids)
            console.print(f"\n[cyan]Training dataset: {len(train_df):,} rows x {train_df.shape[1]} columns[/cyan]\n")

            console.print("[bold cyan]Applying transformations to test set:[/bold cyan]")
            test_df = apply_transformations(test_df, verbose=True, keep_ids=keep_ids)
            console.print(f"\n[cyan]Test dataset: {len(test_df):,} rows x {test_df.shape[1]} columns[/cyan]\n")

            console.print("[bold cyan]Applying transformations to unsampled set:[/bold cyan]")
            unsampled_df = apply_transformations(unsampled_df, verbose=True, keep_ids=keep_ids)
            console.print(f"\n[cyan]Unsampled dataset: {len(unsampled_df):,} rows x {unsampled_df.shape[1]} columns[/cyan]\n")

            # Run validation if IDs are kept
            if keep_ids:
                validate_splits(
                    train_df=train_df,
                    test_df=test_df,
                    unsampled_df=unsampled_df,
                    target_col='READMIT'
                )

            # Determine output paths
            if output:
                # Parse output path to separate directory and base name
                output_path = Path(output)
                if output_path.parent.exists() or str(output_path.parent) in ['.', '']:
                    # Output has a directory component
                    output_base_dir = output_path.parent if output_path.parent != Path('.') else Path.cwd()
                    prefix = output_path.name
                else:
                    # Treat as prefix in current directory
                    output_base_dir = Path.cwd()
                    prefix = output
                base_name = f"{prefix}_sample{sample}_dseed{seed}"
            else:
                # Use default naming based on input file
                base_name = f"{xlsx_path.stem}_transformed_sample{sample}_dseed{seed}"
                output_base_dir = xlsx_path.parent

            # Determine subfolder name (default to dseed{seed} when using --sample)
            subfolder_name = subfolder if subfolder else f"dseed{seed}"
            output_dir = output_base_dir / subfolder_name

            # Check if subfolder already exists - fail to prevent overwrites
            if output_dir.exists():
                console.print(f"[red]Error: Subfolder already exists: {output_dir}[/red]")
                console.print(f"[yellow]Please remove the existing folder or use a different --subfolder name[/yellow]")
                raise typer.Exit(1)

            # Create the subfolder
            output_dir.mkdir(parents=True, exist_ok=False)
            console.print(f"[green]✓ Created output subfolder: {output_dir}[/green]\n")

            train_output = output_dir / f"{base_name}_training.csv"
            test_output = output_dir / f"{base_name}_test.csv"
            unsampled_output = output_dir / f"{base_name}_unsampled.csv"

            # Save all 3 files
            train_df.to_csv(train_output, index=False)
            console.print(f"[green]✓ Saved training data to: {train_output}[/green]")

            test_df.to_csv(test_output, index=False)
            console.print(f"[green]✓ Saved test data to: {test_output}[/green]")

            unsampled_df.to_csv(unsampled_output, index=False)
            console.print(f"[green]✓ Saved unsampled data to: {unsampled_output}[/green]")

            # Generate encoding config and metadata (shared for all files)
            # Skip if --keep-ids is used since ID columns aren't in the schema
            console.print()
            if keep_ids:
                console.print(f"[yellow]⚠ Skipping encoding config and metadata generation with --keep-ids[/yellow]")
                console.print(f"[yellow]  ID columns in CSV won't match the standard schema.[/yellow]")
                console.print(f"[yellow]  Remove --keep-ids flag to generate configs for SDV/RDT use.[/yellow]")
            else:
                config = generate_encoding_config(impute=impute)
                encoding_path = output_dir / f"{base_name}_encoding.yaml"
                with open(encoding_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                console.print(f"[green]✓ Saved encoding config to: {encoding_path}[/green]")

                metadata = generate_metadata()
                metadata_path = output_dir / f"{base_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                console.print(f"[green]✓ Saved metadata to: {metadata_path}[/green]")

        else:
            # Standard transformation without splitting
            df = apply_transformations(df, verbose=True, keep_ids=keep_ids)
            console.print(f"\n[cyan]Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

            # Determine output path
            if output is None:
                output_path = xlsx_path.parent / f"{xlsx_path.stem}_transformed.csv"
            else:
                output_path = Path(output)

            # Export to CSV
            df.to_csv(output_path, index=False)
            console.print(f"[green]✓ Saved CSV to: {output_path}[/green]")

            # Generate encoding config and metadata
            # Skip if --keep-ids is used since ID columns aren't in the schema
            console.print()
            if keep_ids:
                console.print(f"[yellow]⚠ Skipping encoding config and metadata generation with --keep-ids[/yellow]")
                console.print(f"[yellow]  ID columns in CSV won't match the standard schema.[/yellow]")
                console.print(f"[yellow]  Remove --keep-ids flag to generate configs for SDV/RDT use.[/yellow]")
            else:
                config = generate_encoding_config(impute=impute)
                encoding_path = output_path.parent / f"{output_path.stem}_encoding.yaml"
                with open(encoding_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                console.print(f"[green]✓ Saved encoding config to: {encoding_path}[/green]")

                metadata = generate_metadata()
                metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                console.print(f"[green]✓ Saved metadata to: {metadata_path}[/green]")

        console.print()

    except Exception as e:
        console.print(f"[red]Error transforming file: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
