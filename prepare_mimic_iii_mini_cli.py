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


def apply_transformations(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
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
):
    """
    Transform XLSX file: drop unwanted columns, rename to abbreviated format, apply type transformations, and export to CSV.

    With --sample and --seed flags, creates train/test/unsampled split at ICUSTAY_ID level to avoid data leakage.
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
            console.print(f"[bold cyan]Creating train/test split (sample={sample:,}, dseed={seed}):[/bold cyan]")

            # Check for ICUSTAY_ID column
            if 'ICUSTAY_ID' not in df.columns:
                console.print(f"[red]Error: ICUSTAY_ID column not found (required for sampling)[/red]")
                raise typer.Exit(1)

            # Get unique ICUSTAY_IDs and shuffle them
            unique_ids = df['ICUSTAY_ID'].unique()
            rng = np.random.RandomState(seed)
            shuffled_ids = rng.permutation(unique_ids)

            # Split IDs 50/50
            split_point = len(shuffled_ids) // 2
            train_ids = shuffled_ids[:split_point]
            test_ids = shuffled_ids[split_point:]

            # Get rows for each group
            train_group = df[df['ICUSTAY_ID'].isin(train_ids)].copy()
            test_group = df[df['ICUSTAY_ID'].isin(test_ids)].copy()

            console.print(f"  [green]>[/green] Split ICUSTAY_IDs: {len(train_ids):,} train, {len(test_ids):,} test")
            console.print(f"  [green]>[/green] Train group: {len(train_group):,} rows")
            console.print(f"  [green]>[/green] Test group: {len(test_group):,} rows")

            # Check if we have enough rows in each group
            if len(train_group) < sample:
                console.print(f"[red]Error: Cannot sample {sample:,} rows from train group (only {len(train_group):,} available)[/red]")
                console.print(f"[yellow]Suggestion: Use --sample {len(train_group)} or smaller[/yellow]")
                raise typer.Exit(1)
            if len(test_group) < sample:
                console.print(f"[red]Error: Cannot sample {sample:,} rows from test group (only {len(test_group):,} available)[/red]")
                console.print(f"[yellow]Suggestion: Use --sample {len(test_group)} or smaller[/yellow]")
                raise typer.Exit(1)

            # Sample exact number of rows from each group
            train_df = train_group.sample(n=sample, random_state=seed)
            test_df = test_group.sample(n=sample, random_state=seed)

            # Get unsampled data
            sampled_indices = set(train_df.index).union(set(test_df.index))
            unsampled_indices = df.index.difference(sampled_indices)
            unsampled_df = df.loc[unsampled_indices].copy()

            console.print(f"  [green]>[/green] Sampled: {len(train_df):,} train, {len(test_df):,} test")
            console.print(f"  [green]>[/green] Unsampled: {len(unsampled_df):,} rows\n")

            # Apply transformations to all 3 sets
            console.print("[bold cyan]Applying transformations to training set:[/bold cyan]")
            train_df = apply_transformations(train_df, verbose=True)
            console.print(f"\n[cyan]Training dataset: {len(train_df):,} rows x {train_df.shape[1]} columns[/cyan]\n")

            console.print("[bold cyan]Applying transformations to test set:[/bold cyan]")
            test_df = apply_transformations(test_df, verbose=True)
            console.print(f"\n[cyan]Test dataset: {len(test_df):,} rows x {test_df.shape[1]} columns[/cyan]\n")

            console.print("[bold cyan]Applying transformations to unsampled set:[/bold cyan]")
            unsampled_df = apply_transformations(unsampled_df, verbose=True)
            console.print(f"\n[cyan]Unsampled dataset: {len(unsampled_df):,} rows x {unsampled_df.shape[1]} columns[/cyan]\n")

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
            console.print()
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
            df = apply_transformations(df, verbose=True)
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
            console.print()
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
