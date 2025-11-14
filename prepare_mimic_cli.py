#!/usr/bin/env python3
"""
MIMIC Dataset Preparation CLI Tool

Simple CLI tool to load CSV files and display column information
for preparing MIMIC datasets for synthetic data generation.
"""

import typer
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import List, Optional

console = Console()
app = typer.Typer(
    help="MIMIC Dataset Preparation Tool",
    rich_markup_mode="rich",
    no_args_is_help=True
)


@app.command()
def columns(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
):
    """
    Load a CSV file and display its column names.

    This command loads the specified CSV file and outputs all column names
    along with basic dataset information.
    """
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
        raise typer.Exit(1)

    if not csv_path.suffix.lower() == '.csv':
        console.print(f"[yellow]Warning: File does not have .csv extension[/yellow]")

    try:
        console.print(f"[blue]Loading CSV file: {csv_path}[/blue]")
        df = pd.read_csv(csv_path)

        console.print(f"\n[green]Successfully loaded CSV file[/green]")
        console.print(f"[cyan]Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns[/cyan]\n")

        # Display column names
        console.print("[bold cyan]Column names:[/bold cyan]")
        for idx, col in enumerate(df.columns, 1):
            console.print(f"  {idx:3d}. {col}")

        console.print()

    except Exception as e:
        console.print(f"[red]Error loading CSV file: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def unique(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
    column_names: List[str] = typer.Argument(..., help="Column name(s) to show unique values for"),
    show_counts: bool = typer.Option(False, "--counts", "-c", help="Show value counts for each unique value"),
):
    """
    Display unique values for specified column(s).

    Shows all unique values in the specified columns along with the count
    of unique values. Use --counts to also see frequency of each value.
    """
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
        raise typer.Exit(1)

    if not csv_path.suffix.lower() == '.csv':
        console.print(f"[yellow]Warning: File does not have .csv extension[/yellow]")

    try:
        console.print(f"[blue]Loading CSV file: {csv_path}[/blue]")
        df = pd.read_csv(csv_path)

        console.print(f"[green]Successfully loaded CSV file[/green]")
        console.print(f"[cyan]Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns[/cyan]\n")

        # Process each column
        for col_name in column_names:
            if col_name not in df.columns:
                console.print(f"[red]Error: Column '{col_name}' not found in dataset[/red]")
                console.print(f"[yellow]Available columns: {', '.join(df.columns.tolist())}[/yellow]\n")
                continue

            num_unique = df[col_name].nunique(dropna=False)
            num_missing = df[col_name].isna().sum()

            console.print(f"[bold cyan]Column: {col_name}[/bold cyan]")
            console.print(f"[cyan]Unique values: {num_unique:,}[/cyan]")
            console.print(f"[cyan]Missing values: {num_missing:,}[/cyan]\n")

            # Show value counts by default, sorted by value
            value_counts = df[col_name].value_counts(dropna=False).sort_index()

            if show_counts:
                # Show with percentage in table format
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Value", style="cyan")
                table.add_column("Count", justify="right", style="green")
                table.add_column("Percentage", justify="right", style="blue")

                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    value_str = str(value) if pd.notna(value) else "[red]<NA>[/red]"
                    table.add_row(value_str, f"{count:,}", f"{percentage:.2f}%")

                console.print(table)
                console.print()
            else:
                # Simple list format with counts
                console.print("[bold]Unique values (value: count):[/bold]")
                for value, count in value_counts.items():
                    value_str = str(value) if pd.notna(value) else "[red]<NA>[/red]"
                    console.print(f"  {value_str}: {count:,}")
                console.print()

    except Exception as e:
        console.print(f"[red]Error processing CSV file: {str(e)}[/red]")
        raise typer.Exit(1)


# Transformation functions
def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop MIMIC ID columns if they exist."""
    id_columns = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID']
    columns_to_drop = [col for col in id_columns if col in df.columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        console.print(f"  ✓ Dropped ID columns: {', '.join(columns_to_drop)} ({len(columns_to_drop)} columns removed)")
    else:
        console.print(f"  - No ID columns found to drop")

    return df


def transform_age_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform AGE column: fill NaN with 89, cap at 89, round to int."""
    if 'AGE' in df.columns:
        # Fill NaN with 89 (standard approach for MIMIC)
        df['AGE'] = df['AGE'].fillna(89)
        # Cap at 89
        df['AGE'] = df['AGE'].apply(lambda x: min(x, 89))
        # Round and convert to int
        df['AGE'] = df['AGE'].round().astype(int)
        console.print(f"  ✓ Transformed AGE: filled NaN with 89, capped at 89, converted to integers")
    else:
        console.print(f"  - AGE column not found")

    return df


def transform_insurance_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform INSURANCE column: convert to string type."""
    if 'INSURANCE' in df.columns:
        df['INSURANCE'] = df['INSURANCE'].astype(str)
        console.print(f"  ✓ Transformed INSURANCE: converted to string type")
    else:
        console.print(f"  - INSURANCE column not found")

    return df


def transform_admission_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform ADMISSION_TYPE column: convert to string type."""
    if 'ADMISSION_TYPE' in df.columns:
        df['ADMISSION_TYPE'] = df['ADMISSION_TYPE'].astype(str)
        console.print(f"  ✓ Transformed ADMISSION_TYPE: converted to string type")
    else:
        console.print(f"  - ADMISSION_TYPE column not found")

    return df


def transform_marital_status_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform MARITAL_STATUS column: fill NaN with 'Missing', convert to string."""
    if 'MARITAL_STATUS' in df.columns:
        # Fill NaN with 'Missing'
        df['MARITAL_STATUS'] = df['MARITAL_STATUS'].fillna('Missing')
        # Convert to string
        df['MARITAL_STATUS'] = df['MARITAL_STATUS'].astype(str)
        console.print(f"  ✓ Transformed MARITAL_STATUS: filled NaN with 'Missing', converted to string type")
    else:
        console.print(f"  - MARITAL_STATUS column not found")

    return df


def transform_expire_flag_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform EXPIRE_FLAG column: rename to DECEASED, convert to boolean."""
    if 'EXPIRE_FLAG' in df.columns:
        # Rename to DECEASED
        df = df.rename(columns={'EXPIRE_FLAG': 'DECEASED'})
        # Convert to boolean (0 -> False, 1 -> True)
        df['DECEASED'] = df['DECEASED'].astype(bool)
        console.print(f"  ✓ Transformed EXPIRE_FLAG: renamed to DECEASED, converted to boolean")
    else:
        console.print(f"  - EXPIRE_FLAG column not found")

    return df


def transform_readmission_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform READMISSION column: rename to READMITTED, convert to boolean."""
    if 'READMISSION' in df.columns:
        # Rename to READMITTED
        df = df.rename(columns={'READMISSION': 'READMITTED'})
        # Convert to boolean (0 -> False, 1 -> True)
        df['READMITTED'] = df['READMITTED'].astype(bool)
        console.print(f"  ✓ Transformed READMISSION: renamed to READMITTED, converted to boolean")
    else:
        console.print(f"  - READMISSION column not found")

    return df


def transform_potassium_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform POTASSIUM: convert to numeric, drop rows with non-numeric text values."""
    if 'POTASSIUM' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['POTASSIUM'].notna()
        numeric_values = pd.to_numeric(df['POTASSIUM'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows]

        # Convert to numeric (original NaN stays as NaN)
        df['POTASSIUM'] = pd.to_numeric(df['POTASSIUM'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed POTASSIUM: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed POTASSIUM: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - POTASSIUM column not found")

    return df


@app.command()
def transform(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
    output: Optional[Path] = typer.Argument(None, help="Output file path (default: <input>_transformed.csv)"),
):
    """
    Apply transformations to a dataset and save to a new file.

    Currently applies the following transformations:
    - Drops MIMIC ID columns (ROW_ID, SUBJECT_ID, HADM_ID)
    - Transforms AGE: fills NaN with 89, caps at 89, converts to integers
    - Transforms INSURANCE: converts to string type
    - Transforms ADMISSION_TYPE: converts to string type
    - Transforms MARITAL_STATUS: fills NaN with 'Missing', converts to string type
    - Transforms EXPIRE_FLAG: renames to DECEASED, converts to boolean
    - Transforms READMISSION: renames to READMITTED, converts to boolean
    - Transforms POTASSIUM: converts to numeric, drops rows with non-numeric text (preserves NaN)

    More transformations will be added in future versions.
    """
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
        raise typer.Exit(1)

    if not csv_path.suffix.lower() == '.csv':
        console.print(f"[yellow]Warning: File does not have .csv extension[/yellow]")

    # Set default output path if not provided
    if output is None:
        output = csv_path.parent / f"{csv_path.stem}_transformed.csv"

    try:
        console.print(f"[blue]Loading CSV file: {csv_path}[/blue]")
        df = pd.read_csv(csv_path)

        console.print(f"[green]Successfully loaded CSV file[/green]")
        original_rows, original_cols = df.shape
        console.print(f"[cyan]Original dataset: {original_rows:,} rows × {original_cols} columns[/cyan]\n")

        console.print("[bold cyan]Applying transformations:[/bold cyan]")

        # Apply transformations (add more here as needed)
        df = drop_id_columns(df)
        df = transform_age_column(df)
        df = transform_insurance_column(df)
        df = transform_admission_type_column(df)
        df = transform_marital_status_column(df)
        df = transform_expire_flag_column(df)
        df = transform_readmission_column(df)
        df = transform_potassium_column(df)

        # Future transformations will be added here
        # df = filter_rows(df)

        console.print()
        final_rows, final_cols = df.shape
        console.print(f"[cyan]Final dataset: {final_rows:,} rows × {final_cols} columns[/cyan]")

        # Save transformed data
        df.to_csv(output, index=False)
        console.print(f"[green]✓ Saved transformed data to: {output}[/green]\n")

    except Exception as e:
        console.print(f"[red]Error transforming CSV file: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
