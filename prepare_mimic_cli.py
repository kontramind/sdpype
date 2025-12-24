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
        df = pd.read_csv(csv_path, low_memory=False)

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
            # Handle mixed-type columns by converting to string for sorting
            try:
                value_counts = df[col_name].value_counts(dropna=False).sort_index()
            except TypeError:
                # Mixed types (e.g., floats and strings) - convert to string for sorting
                temp_series = df[col_name].astype(str)
                value_counts = temp_series.value_counts(dropna=False).sort_index()
                # Map back to original values for display
                original_value_map = dict(zip(df[col_name].astype(str), df[col_name]))
                console.print(f"[yellow]  (Note: Mixed types detected, sorting as strings)[/yellow]\n")

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
    """Transform AGE: convert to numeric, drop rows with non-numeric text values."""
    if 'AGE' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['AGE'].notna()
        numeric_values = pd.to_numeric(df['AGE'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed AGE: converted to numeric, dropped {dropped} rows with non-numeric text (preserved NaN)")
        else:
            console.print(f"  ✓ Transformed AGE: converted to numeric type")
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


def transform_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform GENDER column: fill NaN with 'Unknown', convert to string."""
    if 'GENDER' in df.columns:
        # Fill NaN with 'Unknown'
        df['GENDER'] = df['GENDER'].fillna('Unknown')
        # Convert to string
        df['GENDER'] = df['GENDER'].astype(str)
        console.print(f"  ✓ Transformed GENDER: filled NaN with 'Unknown', converted to string type")
    else:
        console.print(f"  - GENDER column not found")

    return df


def transform_language_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform LANGUAGE column: fill NaN with 'Missing', convert to string."""
    if 'LANGUAGE' in df.columns:
        # Fill NaN with 'Missing'
        df['LANGUAGE'] = df['LANGUAGE'].fillna('Missing')
        # Convert to string
        df['LANGUAGE'] = df['LANGUAGE'].astype(str)
        console.print(f"  ✓ Transformed LANGUAGE: filled NaN with 'Missing', converted to string type")
    else:
        console.print(f"  - LANGUAGE column not found")

    return df


def transform_religion_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform RELIGION column: fill NaN with 'Missing', convert to string."""
    if 'RELIGION' in df.columns:
        # Fill NaN with 'Missing'
        df['RELIGION'] = df['RELIGION'].fillna('Missing')
        # Convert to string
        df['RELIGION'] = df['RELIGION'].astype(str)
        console.print(f"  ✓ Transformed RELIGION: filled NaN with 'Missing', converted to string type")
    else:
        console.print(f"  - RELIGION column not found")

    return df


def transform_ethnicity_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform ETHNICITY column: fill NaN with 'Missing', convert to string."""
    if 'ETHNICITY' in df.columns:
        # Fill NaN with 'Missing'
        df['ETHNICITY'] = df['ETHNICITY'].fillna('Missing')
        # Convert to string
        df['ETHNICITY'] = df['ETHNICITY'].astype(str)
        console.print(f"  ✓ Transformed ETHNICITY: filled NaN with 'Missing', converted to string type")
    else:
        console.print(f"  - ETHNICITY column not found")

    return df


def transform_diagnosis_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform DIAGNOSIS column: fill NaN with 'Missing', convert to string, strip whitespace."""
    if 'DIAGNOSIS' in df.columns:
        # Fill NaN with 'Missing'
        df['DIAGNOSIS'] = df['DIAGNOSIS'].fillna('Missing')
        # Convert to string
        df['DIAGNOSIS'] = df['DIAGNOSIS'].astype(str)
        # Strip leading and trailing whitespace
        df['DIAGNOSIS'] = df['DIAGNOSIS'].str.strip()
        console.print(f"  ✓ Transformed DIAGNOSIS: filled NaN with 'Missing', converted to string type, stripped whitespace")
    else:
        console.print(f"  - DIAGNOSIS column not found")

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
    """Transform READMISSION column: rename to READMITTED, treat as categorical."""
    if 'READMISSION' in df.columns:
        # Rename to READMITTED
        df = df.rename(columns={'READMISSION': 'READMITTED'})
        # Convert to string for categorical treatment
        df['READMITTED'] = df['READMITTED'].astype(str)
        console.print(f"  ✓ Transformed READMISSION: renamed to READMITTED, converted to categorical (string)")
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
        df = df[~invalid_rows].copy()

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


def transform_creatinine_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Creatinine: convert to numeric, drop rows with non-numeric text values."""
    if 'Creatinine' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['Creatinine'].notna()
        numeric_values = pd.to_numeric(df['Creatinine'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['Creatinine'] = pd.to_numeric(df['Creatinine'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed Creatinine: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed Creatinine: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - Creatinine column not found")

    return df


def transform_blood_urea_nitro_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform BLOOD_UREA_NITRO: convert to numeric, drop rows with non-numeric text values."""
    if 'BLOOD_UREA_NITRO' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['BLOOD_UREA_NITRO'].notna()
        numeric_values = pd.to_numeric(df['BLOOD_UREA_NITRO'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['BLOOD_UREA_NITRO'] = pd.to_numeric(df['BLOOD_UREA_NITRO'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed BLOOD_UREA_NITRO: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed BLOOD_UREA_NITRO: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - BLOOD_UREA_NITRO column not found")

    return df


def transform_cholesterol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CHOLESTEROL: convert to numeric, drop rows with non-numeric text values."""
    if 'CHOLESTEROL' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['CHOLESTEROL'].notna()
        numeric_values = pd.to_numeric(df['CHOLESTEROL'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['CHOLESTEROL'] = pd.to_numeric(df['CHOLESTEROL'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed CHOLESTEROL: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed CHOLESTEROL: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - CHOLESTEROL column not found")

    return df


def transform_heart_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform HEART_VALUE: convert to numeric, drop rows with non-numeric text values."""
    if 'HEART_VALUE' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['HEART_VALUE'].notna()
        numeric_values = pd.to_numeric(df['HEART_VALUE'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['HEART_VALUE'] = pd.to_numeric(df['HEART_VALUE'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed HEART_VALUE: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed HEART_VALUE: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - HEART_VALUE column not found")

    return df


def transform_systolic_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform SYSTOLIC_VALUE: convert to numeric, drop rows with non-numeric text values."""
    if 'SYSTOLIC_VALUE' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['SYSTOLIC_VALUE'].notna()
        numeric_values = pd.to_numeric(df['SYSTOLIC_VALUE'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['SYSTOLIC_VALUE'] = pd.to_numeric(df['SYSTOLIC_VALUE'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed SYSTOLIC_VALUE: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed SYSTOLIC_VALUE: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - SYSTOLIC_VALUE column not found")

    return df


def transform_diastolic_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform DIASTOLIC_VALUE: convert to numeric, drop rows with non-numeric text values."""
    if 'DIASTOLIC_VALUE' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['DIASTOLIC_VALUE'].notna()
        numeric_values = pd.to_numeric(df['DIASTOLIC_VALUE'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['DIASTOLIC_VALUE'] = pd.to_numeric(df['DIASTOLIC_VALUE'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed DIASTOLIC_VALUE: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed DIASTOLIC_VALUE: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - DIASTOLIC_VALUE column not found")

    return df


def transform_nt_probnp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Transform NT-proBNP: convert to numeric, drop rows with non-numeric text values."""
    if 'NT-proBNP' in df.columns:
        original_count = len(df)

        # Identify rows with non-numeric string values (not NaN)
        not_null = df['NT-proBNP'].notna()
        numeric_values = pd.to_numeric(df['NT-proBNP'], errors='coerce')
        # Rows that were not-null but became null after conversion are invalid text
        invalid_rows = not_null & numeric_values.isna()

        # Drop rows with invalid text values
        df = df[~invalid_rows].copy()

        # Convert to numeric (original NaN stays as NaN)
        df['NT-proBNP'] = pd.to_numeric(df['NT-proBNP'], errors='coerce')

        dropped = original_count - len(df)
        if dropped > 0:
            console.print(f"  ✓ Transformed NT-proBNP: converted to numeric, dropped {dropped} rows with non-numeric text values")
        else:
            console.print(f"  ✓ Transformed NT-proBNP: converted to numeric (no invalid values found)")
    else:
        console.print(f"  - NT-proBNP column not found")

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
    - Transforms AGE: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms INSURANCE: converts to string type
    - Transforms ADMISSION_TYPE: converts to string type
    - Transforms MARITAL_STATUS: fills NaN with 'Missing', converts to string type
    - Transforms GENDER: fills NaN with 'Unknown', converts to string type
    - Transforms LANGUAGE: fills NaN with 'Missing', converts to string type
    - Transforms RELIGION: fills NaN with 'Missing', converts to string type
    - Transforms ETHNICITY: fills NaN with 'Missing', converts to string type
    - Transforms DIAGNOSIS: fills NaN with 'Missing', converts to string type, strips whitespace
    - Transforms EXPIRE_FLAG: renames to DECEASED, converts to boolean
    - Transforms READMISSION: renames to READMITTED, converts to categorical (string)
    - Transforms POTASSIUM: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms Creatinine: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms BLOOD_UREA_NITRO: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms CHOLESTEROL: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms HEART_VALUE: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms SYSTOLIC_VALUE: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms DIASTOLIC_VALUE: converts to numeric, drops rows with non-numeric text (preserves NaN)
    - Transforms NT-proBNP: converts to numeric, drops rows with non-numeric text (preserves NaN)

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
        df = pd.read_csv(csv_path, low_memory=False)

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
        df = transform_gender_column(df)
        df = transform_expire_flag_column(df)
        df = transform_readmission_column(df)
        df = transform_potassium_column(df)
        df = transform_creatinine_column(df)
        df = transform_blood_urea_nitro_column(df)
        df = transform_cholesterol_column(df)
        df = transform_heart_value_column(df)
        df = transform_systolic_value_column(df)
        df = transform_diastolic_value_column(df)
        df = transform_nt_probnp_column(df)
        df = transform_language_column(df)
        df = transform_religion_column(df)
        df = transform_ethnicity_column(df)
        df = transform_diagnosis_column(df)

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
