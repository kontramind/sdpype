#!/usr/bin/env python3
"""
MIMIC-III Mini CLI Tool

A simplified CLI tool for exploring and transforming MIMIC-III data from XLSX files.
"""

import typer
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(
    help="MIMIC-III Mini Data Preparation Tool",
    rich_markup_mode="rich",
    no_args_is_help=True
)


def load_data_file(file_path: Path) -> pd.DataFrame:
    """Load data from XLSX or CSV file based on extension."""
    suffix = file_path.suffix.lower()

    if suffix == '.xlsx':
        return pd.read_excel(file_path)
    elif suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .xlsx or .csv")


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
    output: Path = typer.Option(None, "--output", "-o", help="Output CSV path (default: <input>_transformed.csv)"),
):
    """
    Transform XLSX file: drop unwanted columns, rename to abbreviated format, and export to CSV.

    Steps:
    1. Drop unwanted columns (19 total):
       - ID columns: SUBJECT_ID, HADM_ID, ICUSTAY_ID
       - Other: IS_NEWBORN, ICD9_CHAPTER, WEIGHT, HEIGHT, INSURANCE, RELIGION_GROUP, TEMP,
                EXPIRE_FLAG, HOSPITAL_EXPIRE_FLAG, SPO2, ICUSTAY_EXPIRE, HEMOGLOBIN, ALBUMIN,
                LANGUAGE_GROUP, MARITAL_GROUP, GLUCOSE_BLOOD

    2. Rename remaining columns to abbreviated uppercase format:
       ADMISSION_TYPE → ADMTYPE, ETHNICITY_GROUPED → ETHGRP, NTPROBNP_FIRST → NTproBNP,
       CREATININE_FIRST → CREAT, BUN_FIRST → BUN, POTASSIUM_FIRST → POTASS,
       TOTAL_CHOLESTEROL_FIRST → CHOL, HR_FIRST → HR, SYSBP_FIRST → SBP,
       DIASBP_FIRST → DBP, RESPRATE_FIRST → RR, IS_READMISSION_30D → READMIT
    """
    if not xlsx_path.exists():
        console.print(f"[red]Error: XLSX file not found: {xlsx_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading XLSX file: {xlsx_path}[/blue]")

        # Read XLSX file (first sheet)
        df = pd.read_excel(xlsx_path)
        console.print(f"[green]Successfully loaded data[/green]")
        console.print(f"[cyan]Original dataset: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        # Drop unwanted columns
        console.print("[bold cyan]Dropping unwanted columns:[/bold cyan]")
        unwanted_columns = [
            # ID columns
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
            for col in columns_to_drop:
                console.print(f"  [green]>[/green] Dropped: {col}")
        else:
            console.print(f"  [yellow]No unwanted columns found to drop[/yellow]")

        console.print(f"\n[cyan]Dataset after dropping: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        # Rename columns to abbreviated uppercase versions
        console.print("[bold cyan]Renaming columns:[/bold cyan]")
        column_mapping = {
            # Handle various possible column name formats
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

        # Only rename columns that exist in the dataframe
        columns_to_rename = {old: new for old, new in column_mapping.items() if old in df.columns}

        if columns_to_rename:
            for old_name, new_name in columns_to_rename.items():
                if old_name != new_name:
                    console.print(f"  [green]>[/green] {old_name} → {new_name}")
                else:
                    console.print(f"  [dim]·[/dim] {old_name} (unchanged)")
            df = df.rename(columns=columns_to_rename)
        else:
            console.print(f"  [yellow]No columns found to rename[/yellow]")

        console.print(f"\n[cyan]Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

        # Determine output path
        if output is None:
            output = xlsx_path.parent / f"{xlsx_path.stem}_transformed.csv"

        # Export to CSV
        df.to_csv(output, index=False)
        console.print(f"[green]✓ Saved to: {output}[/green]\n")

    except Exception as e:
        console.print(f"[red]Error transforming file: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
