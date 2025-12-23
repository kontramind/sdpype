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
        table.add_column("Non-Null Count", justify="right", style="magenta")
        table.add_column("Null Count", justify="right", style="yellow")
        table.add_column("Null %", justify="right", style="red")

        for idx, col in enumerate(df.columns, 1):
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100

            table.add_row(
                str(idx),
                str(col),
                str(df[col].dtype),
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
def transform(
    xlsx_path: Path = typer.Argument(..., help="Path to XLSX file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output CSV path (default: <input>_transformed.csv)"),
):
    """
    Transform XLSX file: drop ID and unwanted columns and export to CSV.

    Drops the following columns if present:
    - ID columns: SUBJECT_ID, HADM_ID, ICUSTAY_ID
    - Unwanted columns: IS_NEWBORN, ICD9_CHAPTER, WEIGHT, HEIGHT, INSURANCE, RELIGION_GROUP, TEMP
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
            'INSURANCE', 'RELIGION_GROUP', 'TEMP'
        ]
        columns_to_drop = [col for col in unwanted_columns if col in df.columns]

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            for col in columns_to_drop:
                console.print(f"  [green]>[/green] Dropped: {col}")
        else:
            console.print(f"  [yellow]No unwanted columns found to drop[/yellow]")

        console.print(f"\n[cyan]Transformed dataset: {df.shape[0]:,} rows x {df.shape[1]} columns[/cyan]\n")

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
