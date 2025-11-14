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


@app.command()
def transform(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
    output: Optional[Path] = typer.Argument(None, help="Output file path (default: <input>_transformed.csv)"),
):
    """
    Apply transformations to a dataset and save to a new file.

    Currently applies the following transformations:
    - Drops MIMIC ID columns (ROW_ID, SUBJECT_ID, HADM_ID)

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

        # Future transformations will be added here
        # df = filter_rows(df)
        # df = convert_categorical_to_boolean(df)

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
