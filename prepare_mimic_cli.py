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
from typing import List

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

            # Show value counts by default
            value_counts = df[col_name].value_counts(dropna=False)

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


if __name__ == "__main__":
    app()
