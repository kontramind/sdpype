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


if __name__ == "__main__":
    app()
