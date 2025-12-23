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


@app.command()
def show(
    xlsx_path: Path = typer.Argument(..., help="Path to XLSX file"),
    rows: int = typer.Option(10, "--rows", "-n", help="Number of rows to display"),
):
    """
    Load an XLSX file and display the first N rows in a table.
    """
    if not xlsx_path.exists():
        console.print(f"[red]Error: XLSX file not found: {xlsx_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading XLSX file: {xlsx_path}[/blue]")

        # Read XLSX file (first sheet)
        df = pd.read_excel(xlsx_path)
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
        console.print(f"[red]Error loading XLSX file: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    xlsx_path: Path = typer.Argument(..., help="Path to XLSX file"),
):
    """
    Display information about the XLSX file (columns, types, null counts).
    """
    if not xlsx_path.exists():
        console.print(f"[red]Error: XLSX file not found: {xlsx_path}[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading XLSX file: {xlsx_path}[/blue]")

        # Read XLSX file (first sheet)
        df = pd.read_excel(xlsx_path)
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
        console.print(f"[red]Error loading XLSX file: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
