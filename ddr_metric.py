#!/usr/bin/env python3
"""
Desirable Diverse Records Rate (DDR) Metric Tool - DuckDB Implementation

Evaluates synthetic tabular data quality by measuring the proportion of
synthetic records that are both factual (exist in population) AND novel
(not copied from training data).

Based on "Magnitude and Impact of Hallucinations in Tabular Synthetic Health Data"

Metrics:
- DDR (Desirable Diverse Records): |(S ∩ P) \ T| / |S|
- Hallucination Rate (HR): |S \ P| / |S|
- Training Copy Rate: |S ∩ T| / |S|
- Population Match Rate: |S ∩ P| / |S|

Where:
- S = Synthetic dataset
- P = Population dataset (ground truth)
- T = Training dataset (subset of P used to train generator)

Implementation:
Uses DuckDB for efficient set operations on pandas DataFrames.
SQL queries are loaded from external files for flexibility.
"""

import json
import duckdb
import pandas as pd
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional, Dict, Any

# Import metadata utilities for type-safe loading
from sdpype.metadata import load_csv_with_metadata

console = Console()
app = typer.Typer(add_completion=False)


def load_datasets(
    pop_csv: Path,
    train_csv: Path,
    syn_csv: Path,
    metadata_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three datasets with metadata-enforced type consistency.

    Args:
        pop_csv: Path to population CSV
        train_csv: Path to training CSV
        syn_csv: Path to synthetic CSV
        metadata_path: Path to SDV metadata JSON

    Returns:
        Tuple of (population_df, training_df, synthetic_df)
    """
    console.print("📂 Loading datasets with metadata...", style="bold blue")
    console.print(f"  Using metadata: {metadata_path}")

    # Load all datasets with metadata-enforced types
    population = load_csv_with_metadata(pop_csv, metadata_path, low_memory=False)
    training = load_csv_with_metadata(train_csv, metadata_path, low_memory=False)
    synthetic = load_csv_with_metadata(syn_csv, metadata_path, low_memory=False)

    console.print(f"  Population: {len(population):,} rows × {len(population.columns)} columns")
    console.print(f"  Training:   {len(training):,} rows × {len(training.columns)} columns")
    console.print(f"  Synthetic:  {len(synthetic):,} rows × {len(synthetic.columns)} columns")

    # Verify all have same columns
    pop_cols = set(population.columns)
    train_cols = set(training.columns)
    syn_cols = set(synthetic.columns)

    if not (pop_cols == train_cols == syn_cols):
        missing_in_syn = pop_cols - syn_cols
        extra_in_syn = syn_cols - pop_cols
        missing_in_train = pop_cols - train_cols
        extra_in_train = train_cols - pop_cols

        error_msg = "Column mismatch detected!\n"
        if missing_in_syn:
            error_msg += f"  Missing in synthetic: {missing_in_syn}\n"
        if extra_in_syn:
            error_msg += f"  Extra in synthetic: {extra_in_syn}\n"
        if missing_in_train:
            error_msg += f"  Missing in training: {missing_in_train}\n"
        if extra_in_train:
            error_msg += f"  Extra in training: {extra_in_train}\n"

        raise ValueError(error_msg)

    console.print("✓ All datasets aligned with consistent types\n", style="green")

    return population, training, synthetic


def load_query(query_path: Path) -> str:
    """Load SQL query from file."""
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    with open(query_path, 'r') as f:
        return f.read()


def execute_ddr_queries(
    population: pd.DataFrame,
    training: pd.DataFrame,
    synthetic: pd.DataFrame,
    query_dir: Path
) -> Dict[str, Any]:
    """
    Execute DDR metric queries using DuckDB.

    Args:
        population: Population DataFrame
        training: Training DataFrame
        synthetic: Synthetic DataFrame
        query_dir: Directory containing SQL query files

    Returns:
        Dictionary with all metrics and results
    """
    console.print("🔍 Executing DDR queries with DuckDB...", style="bold blue")

    # Create DuckDB connection
    con = duckdb.connect()

    # Register DataFrames (DuckDB will query them directly - zero-copy!)
    con.register('population', population)
    con.register('training', training)
    con.register('synthetic', synthetic)

    console.print("  ✓ Registered DataFrames with DuckDB")

    # Execute summary query to get all metrics at once
    summary_query_path = query_dir / "summary.sql"
    console.print(f"  📊 Executing summary query: {summary_query_path.name}")

    summary_query = load_query(summary_query_path)
    summary_result = con.execute(summary_query).fetchdf()

    # Convert to dict (single row)
    metrics = summary_result.to_dict('records')[0]

    console.print("  ✓ Summary metrics computed")

    # Optionally execute individual queries for detailed records (if needed)
    # For now, we'll just use the summary

    con.close()
    console.print("✓ All queries executed successfully\n", style="green")

    return metrics


def display_metrics(metrics: Dict[str, Any]):
    """Display metrics in formatted Rich tables."""

    console.print()
    console.print("📦 Synthetic Data Summary", style="bold blue")
    console.print(f"  Total Generated Rows:     {metrics['total_synthetic_records']:,}")
    console.print(f"  Unique Records:           {metrics['unique_synthetic_records']:,} ({100 - metrics['duplicate_rate_pct']:.2f}%)")
    console.print(f"  Duplicate Records:        {metrics['duplicate_records']:,} ({metrics['duplicate_rate_pct']:.2f}%)")
    console.print()

    # Main metrics table
    metrics_table = Table(
        title="📊 DDR Metrics - Synthetic Data Quality",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title_style="bold blue"
    )

    metrics_table.add_column("Metric", style="cyan", no_wrap=True)
    metrics_table.add_column("Count", justify="right", style="yellow")
    metrics_table.add_column("Rate", justify="right", style="green")
    metrics_table.add_column("Interpretation", style="white")

    # Total row
    metrics_table.add_row(
        "Total Synthetic Records",
        f"{metrics['total_synthetic_records']:,}",
        "100.00%",
        "All generated records"
    )

    metrics_table.add_section()

    # DDR row
    metrics_table.add_row(
        "✓ DDR (Desirable Diverse)",
        f"{metrics['ddr_count']:,}",
        f"{metrics['ddr_rate_pct']:.2f}%",
        "[bold green]Factual AND Novel (IDEAL)[/bold green]"
    )

    metrics_table.add_section()

    # Training copies
    metrics_table.add_row(
        "⚠ Training Copies",
        f"{metrics['training_copy_count']:,}",
        f"{metrics['training_copy_rate_pct']:.2f}%",
        "[yellow]Privacy risk - memorized[/yellow]"
    )

    metrics_table.add_section()

    # Hallucinations
    metrics_table.add_row(
        "✗ Hallucinations",
        f"{metrics['hallucination_count']:,}",
        f"{metrics['hallucination_rate_pct']:.2f}%",
        "[bold red]Fabricated - not in population[/bold red]"
    )

    metrics_table.add_section()

    # Population matches
    metrics_table.add_row(
        "Population Matches",
        f"{metrics['population_match_count']:,}",
        f"{metrics['population_match_rate_pct']:.2f}%",
        "Factual (includes copies)"
    )

    console.print(metrics_table)
    console.print()

    # Quality assessment
    ddr_rate = metrics['ddr_rate_pct']
    if ddr_rate >= 70:
        quality = "[bold green]EXCELLENT[/bold green]"
    elif ddr_rate >= 50:
        quality = "[bold yellow]GOOD[/bold yellow]"
    elif ddr_rate >= 30:
        quality = "[yellow]MODERATE[/yellow]"
    else:
        quality = "[bold red]POOR[/bold red]"

    console.print(f"Overall Quality: {quality}")
    console.print(f"  DDR Rate: {ddr_rate:.2f}%")
    console.print()


def display_formula_explanation():
    """Display the mathematical formulas and explanations."""

    explanation = """
[bold cyan]Mathematical Definitions[/bold cyan]

Let:
  • S = Synthetic dataset
  • P = Population dataset (ground truth)
  • T = Training dataset (⊆ P)

[bold]Key Metrics:[/bold]

[bold green]DDR (Desirable Diverse Records Rate)[/bold green]
  Formula: DDR = |(S ∩ P) \\ T| / |S|
  Meaning: Proportion of synthetic records that are factual AND novel
  Goal:    [bold]MAXIMIZE[/bold] - the "sweet spot"

[yellow]Training Copy Rate[/yellow]
  Formula: |S ∩ T| / |S|
  Meaning: Proportion that exactly match training data
  Goal:    [bold]MINIMIZE[/bold] - privacy risk

[bold red]Hallucination Rate (HR)[/bold red]
  Formula: |S \\ P| / |S|
  Meaning: Proportion that don't exist in population (fabricated)
  Goal:    [bold]MINIMIZE[/bold] - factual correctness

[bold]Population Match Rate[/bold]
  Formula: |S ∩ P| / |S|
  Meaning: Proportion that exist somewhere in population
  Goal:    High is good (but includes training copies)

[bold]Relationship:[/bold]
  Population Match Rate = DDR Rate + Training Copy Rate
  100% = DDR Rate + Training Copy Rate + Hallucination Rate
"""

    console.print(Panel(explanation, title="📐 Formula Reference", border_style="blue"))
    console.print()


@app.command()
def evaluate(
    population_csv: Path = typer.Option(
        ...,
        "--population", "-p",
        help="Path to population CSV (ground truth)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    training_csv: Path = typer.Option(
        ...,
        "--training", "-t",
        help="Path to training CSV (subset of population)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    synthetic_csv: Path = typer.Option(
        ...,
        "--synthetic", "-s",
        help="Path to synthetic CSV (generated data)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    metadata_json: Path = typer.Option(
        ...,
        "--metadata", "-m",
        help="Path to SDV metadata JSON (REQUIRED for type consistency)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    query_dir: Path = typer.Option(
        "queries/ddr",
        "--query-dir", "-q",
        help="Directory containing SQL query files",
        file_okay=False,
        dir_okay=True
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save metrics to JSON file"
    ),
    show_formula: bool = typer.Option(
        False,
        "--formula", "-f",
        help="Show mathematical formula explanation"
    ),
):
    """
    Evaluate synthetic data quality using the DDR (Desirable Diverse Records) metric.

    Computes the proportion of synthetic records that are both factual (exist in
    population) AND novel (not copied from training data).

    Uses DuckDB for efficient set operations. SQL queries are loaded from files
    for flexibility and customization.
    """

    console.print()
    console.print("=" * 80, style="blue")
    console.print("  DDR METRIC EVALUATION (DuckDB)", style="bold blue")
    console.print("  Desirable Diverse Records Rate for Synthetic Data Quality", style="blue")
    console.print("=" * 80, style="blue")
    console.print()

    try:
        # Verify query directory exists
        if not query_dir.exists():
            raise FileNotFoundError(
                f"Query directory not found: {query_dir}\n"
                f"Expected SQL files: summary.sql, ddr.sql, hallucinations.sql, etc."
            )

        # Load datasets with metadata
        population, training, synthetic = load_datasets(
            population_csv, training_csv, synthetic_csv, metadata_json
        )

        # Execute DDR queries
        metrics = execute_ddr_queries(population, training, synthetic, query_dir)

        # Display formula if requested
        if show_formula:
            display_formula_explanation()

        # Display results
        display_metrics(metrics)

        # Save JSON if requested
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            console.print(f"💾 Saved metrics to: {output_json}", style="green")
            console.print()

        console.print("=" * 80, style="blue")
        console.print("✓ Evaluation complete!", style="bold green")
        console.print("=" * 80, style="blue")
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def formula():
    """
    Display the mathematical formulas and explanations for DDR metric.
    """
    console.print()
    display_formula_explanation()


if __name__ == "__main__":
    app()
