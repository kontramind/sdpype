#!/usr/bin/env python3
"""
Hallucination Score - Unified DDR and Plausibility Validation for Synthetic Data

Computes comprehensive quality metrics for synthetic tabular data using DuckDB:

1. DDR (Desirable Diverse Records): Records that exist in population but not in training
   - Measures diversity and coverage of the synthetic data

2. Plausibility: Records that pass all validation rules
   - Categorical membership (values found in population)
   - Date ranges (within population bounds)
   - Combination constraints (valid multi-column tuples)

Both metrics computed with dual perspectives:
- Unique: Count of distinct records (by hash)
- Total: Count of all records including duplicates

Usage:
    python hallucination_score.py \\
        --population population.csv \\
        --training training.csv \\
        --synthetic synthetic.csv \\
        --metadata metadata.json \\
        --output results.json
"""

import duckdb
import pandas as pd
import typer
import json
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import metadata utilities for type-safe loading
from sdpype.metadata import load_csv_with_metadata

console = Console()


# ============================================================================
# Query Parsing
# ============================================================================

def parse_query_file(query_file: Path) -> Dict[str, str]:
    """Parse SQL file with @query: markers into dictionary of named queries."""

    with open(query_file, 'r') as f:
        content = f.read()

    queries = {}
    current_query_name = None
    current_query_lines = []

    for line in content.split('\n'):
        # Check for query marker
        if line.strip().startswith('-- @query:'):
            # Save previous query if exists
            if current_query_name:
                queries[current_query_name] = '\n'.join(current_query_lines)

            # Start new query
            current_query_name = line.strip().replace('-- @query:', '').strip()
            current_query_lines = []
        else:
            # Accumulate query lines (skip if no query started yet)
            if current_query_name:
                current_query_lines.append(line)

    # Save last query
    if current_query_name:
        queries[current_query_name] = '\n'.join(current_query_lines)

    return queries


# ============================================================================
# DuckDB Query Execution
# ============================================================================

def execute_validation_queries(
    population: pd.DataFrame,
    training: pd.DataFrame,
    synthetic: pd.DataFrame,
    query_file: Path
) -> Dict[str, Any]:
    """
    Execute unified validation queries using DuckDB.

    Returns dictionary with all metrics:
    - synthetic_total_count, synthetic_unique_count
    - ddr_unique_count, ddr_unique_rate_pct, ddr_total_count, ddr_total_rate_pct
    - hallucinated_unique_count, hallucinated_unique_rate_pct, etc.
    - training_copy_unique_count, training_copy_unique_rate_pct, etc.
    - plausible_unique_count, plausible_unique_rate_pct, etc.
    - implausible_unique_count, implausible_unique_rate_pct, etc.
    """

    # Create DuckDB connection and register DataFrames
    con = duckdb.connect()
    con.register('population', population)
    con.register('training', training)
    con.register('synthetic', synthetic)

    # Parse query file
    queries = parse_query_file(query_file)

    if 'summary' not in queries:
        raise ValueError("Query file must contain a 'summary' query marked with '-- @query: summary'")

    # Build hash expression with actual column names
    columns = synthetic.columns.tolist()
    hash_cols = ', '.join([f'"{col}"' for col in columns])

    # Replace template placeholder with actual columns
    summary_query = queries['summary'].replace('{{HASH_COLS}}', hash_cols)

    # Execute query
    console.print("⚙️  Executing validation queries in DuckDB...", style="bold blue")
    result_df = con.execute(summary_query).fetchdf()

    # Convert to dictionary
    result_dict = result_df.to_dict('records')[0]

    # Close connection
    con.close()

    return result_dict


# ============================================================================
# Output Formatting
# ============================================================================

def display_results(results: Dict[str, Any]):
    """Display validation results in rich formatted tables with interpretations."""

    console.print()
    console.print("=" * 80, style="blue")
    console.print("  HALLUCINATION SCORE RESULTS", style="bold blue")
    console.print("  DDR + Plausibility Validation", style="blue")
    console.print("=" * 80, style="blue")
    console.print()

    # ========================================================================
    # Unified Table with All Metrics and Interpretations
    # ========================================================================

    main_table = Table(
        title="📊 Synthetic Data Quality Metrics - Dual Perspective",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )

    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Unique Count", justify="right", style="yellow")
    main_table.add_column("Unique Rate", justify="right", style="yellow")
    main_table.add_column("Total Count", justify="right", style="yellow")
    main_table.add_column("Total Rate", justify="right", style="yellow")
    main_table.add_column("Interpretation", style="white")

    # Population dataset statistics
    main_table.add_row(
        "Population Dataset",
        f"{results['population_unique_count']:,}",
        "—",
        f"{results['population_total_count']:,}",
        "—",
        "Source population data"
    )

    # Training dataset statistics
    main_table.add_row(
        "Training Dataset",
        f"{results['training_unique_count']:,}",
        "—",
        f"{results['training_total_count']:,}",
        "—",
        "Training data subset"
    )

    # Total synthetic records
    main_table.add_row(
        "Total Synthetic Records",
        f"{results['synthetic_unique_count']:,}",
        "100.00%",
        f"{results['synthetic_total_count']:,}",
        "100.00%",
        "All generated records"
    )

    # DDR - Desirable Diverse Records
    main_table.add_row(
        "✓ DDR (Desirable Diverse)",
        f"{results['ddr_unique_count']:,}",
        f"{results['ddr_unique_rate_pct']:.2f}%",
        f"{results['ddr_total_count']:,}",
        f"{results['ddr_total_rate_pct']:.2f}%",
        "Factual AND Novel (IDEAL)"
    )

    # Training Copies
    main_table.add_row(
        "⚠ Training Copies",
        f"{results['training_copy_unique_count']:,}",
        f"{results['training_copy_unique_rate_pct']:.2f}%",
        f"{results['training_copy_total_count']:,}",
        f"{results['training_copy_total_rate_pct']:.2f}%",
        "Privacy risk - memorized"
    )

    # Hallucinations
    main_table.add_row(
        "✗ Hallucinations",
        f"{results['hallucinated_unique_count']:,}",
        f"{results['hallucinated_unique_rate_pct']:.2f}%",
        f"{results['hallucinated_total_count']:,}",
        f"{results['hallucinated_total_rate_pct']:.2f}%",
        "Fabricated - not in population"
    )

    # Plausible Records
    main_table.add_row(
        "✅ Plausible Records",
        f"{results['plausible_unique_count']:,}",
        f"{results['plausible_unique_rate_pct']:.2f}%",
        f"{results['plausible_total_count']:,}",
        f"{results['plausible_total_rate_pct']:.2f}%",
        "Passes all validation rules"
    )

    # Implausible Records
    main_table.add_row(
        "❌ Implausible Records",
        f"{results['implausible_unique_count']:,}",
        f"{results['implausible_unique_rate_pct']:.2f}%",
        f"{results['implausible_total_count']:,}",
        f"{results['implausible_total_rate_pct']:.2f}%",
        "Fails validation rules"
    )

    console.print(main_table)
    console.print()

    # Summary info
    console.print("=" * 80, style="blue")
    console.print("✓ Validation complete!", style="bold green")
    console.print("=" * 80, style="blue")
    console.print()


def save_results_json(results: Dict[str, Any], output_path: Path):
    """Save validation results to JSON file."""

    # Organize results into logical sections
    output = {
        "population_statistics": {
            "total_count": int(results['population_total_count']),
            "unique_count": int(results['population_unique_count'])
        },
        "training_statistics": {
            "total_count": int(results['training_total_count']),
            "unique_count": int(results['training_unique_count'])
        },
        "synthetic_statistics": {
            "total_count": int(results['synthetic_total_count']),
            "unique_count": int(results['synthetic_unique_count'])
        },
        "ddr_metrics": {
            "unique": {
                "count": int(results['ddr_unique_count']),
                "rate_pct": float(results['ddr_unique_rate_pct'])
            },
            "total": {
                "count": int(results['ddr_total_count']),
                "rate_pct": float(results['ddr_total_rate_pct'])
            }
        },
        "hallucination_metrics": {
            "unique": {
                "count": int(results['hallucinated_unique_count']),
                "rate_pct": float(results['hallucinated_unique_rate_pct'])
            },
            "total": {
                "count": int(results['hallucinated_total_count']),
                "rate_pct": float(results['hallucinated_total_rate_pct'])
            }
        },
        "training_copy_metrics": {
            "unique": {
                "count": int(results['training_copy_unique_count']),
                "rate_pct": float(results['training_copy_unique_rate_pct'])
            },
            "total": {
                "count": int(results['training_copy_total_count']),
                "rate_pct": float(results['training_copy_total_rate_pct'])
            }
        },
        "plausibility_metrics": {
            "unique": {
                "count": int(results['plausible_unique_count']),
                "rate_pct": float(results['plausible_unique_rate_pct'])
            },
            "total": {
                "count": int(results['plausible_total_count']),
                "rate_pct": float(results['plausible_total_rate_pct'])
            }
        },
        "implausibility_metrics": {
            "unique": {
                "count": int(results['implausible_unique_count']),
                "rate_pct": float(results['implausible_unique_rate_pct'])
            },
            "total": {
                "count": int(results['implausible_total_count']),
                "rate_pct": float(results['implausible_total_rate_pct'])
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    console.print(f"💾 Saved results to: {output_path}", style="green")


# ============================================================================
# CLI Main Function
# ============================================================================

def main(
    population_csv: Path = typer.Option(
        ...,
        "--population", "-p",
        help="Path to population CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    training_csv: Path = typer.Option(
        ...,
        "--training", "-t",
        help="Path to training CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    synthetic_csv: Path = typer.Option(
        ...,
        "--synthetic", "-s",
        help="Path to synthetic CSV file",
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
    query_file: Path = typer.Option(
        Path(__file__).parent / "queries" / "validation.sql",
        "--queries", "-q",
        help="Path to SQL queries file",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    output_json: Path = typer.Option(
        None,
        "--output", "-o",
        help="Save results to JSON file"
    ),
):
    """
    Hallucination Score - Unified DDR and Plausibility Validation.

    Computes comprehensive quality metrics for synthetic data:
    - DDR: Records in population but not in training (desirable diversity)
    - Hallucination: Records not in population
    - Training copies: Records matching training data
    - Plausibility: Records passing all validation rules

    All metrics computed with dual perspectives (unique + total counts).
    """

    try:
        # Load all datasets with metadata for type consistency
        console.print()
        console.print("📂 Loading datasets...", style="bold blue")

        console.print(f"   Population: {population_csv}")
        population = load_csv_with_metadata(population_csv, metadata_json, low_memory=False)
        console.print(f"   Loaded {len(population):,} rows")

        console.print(f"   Training: {training_csv}")
        training = load_csv_with_metadata(training_csv, metadata_json, low_memory=False)
        console.print(f"   Loaded {len(training):,} rows")

        console.print(f"   Synthetic: {synthetic_csv}")
        synthetic = load_csv_with_metadata(synthetic_csv, metadata_json, low_memory=False)
        console.print(f"   Loaded {len(synthetic):,} rows")

        console.print(f"   Metadata: {metadata_json}")
        console.print()

        # Execute validation queries
        results = execute_validation_queries(
            population=population,
            training=training,
            synthetic=synthetic,
            query_file=query_file
        )

        # Display results
        display_results(results)

        # Save JSON if requested
        if output_json:
            save_results_json(results, output_json)
            console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
