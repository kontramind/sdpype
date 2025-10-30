#!/usr/bin/env python3
"""
Desirable Diverse Records Rate (DDR) Metric Tool

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
"""

import pandas as pd
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional, Tuple, Set

console = Console()
app = typer.Typer(add_completion=False)

# --- Defaults (can be overridden via CLI) ---
DEFAULT_POPULATION_FILE = "experiments/data/processed/canada_covid19_case_details_population.csv"
DEFAULT_TRAINING_FILE = "experiments/data/processed/canada_covid19_case_details_train.csv"
DEFAULT_SYNTHETIC_FILE = "experiments/data/synthetic/synthetic_data_sdv_ctgan_fe6856ff_fe6856ff_fe6856ff_gen_0_b7748603_99_decoded.csv"


def read_and_align(
    pop_csv: Path, train_csv: Path, syn_csv: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    Read all three CSVs and align their columns.

    Returns:
        Tuple of (population_df, training_df, synthetic_df, sorted_columns)
    """
    console.print("📂 Loading datasets...", style="bold blue")

    pop = pd.read_csv(pop_csv, low_memory=False)
    train = pd.read_csv(train_csv, low_memory=False)
    syn = pd.read_csv(syn_csv, low_memory=False)

    console.print(f"  Population: {len(pop):,} rows × {len(pop.columns)} columns")
    console.print(f"  Training:   {len(train):,} rows × {len(train.columns)} columns")
    console.print(f"  Synthetic:  {len(syn):,} rows × {len(syn.columns)} columns")

    # Verify all have same columns
    pop_cols = set(pop.columns)
    train_cols = set(train.columns)
    syn_cols = set(syn.columns)

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

    # Sort columns for consistent comparison
    cols = sorted(pop.columns)

    # Convert to strings to avoid dtype surprises
    pop = pop[cols].astype(str)
    train = train[cols].astype(str)
    syn = syn[cols].astype(str)

    console.print("✓ All datasets aligned\n", style="green")

    return pop, train, syn, cols


def compute_ddr_metrics(
    pop: pd.DataFrame, train: pd.DataFrame, syn: pd.DataFrame
) -> dict:
    """
    Compute DDR and related metrics using hash-based set operations.
    Computes metrics both for unique records and including duplicates.

    Returns:
        Dictionary with metric results and record sets
    """
    console.print("🔢 Computing metrics...", style="bold blue")

    # Compute row hashes for fast set operations
    pop_hashes = pd.util.hash_pandas_object(pop, index=False).astype("uint64")
    train_hashes = pd.util.hash_pandas_object(train, index=False).astype("uint64")
    syn_hashes = pd.util.hash_pandas_object(syn, index=False).astype("uint64")

    # Count duplicates in synthetic data
    syn_hash_counts = syn_hashes.value_counts()
    total_syn_rows = len(syn_hashes)  # Total including duplicates
    unique_syn_rows = len(syn_hash_counts)  # Unique records
    duplicate_rows = total_syn_rows - unique_syn_rows

    # Convert to sets
    P = set(pop_hashes.tolist())
    T = set(train_hashes.tolist())
    S = set(syn_hashes.tolist())

    # Create hash to index mappings for later lookup
    syn_hash_to_idx = {h: idx for idx, h in enumerate(syn_hashes)}

    # Compute set operations (unique-based)
    S_intersect_P = S & P  # Synthetic records in population (factual)
    S_intersect_T = S & T  # Synthetic records in training (copies)
    S_minus_P = S - P      # Synthetic records not in population (hallucinated)

    # DDR: Records in population but not in training
    # (S ∩ P) \ T = factual AND novel
    DDR_set = S_intersect_P - T

    # ===== UNIQUE-BASED METRICS =====
    unique_total = len(S)
    unique_hallucinated = len(S_minus_P)
    unique_training_copies = len(S_intersect_T)
    unique_population_matches = len(S_intersect_P)
    unique_ddr = len(DDR_set)

    unique_hr = (unique_hallucinated / unique_total * 100) if unique_total > 0 else 0.0
    unique_training_copy_rate = (unique_training_copies / unique_total * 100) if unique_total > 0 else 0.0
    unique_pop_match_rate = (unique_population_matches / unique_total * 100) if unique_total > 0 else 0.0
    unique_ddr_rate = (unique_ddr / unique_total * 100) if unique_total > 0 else 0.0

    # ===== TOTAL-BASED METRICS (including duplicates) =====
    # Count how many times each category appears (weighted by duplicates)
    total_hallucinated = sum(syn_hash_counts[h] for h in S_minus_P)
    total_training_copies = sum(syn_hash_counts[h] for h in S_intersect_T)
    total_population_matches = sum(syn_hash_counts[h] for h in S_intersect_P)
    total_ddr = sum(syn_hash_counts[h] for h in DDR_set)

    total_hr = (total_hallucinated / total_syn_rows * 100) if total_syn_rows > 0 else 0.0
    total_training_copy_rate = (total_training_copies / total_syn_rows * 100) if total_syn_rows > 0 else 0.0
    total_pop_match_rate = (total_population_matches / total_syn_rows * 100) if total_syn_rows > 0 else 0.0
    total_ddr_rate = (total_ddr / total_syn_rows * 100) if total_syn_rows > 0 else 0.0

    # Duplicate rate
    duplicate_rate = (duplicate_rows / total_syn_rows * 100) if total_syn_rows > 0 else 0.0

    console.print("✓ Metrics computed (unique + total)\n", style="green")

    return {
        # Duplicate info
        "total_rows": total_syn_rows,
        "unique_rows": unique_syn_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rate,

        # UNIQUE-based metrics
        "unique": {
            "total_synthetic": unique_total,
            "hallucinated_count": unique_hallucinated,
            "training_copy_count": unique_training_copies,
            "population_match_count": unique_population_matches,
            "ddr_count": unique_ddr,
            "hallucination_rate": unique_hr,
            "training_copy_rate": unique_training_copy_rate,
            "population_match_rate": unique_pop_match_rate,
            "ddr_rate": unique_ddr_rate,
        },

        # TOTAL-based metrics (including duplicates)
        "total": {
            "total_synthetic": total_syn_rows,
            "hallucinated_count": total_hallucinated,
            "training_copy_count": total_training_copies,
            "population_match_count": total_population_matches,
            "ddr_count": total_ddr,
            "hallucination_rate": total_hr,
            "training_copy_rate": total_training_copy_rate,
            "population_match_rate": total_pop_match_rate,
            "ddr_rate": total_ddr_rate,
        },

        # Sets for visualization
        "S_minus_P": S_minus_P,
        "S_intersect_T": S_intersect_T,
        "DDR_set": DDR_set,
        "syn_hash_to_idx": syn_hash_to_idx,
        "syn_hash_counts": syn_hash_counts,
    }


def display_duplicate_breakdown(metrics: dict):
    """Display detailed breakdown of duplicates by category."""

    if metrics['duplicate_rows'] == 0:
        return  # No duplicates, skip this section

    syn_hash_counts = metrics['syn_hash_counts']

    # Find records that appear more than once
    duplicated_hashes = syn_hash_counts[syn_hash_counts > 1]

    if len(duplicated_hashes) == 0:
        return

    # Categorize duplicated records
    ddr_set = metrics['DDR_set']
    train_copies = metrics['S_intersect_T']
    hallucinations = metrics['S_minus_P']

    ddr_dups = {h: count for h, count in duplicated_hashes.items() if h in ddr_set}
    train_dups = {h: count for h, count in duplicated_hashes.items() if h in train_copies}
    hall_dups = {h: count for h, count in duplicated_hashes.items() if h in hallucinations}

    total_ddr_dup_rows = sum(ddr_dups.values()) - len(ddr_dups) if ddr_dups else 0
    total_train_dup_rows = sum(train_dups.values()) - len(train_dups) if train_dups else 0
    total_hall_dup_rows = sum(hall_dups.values()) - len(hall_dups) if hall_dups else 0

    console.print("🔄 Duplicate Breakdown by Category", style="bold yellow")
    console.print(f"  ✓ DDR duplicates:          {len(ddr_dups):,} unique records → {total_ddr_dup_rows:,} duplicate rows")
    console.print(f"  ⚠ Training copy duplicates: {len(train_dups):,} unique records → {total_train_dup_rows:,} duplicate rows")
    console.print(f"  ✗ Hallucination duplicates: {len(hall_dups):,} unique records → {total_hall_dup_rows:,} duplicate rows")

    # Show most duplicated record
    if len(duplicated_hashes) > 0:
        most_dup_hash = duplicated_hashes.idxmax()
        most_dup_count = duplicated_hashes.max()

        if most_dup_hash in ddr_set:
            category = "✓ DDR"
            color = "green"
        elif most_dup_hash in train_copies:
            category = "⚠ Training Copy"
            color = "yellow"
        else:
            category = "✗ Hallucination"
            color = "red"

        console.print(f"  Most duplicated: [{color}]{category}[/{color}] record appears {most_dup_count:,} times")
    console.print()


def display_metrics_table(metrics: dict):
    """Display metrics in formatted Rich tables (both unique and total perspectives)."""

    # First show duplicate summary
    console.print()
    console.print("📦 Duplicate Analysis", style="bold blue")
    console.print(f"  Total Generated Rows:     {metrics['total_rows']:,}")
    console.print(f"  Unique Records:           {metrics['unique_rows']:,} ({100 - metrics['duplicate_rate']:.2f}%)")
    console.print(f"  Duplicate Records:        {metrics['duplicate_rows']:,} ({metrics['duplicate_rate']:.2f}%)")
    console.print()

    # Show duplicate breakdown if there are duplicates
    display_duplicate_breakdown(metrics)

    # Create side-by-side comparison table
    comparison_table = Table(
        title="📊 Synthetic Data Quality Metrics - Dual Perspective",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title_style="bold blue"
    )

    comparison_table.add_column("Metric", style="cyan", no_wrap=True)
    comparison_table.add_column("Unique Count", justify="right", style="yellow")
    comparison_table.add_column("Unique Rate", justify="right", style="green")
    comparison_table.add_column("Total Count", justify="right", style="yellow")
    comparison_table.add_column("Total Rate", justify="right", style="green")
    comparison_table.add_column("Interpretation", style="white")

    unique = metrics["unique"]
    total = metrics["total"]

    # Total row
    comparison_table.add_row(
        "Total Synthetic Records",
        f"{unique['total_synthetic']:,}",
        "100.00%",
        f"{total['total_synthetic']:,}",
        "100.00%",
        "All records (unique vs including duplicates)"
    )

    comparison_table.add_section()

    # DDR row
    comparison_table.add_row(
        "✓ DDR (Desirable Diverse)",
        f"{unique['ddr_count']:,}",
        f"{unique['ddr_rate']:.2f}%",
        f"{total['ddr_count']:,}",
        f"{total['ddr_rate']:.2f}%",
        "[bold green]Factual AND Novel (IDEAL)[/bold green]"
    )

    comparison_table.add_section()

    # Training copies
    comparison_table.add_row(
        "⚠ Training Copies",
        f"{unique['training_copy_count']:,}",
        f"{unique['training_copy_rate']:.2f}%",
        f"{total['training_copy_count']:,}",
        f"{total['training_copy_rate']:.2f}%",
        "[yellow]Privacy risk - memorized[/yellow]"
    )

    comparison_table.add_section()

    # Hallucinations
    comparison_table.add_row(
        "✗ Hallucinations",
        f"{unique['hallucinated_count']:,}",
        f"{unique['hallucination_rate']:.2f}%",
        f"{total['hallucinated_count']:,}",
        f"{total['hallucination_rate']:.2f}%",
        "[bold red]Fabricated - not in population[/bold red]"
    )

    comparison_table.add_section()

    # Population matches
    comparison_table.add_row(
        "Population Matches",
        f"{unique['population_match_count']:,}",
        f"{unique['population_match_rate']:.2f}%",
        f"{total['population_match_count']:,}",
        f"{total['population_match_rate']:.2f}%",
        "Factual (includes copies)"
    )

    console.print(comparison_table)
    console.print()

    # Show interpretation note
    console.print("[bold cyan]Interpretation:[/bold cyan]")
    console.print("  • [yellow]Unique Count/Rate[/yellow]: Metrics based on distinct records only")
    console.print("  • [yellow]Total Count/Rate[/yellow]:  Metrics including all duplicates (as generated)")
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


def render_record_comparison(
    syn_row: pd.Series,
    pop_row: Optional[pd.Series],
    cols: list,
    category: str,
    record_id: int
):
    """
    Render a single record with optional comparison to population match.

    Args:
        syn_row: Synthetic record
        pop_row: Matching population record (None if hallucinated)
        cols: Column names
        category: Category label
        record_id: Record number for display
    """
    table = Table(show_header=True, header_style="bold", box=box.SIMPLE_HEAVY)
    table.add_column("Column", style="bold")
    table.add_column("Synthetic", style="cyan")

    if pop_row is not None:
        table.add_column("Population", style="green")

        for c in cols:
            sv = str(syn_row[c])
            pv = str(pop_row[c])

            if sv == pv:
                syn_cell = f"[green]{sv}[/green]"
                pop_cell = f"[green]{pv}[/green]"
            else:
                syn_cell = f"[bold red]{sv}[/bold red]"
                pop_cell = f"[bold red]{pv}[/bold red]"

            table.add_row(c, syn_cell, pop_cell)

        panel_title = f"{category} Record #{record_id}"
    else:
        # Hallucinated - no population match
        for c in cols:
            sv = str(syn_row[c])
            table.add_row(c, f"[yellow]{sv}[/yellow]")

        panel_title = f"{category} Record #{record_id} (No population match)"

    console.print(Panel(table, title=panel_title, title_align="left", border_style="blue"))


def visualize_samples(
    syn: pd.DataFrame,
    pop: pd.DataFrame,
    train: pd.DataFrame,
    metrics: dict,
    cols: list,
    n_samples: int = 3,
    seed: int = 42
):
    """
    Visualize sample records from each category.

    Args:
        syn: Synthetic dataframe
        pop: Population dataframe
        train: Training dataframe
        metrics: Computed metrics dictionary
        cols: Column names
        n_samples: Number of samples per category
        seed: Random seed for sampling
    """
    console.print("📋 Sample Records by Category", style="bold blue")
    console.print()

    syn_hash_to_idx = metrics["syn_hash_to_idx"]

    # Get synthetic hashes for indexing
    syn_hashes = pd.util.hash_pandas_object(syn, index=False).astype("uint64")
    pop_hashes = pd.util.hash_pandas_object(pop, index=False).astype("uint64")

    # Create population hash to row mapping
    pop_hash_to_row = {h: pop.iloc[i] for i, h in enumerate(pop_hashes)}

    # Sample from each category
    categories = [
        ("DDR_set", "✓ DDR (Factual & Novel)", "green", True),
        ("S_intersect_T", "⚠ Training Copy", "yellow", True),
        ("S_minus_P", "✗ Hallucination", "red", False),
    ]

    for set_key, label, color, has_pop_match in categories:
        record_set = metrics[set_key]

        if not record_set:
            console.print(f"[{color}]{label}:[/{color}] No records in this category")
            console.print()
            continue

        console.print(f"[bold {color}]{label}[/bold {color}] ({len(record_set):,} records)")

        # Sample records
        sample_hashes = list(record_set)[:n_samples] if len(record_set) <= n_samples else \
                       pd.Series(list(record_set)).sample(n=n_samples, random_state=seed).tolist()

        for i, h in enumerate(sample_hashes, start=1):
            syn_idx = syn_hash_to_idx[h]
            syn_row = syn.iloc[syn_idx]

            if has_pop_match:
                pop_row = pop_hash_to_row.get(h)
            else:
                pop_row = None

            render_record_comparison(syn_row, pop_row, cols, label, i)

        console.print()


def sanity_checks(metrics: dict):
    """
    Perform sanity checks on computed metrics (both unique and total).
    """
    # Check unique metrics
    unique = metrics["unique"]
    unique_total = unique["total_synthetic"]
    unique_ddr = unique["ddr_count"]
    unique_train_copies = unique["training_copy_count"]
    unique_hallucinations = unique["hallucinated_count"]
    unique_computed = unique_ddr + unique_train_copies + unique_hallucinations

    # Check total metrics
    total = metrics["total"]
    total_total = total["total_synthetic"]
    total_ddr = total["ddr_count"]
    total_train_copies = total["training_copy_count"]
    total_hallucinations = total["hallucinated_count"]
    total_computed = total_ddr + total_train_copies + total_hallucinations

    console.print("🔍 Sanity Checks:", style="bold blue")

    # Check unique
    if unique_computed != unique_total:
        console.print(
            f"  [bold red]✗ Unique metrics:[/bold red] "
            f"DDR({unique_ddr}) + Copies({unique_train_copies}) + Hallucinations({unique_hallucinations}) "
            f"= {unique_computed} ≠ Total({unique_total})",
            style="red"
        )
    else:
        console.print(f"  [green]✓ Unique metrics:[/green] All {unique_total:,} records accounted for")

    # Check total
    if total_computed != total_total:
        console.print(
            f"  [bold red]✗ Total metrics:[/bold red] "
            f"DDR({total_ddr}) + Copies({total_train_copies}) + Hallucinations({total_hallucinations}) "
            f"= {total_computed} ≠ Total({total_total})",
            style="red"
        )
    else:
        console.print(f"  [green]✓ Total metrics:[/green] All {total_total:,} records accounted for")


@app.command()
def evaluate(
    population_csv: Path = typer.Option(
        DEFAULT_POPULATION_FILE,
        "--population", "-p",
        help="Path to population CSV (ground truth)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    training_csv: Path = typer.Option(
        DEFAULT_TRAINING_FILE,
        "--training", "-t",
        help="Path to training CSV (subset of population)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    synthetic_csv: Path = typer.Option(
        DEFAULT_SYNTHETIC_FILE,
        "--synthetic", "-s",
        help="Path to synthetic CSV (generated data)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    n_samples: int = typer.Option(
        3,
        "--samples", "-n",
        min=0,
        help="Number of sample records to visualize per category"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for sampling"
    ),
    show_formula: bool = typer.Option(
        False,
        "--formula", "-f",
        help="Show mathematical formula explanation"
    ),
    no_visualization: bool = typer.Option(
        False,
        "--no-viz",
        help="Skip sample record visualization"
    ),
):
    """
    Evaluate synthetic data quality using the DDR (Desirable Diverse Records) metric.

    Computes the proportion of synthetic records that are both factual (exist in
    population) AND novel (not copied from training data).
    """

    console.print()
    console.print("=" * 80, style="blue")
    console.print("  DDR METRIC EVALUATION", style="bold blue")
    console.print("  Desirable Diverse Records Rate for Synthetic Data Quality", style="blue")
    console.print("=" * 80, style="blue")
    console.print()

    try:
        # Load and align datasets
        pop, train, syn, cols = read_and_align(population_csv, training_csv, synthetic_csv)

        # Compute metrics
        metrics = compute_ddr_metrics(pop, train, syn)

        # Sanity checks
        sanity_checks(metrics)
        console.print()

        # Display results
        if show_formula:
            display_formula_explanation()

        display_metrics_table(metrics)

        # Interpretation (based on total metrics as this reflects actual generation quality)
        ddr_rate_total = metrics["total"]["ddr_rate"]
        ddr_rate_unique = metrics["unique"]["ddr_rate"]

        if ddr_rate_total >= 70:
            quality = "[bold green]EXCELLENT[/bold green]"
        elif ddr_rate_total >= 50:
            quality = "[bold yellow]GOOD[/bold yellow]"
        elif ddr_rate_total >= 30:
            quality = "[yellow]MODERATE[/yellow]"
        else:
            quality = "[bold red]POOR[/bold red]"

        console.print(f"Overall Quality: {quality}")
        console.print(f"  DDR (Total):  {ddr_rate_total:.2f}% - includes all duplicates as generated")
        console.print(f"  DDR (Unique): {ddr_rate_unique:.2f}% - distinct records only")
        console.print()

        # Visualize samples
        if not no_visualization and n_samples > 0:
            visualize_samples(syn, pop, train, metrics, cols, n_samples, seed)

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
