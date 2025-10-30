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

    Returns:
        Dictionary with metric results and record sets
    """
    console.print("🔢 Computing metrics...", style="bold blue")

    # Compute row hashes for fast set operations
    pop_hashes = pd.util.hash_pandas_object(pop, index=False).astype("uint64")
    train_hashes = pd.util.hash_pandas_object(train, index=False).astype("uint64")
    syn_hashes = pd.util.hash_pandas_object(syn, index=False).astype("uint64")

    # Convert to sets
    P = set(pop_hashes.tolist())
    T = set(train_hashes.tolist())
    S = set(syn_hashes.tolist())

    # Create hash to index mappings for later lookup
    syn_hash_to_idx = {h: idx for idx, h in enumerate(syn_hashes)}

    # Compute set operations
    S_intersect_P = S & P  # Synthetic records in population (factual)
    S_intersect_T = S & T  # Synthetic records in training (copies)
    S_minus_P = S - P      # Synthetic records not in population (hallucinated)

    # DDR: Records in population but not in training
    # (S ∩ P) \ T = factual AND novel
    DDR_set = S_intersect_P - T

    # Compute counts
    total_syn = len(S)
    hallucinated_count = len(S_minus_P)
    training_copy_count = len(S_intersect_T)
    population_match_count = len(S_intersect_P)
    ddr_count = len(DDR_set)

    # Compute rates
    hr = (hallucinated_count / total_syn * 100) if total_syn > 0 else 0.0
    training_copy_rate = (training_copy_count / total_syn * 100) if total_syn > 0 else 0.0
    pop_match_rate = (population_match_count / total_syn * 100) if total_syn > 0 else 0.0
    ddr_rate = (ddr_count / total_syn * 100) if total_syn > 0 else 0.0

    console.print("✓ Metrics computed\n", style="green")

    return {
        "total_synthetic": total_syn,
        "hallucinated_count": hallucinated_count,
        "training_copy_count": training_copy_count,
        "population_match_count": population_match_count,
        "ddr_count": ddr_count,
        "hallucination_rate": hr,
        "training_copy_rate": training_copy_rate,
        "population_match_rate": pop_match_rate,
        "ddr_rate": ddr_rate,
        # Sets for visualization
        "S_minus_P": S_minus_P,
        "S_intersect_T": S_intersect_T,
        "DDR_set": DDR_set,
        "syn_hash_to_idx": syn_hash_to_idx,
    }


def display_metrics_table(metrics: dict):
    """Display metrics in a formatted Rich table."""

    table = Table(
        title="📊 Synthetic Data Quality Metrics",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title_style="bold blue"
    )

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right", style="yellow")
    table.add_column("Rate (%)", justify="right", style="green")
    table.add_column("Interpretation", style="white")

    total = metrics["total_synthetic"]

    # Add rows
    table.add_row(
        "Total Synthetic Records",
        f"{total:,}",
        "100.00",
        "All generated records"
    )

    table.add_section()

    table.add_row(
        "✓ DDR (Desirable Diverse)",
        f"{metrics['ddr_count']:,}",
        f"{metrics['ddr_rate']:.2f}",
        "[bold green]Factual AND Novel (IDEAL)[/bold green]"
    )

    table.add_section()

    table.add_row(
        "⚠ Training Copies",
        f"{metrics['training_copy_count']:,}",
        f"{metrics['training_copy_rate']:.2f}",
        "[yellow]Privacy risk - memorized training[/yellow]"
    )

    table.add_row(
        "✗ Hallucinations",
        f"{metrics['hallucinated_count']:,}",
        f"{metrics['hallucination_rate']:.2f}",
        "[bold red]Fabricated - not in population[/bold red]"
    )

    table.add_section()

    table.add_row(
        "Population Matches (Total)",
        f"{metrics['population_match_count']:,}",
        f"{metrics['population_match_rate']:.2f}",
        "Factual (includes training copies)"
    )

    console.print()
    console.print(table)
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
    Perform sanity checks on computed metrics.
    """
    total = metrics["total_synthetic"]
    ddr = metrics["ddr_count"]
    train_copies = metrics["training_copy_count"]
    hallucinations = metrics["hallucinated_count"]

    # Check: DDR + Training Copies + Hallucinations should equal Total
    computed_total = ddr + train_copies + hallucinations

    if computed_total != total:
        console.print(
            f"[bold red]⚠ Warning:[/bold red] Sanity check failed! "
            f"DDR({ddr}) + Copies({train_copies}) + Hallucinations({hallucinations}) "
            f"= {computed_total} ≠ Total({total})",
            style="red"
        )
    else:
        console.print("✓ Sanity check passed: All records accounted for", style="green")


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

        # Interpretation
        ddr_rate = metrics["ddr_rate"]
        if ddr_rate >= 70:
            quality = "[bold green]EXCELLENT[/bold green]"
        elif ddr_rate >= 50:
            quality = "[bold yellow]GOOD[/bold yellow]"
        elif ddr_rate >= 30:
            quality = "[yellow]MODERATE[/yellow]"
        else:
            quality = "[bold red]POOR[/bold red]"

        console.print(f"Overall Quality: {quality} (DDR = {ddr_rate:.2f}%)")
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
