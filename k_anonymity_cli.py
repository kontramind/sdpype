#!/usr/bin/env python
"""
K-Anonymity Evaluation CLI Tool

Computes k-anonymity metrics for real and synthetic datasets using synthcity's kAnonymization.

Note: Synthcity uses a clustering-based approach to compute k-anonymity, which may differ
from traditional equivalence class-based k-anonymity. Only works with numeric QI columns.
"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_privacy import kAnonymization

console = Console()
app = typer.Typer(add_completion=False)


def interpret_k_value(k: int) -> tuple[str, str]:
    """
    Interpret k-anonymity value and return (interpretation, color).

    Returns:
        tuple of (interpretation_text, rich_color)
    """
    if k >= 10:
        return "Excellent", "green"
    elif k >= 5:
        return "Good", "yellow"
    elif k >= 3:
        return "Moderate", "orange"
    else:
        return "Poor", "red"


@app.command()
def main(
    real_csv: Path = typer.Argument(..., exists=True, help="Path to real dataset CSV."),
    synthetic_csv: Path = typer.Argument(..., exists=True, help="Path to synthetic dataset CSV."),
    qi_cols: str = typer.Option(
        ...,
        "--qi-cols",
        help="Comma-separated quasi-identifier columns.",
    ),
    sep: str = typer.Option(",", "--sep", help="CSV delimiter."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to JSON file (optional)."
    ),
    experiment_name: str = typer.Option(
        "k_anonymity_evaluation",
        "--experiment-name",
        help="Experiment name for output metadata."
    ),
):
    """
    🔒 Compute k-anonymity metrics between real and synthetic datasets.

    Uses synthcity's kAnonymization metric to evaluate privacy protection.
    """

    # Parse QI columns
    qi_list = [c.strip() for c in qi_cols.split(",") if c.strip()]

    console.print(f"\n🔍 K-Anonymity Evaluation", style="bold cyan")
    console.print(f"Real dataset:      {real_csv}")
    console.print(f"Synthetic dataset: {synthetic_csv}")
    console.print(f"QI columns:        {qi_list}\n")

    # Load datasets
    with console.status("[bold blue]Loading datasets..."):
        real_df = pd.read_csv(real_csv, sep=sep)
        syn_df = pd.read_csv(synthetic_csv, sep=sep)

    console.print(f"✅ Loaded real dataset: {real_df.shape[0]:,} rows × {real_df.shape[1]} columns")
    console.print(f"✅ Loaded synthetic dataset: {syn_df.shape[0]:,} rows × {syn_df.shape[1]} columns\n")

    # Validate QI presence
    missing_real = [c for c in qi_list if c not in real_df.columns]
    missing_syn = [c for c in qi_list if c not in syn_df.columns]

    if missing_real:
        console.print(f"❌ QI columns missing from real dataset: {missing_real}", style="red")
        raise typer.Exit(code=1)

    if missing_syn:
        console.print(f"❌ QI columns missing from synthetic dataset: {missing_syn}", style="red")
        raise typer.Exit(code=1)

    # Filter to only QI columns (synthcity approach)
    real_qi = real_df[qi_list]
    syn_qi = syn_df[qi_list]

    # Check if all QI columns are numeric
    non_numeric_cols = []
    for col in qi_list:
        if not pd.api.types.is_numeric_dtype(real_qi[col]) or not pd.api.types.is_numeric_dtype(syn_qi[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        console.print(f"\n⚠️  [yellow]Warning: Non-numeric QI columns detected: {non_numeric_cols}[/yellow]")
        console.print(f"[yellow]Synthcity's kAnonymization requires numeric data only.[/yellow]")
        console.print(f"[yellow]This may cause errors or unexpected results.[/yellow]\n")

    # Compute k-anonymity using synthcity
    console.print("🔒 Computing k-anonymity using synthcity kAnonymization...\n")

    try:
        with console.status("[bold blue]Running synthcity kAnonymization..."):
            # Create dataloaders
            real_loader = GenericDataLoader(real_qi)
            syn_loader = GenericDataLoader(syn_qi)

            # Compute k-anonymity using synthcity
            # The evaluate() method returns {"gt": k_real, "syn": k_syn}
            k_anon_metric = kAnonymization()
            result = k_anon_metric.evaluate(real_loader, syn_loader)

        # Extract results from synthcity
        # Result format: {"gt": <k_real>, "syn": <k_syn>}
        k_real = int(result["gt"])
        k_syn = int(result["syn"])

        console.print(f"✅ Synthcity computation complete\n", style="green")

    except Exception as e:
        console.print(f"\n❌ [red bold]Error: Synthcity kAnonymization failed[/red bold]", style="red")
        console.print(f"[red]Error details: {e}[/red]\n")

        if non_numeric_cols:
            console.print("[yellow]This is likely due to non-numeric QI columns.[/yellow]")
            console.print("[yellow]Synthcity's kAnonymization only supports numeric data.[/yellow]")
            console.print(f"[yellow]Non-numeric columns: {non_numeric_cols}[/yellow]\n")

        raise typer.Exit(code=1)

    # Compute k-ratio
    k_ratio = k_syn / k_real if k_real > 0 else float("inf")

    # Interpret k-values
    real_interp, real_color = interpret_k_value(k_real)
    syn_interp, syn_color = interpret_k_value(k_syn)

    # Display results in Rich table
    results_table = Table(
        title="🔒 K-Anonymity Results (Synthcity)",
        show_header=True,
        header_style="bold blue"
    )
    results_table.add_column("Metric", style="cyan", no_wrap=True)
    results_table.add_column("Value", justify="right", style="white")
    results_table.add_column("Interpretation", style="yellow")

    results_table.add_row(
        "k-anonymity (Real)",
        f"[{real_color}]{k_real}[/{real_color}]",
        real_interp
    )

    results_table.add_row(
        "k-anonymity (Synthetic)",
        f"[{syn_color}]{k_syn}[/{syn_color}]",
        syn_interp
    )

    results_table.add_row(
        "k-ratio (syn/real)",
        f"{k_ratio:.4f}",
        "Higher = better synthetic privacy"
    )

    console.print(results_table)

    # Interpretation guide
    guide_panel = Panel.fit(
        """🔒 Synthcity K-Anonymity Guide:

• k-anonymity (Real) = Privacy protection level of real dataset
• k-anonymity (Synthetic) = Privacy protection level of synthetic dataset
• Higher k values indicate better privacy protection

Interpretation thresholds:
• k ≥ 10:  Excellent privacy protection
• k ≥ 5:   Good privacy protection
• k ≥ 3:   Moderate privacy protection
• k < 3:   Poor privacy protection

• k-ratio = Synthetic k / Real k
• k-ratio > 1.0: Synthetic data provides better privacy than real data
• k-ratio < 1.0: Synthetic data provides worse privacy than real data

Note: Synthcity uses a clustering-based approach, not traditional equivalence classes.
Only supports numeric QI columns.""",
        title="📖 Interpretation Guide",
        border_style="blue"
    )
    console.print("\n")
    console.print(guide_panel)

    # Prepare results for JSON export
    results_dict = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "real_dataset": str(real_csv),
            "synthetic_dataset": str(synthetic_csv),
            "real_shape": list(real_df.shape),
            "synthetic_shape": list(syn_df.shape),
            "qi_columns": qi_list,
            "evaluation_type": "k_anonymity",
            "method": "synthcity_kAnonymization"
        },
        "metrics": {
            "k_anonymization": {
                "status": "success",
                "k_real": int(k_real),
                "k_synthetic": int(k_syn),
                "k_ratio": float(k_ratio),
                "interpretation_real": real_interp,
                "interpretation_synthetic": syn_interp,
                "source": "synthcity"
            }
        }
    }

    # Save to JSON if output path provided
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        console.print(f"\n✅ Results saved to: {output}", style="green")

    console.print("\n🎉 K-anonymity evaluation complete!\n", style="bold green")


if __name__ == "__main__":
    app()
