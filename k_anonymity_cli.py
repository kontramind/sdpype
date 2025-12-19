#!/usr/bin/env python
"""
K-Anonymity Evaluation CLI Tool

Computes k-anonymity metrics for real and synthetic datasets using synthcity.
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


def compute_k_anonymity_manual(df: pd.DataFrame, qi_cols: List[str]) -> int:
    """
    Manually compute k-anonymity for validation.

    k-anonymity is the minimum equivalence class size when grouping by QI columns.
    """
    ec_sizes = df.groupby(qi_cols, dropna=False).size()
    if ec_sizes.empty:
        raise ValueError("No equivalence classes found. Check QI columns.")
    return int(ec_sizes.min())


def compute_equivalence_classes(df: pd.DataFrame, qi_cols: List[str]) -> pd.Series:
    """Return equivalence class sizes as a Pandas Series."""
    return df.groupby(qi_cols, dropna=False).size()


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

    # Compute k-anonymity using synthcity
    console.print("🔒 Computing k-anonymity using synthcity...\n")

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
        k_real_synthcity = int(result["gt"]) if "gt" in result else None
        k_syn_synthcity = int(result["syn"]) if "syn" in result else None

        # Note: synthcity adds epsilon (1e-8) to synthetic score, but int() will floor it
        console.print(f"✅ Synthcity computation complete\n", style="green")

    except Exception as e:
        console.print(f"❌ Error computing synthcity k-anonymity: {e}", style="red")
        console.print("\n⚠️  Falling back to manual calculation only...\n", style="yellow")
        k_real_synthcity = None
        k_syn_synthcity = None

    # Also compute manual k-anonymity for validation
    with console.status("[bold blue]Computing manual k-anonymity for validation..."):
        real_ec = compute_equivalence_classes(real_qi, qi_list)
        syn_ec = compute_equivalence_classes(syn_qi, qi_list)

        k_real_manual = int(real_ec.min())
        k_syn_manual = int(syn_ec.min())

    # Use synthcity values if available, otherwise use manual
    k_real = k_real_synthcity if k_real_synthcity is not None else k_real_manual
    k_syn = k_syn_synthcity if k_syn_synthcity is not None else k_syn_manual

    # Compute k-ratio
    k_ratio = k_syn / k_real if k_real > 0 else float("inf")

    # Interpret k-values
    real_interp, real_color = interpret_k_value(k_real)
    syn_interp, syn_color = interpret_k_value(k_syn)

    # Display results in Rich table
    results_table = Table(
        title="🔒 K-Anonymity Results",
        show_header=True,
        header_style="bold blue"
    )
    results_table.add_column("Metric", style="cyan", no_wrap=True)
    results_table.add_column("Real Dataset", justify="right")
    results_table.add_column("Synthetic Dataset", justify="right")
    results_table.add_column("Interpretation", style="yellow")

    results_table.add_row(
        "k-anonymity",
        f"[{real_color}]{k_real}[/{real_color}]",
        f"[{syn_color}]{k_syn}[/{syn_color}]",
        f"Real: {real_interp} | Syn: {syn_interp}"
    )

    results_table.add_row(
        "Equivalence Classes",
        f"{len(real_ec):,}",
        f"{len(syn_ec):,}",
        ""
    )

    results_table.add_row(
        "k-ratio (syn/real)",
        "",
        f"{k_ratio:.4f}",
        "Higher means better synthetic privacy"
    )

    console.print(results_table)

    # Validation table (compare synthcity vs manual)
    if k_real_synthcity is not None and k_syn_synthcity is not None:
        validation_table = Table(
            title="🔍 Validation: Synthcity vs Manual",
            show_header=True,
            header_style="bold magenta"
        )
        validation_table.add_column("Method", style="cyan")
        validation_table.add_column("Real k", justify="right")
        validation_table.add_column("Synthetic k", justify="right")
        validation_table.add_column("Match", justify="center")

        real_match = "✅" if k_real_synthcity == k_real_manual else "❌"
        syn_match = "✅" if k_syn_synthcity == k_syn_manual else "❌"

        validation_table.add_row(
            "Synthcity",
            str(k_real_synthcity),
            str(k_syn_synthcity),
            ""
        )
        validation_table.add_row(
            "Manual",
            str(k_real_manual),
            str(k_syn_manual),
            ""
        )
        validation_table.add_row(
            "Match?",
            real_match,
            syn_match,
            ""
        )

        console.print("\n")
        console.print(validation_table)

    # Interpretation guide
    guide_panel = Panel.fit(
        """🔒 K-Anonymity Guide:

• k-anonymity = Minimum equivalence class size (privacy protection level)
• k ≥ 10:  Excellent privacy protection
• k ≥ 5:   Good privacy protection
• k ≥ 3:   Moderate privacy protection
• k < 3:   Poor privacy protection

• k-ratio = How much better synthetic data protects privacy vs real data
• k-ratio > 1.0: Synthetic data provides better privacy than real data
• k-ratio < 1.0: Synthetic data provides worse privacy than real data

• Equivalence Classes = Number of unique QI combinations in dataset""",
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
            "evaluation_type": "k_anonymity"
        },
        "metrics": {
            "k_anonymization": {
                "status": "success",
                "k_real": int(k_real),
                "k_synthetic": int(k_syn),
                "k_ratio": float(k_ratio),
                "equivalence_classes_real": int(len(real_ec)),
                "equivalence_classes_synthetic": int(len(syn_ec)),
                "interpretation_real": real_interp,
                "interpretation_synthetic": syn_interp,
                "source": "synthcity" if k_real_synthcity is not None else "manual"
            }
        }
    }

    # Add validation info if available
    if k_real_synthcity is not None and k_syn_synthcity is not None:
        results_dict["metrics"]["k_anonymization"]["validation"] = {
            "synthcity_k_real": int(k_real_synthcity),
            "synthcity_k_synthetic": int(k_syn_synthcity),
            "manual_k_real": int(k_real_manual),
            "manual_k_synthetic": int(k_syn_manual),
            "synthcity_manual_match": (k_real_synthcity == k_real_manual and
                                       k_syn_synthcity == k_syn_manual)
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
