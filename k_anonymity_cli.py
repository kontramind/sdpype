#!/usr/bin/env python
"""
K-Anonymity Evaluation CLI Tool

Computes k-anonymity metrics for multiple datasets using synthcity's kAnonymization.
Supports population, reference, training, and synthetic datasets.

Note: Synthcity uses a clustering-based approach to compute k-anonymity, which may differ
from traditional equivalence class-based k-anonymity.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sklearn.preprocessing import LabelEncoder

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


def interpret_ratio(ratio: float) -> str:
    """Interpret k-ratio value."""
    if ratio > 1.5:
        return "Much better privacy"
    elif ratio > 1.1:
        return "Better privacy"
    elif ratio > 0.9:
        return "Similar privacy"
    elif ratio > 0.7:
        return "Worse privacy"
    else:
        return "Much worse privacy"


@app.command()
def main(
    population: Optional[Path] = typer.Option(None, "--population", help="Path to population dataset CSV."),
    reference: Optional[Path] = typer.Option(None, "--reference", help="Path to reference dataset CSV."),
    training: Optional[Path] = typer.Option(None, "--training", help="Path to training dataset CSV."),
    synthetic: Optional[Path] = typer.Option(None, "--synthetic", help="Path to synthetic dataset CSV."),
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
    🔒 Compute k-anonymity metrics across multiple datasets.

    Supports population, reference, training, and synthetic datasets.
    Computes k-anonymity values for all provided datasets and k-ratios between relevant pairs.
    """

    # Parse QI columns
    qi_list = [c.strip() for c in qi_cols.split(",") if c.strip()]

    # Collect provided datasets
    datasets_config = {
        "population": population,
        "reference": reference,
        "training": training,
        "synthetic": synthetic
    }

    provided_datasets = {name: path for name, path in datasets_config.items() if path is not None}

    # Validate at least 2 datasets provided
    if len(provided_datasets) < 2:
        console.print("❌ [red]Error: At least 2 datasets must be provided[/red]\n", style="red")
        console.print("Available options:")
        console.print("  --population <path>")
        console.print("  --reference <path>")
        console.print("  --training <path>")
        console.print("  --synthetic <path>")
        raise typer.Exit(code=1)

    console.print(f"\n🔍 K-Anonymity Evaluation", style="bold cyan")
    console.print(f"Datasets provided: {', '.join(provided_datasets.keys())}")
    console.print(f"QI columns: {qi_list}\n")

    # Load all datasets
    datasets = {}
    with console.status("[bold blue]Loading datasets..."):
        for name, path in provided_datasets.items():
            if not path.exists():
                console.print(f"❌ File not found: {path}", style="red")
                raise typer.Exit(code=1)
            datasets[name] = pd.read_csv(path, sep=sep)
            console.print(f"✅ Loaded {name}: {datasets[name].shape[0]:,} rows × {datasets[name].shape[1]} columns")

    console.print()

    # Validate QI columns exist in all datasets
    for name, df in datasets.items():
        missing_cols = [c for c in qi_list if c not in df.columns]
        if missing_cols:
            console.print(f"❌ QI columns missing from {name} dataset: {missing_cols}", style="red")
            raise typer.Exit(code=1)

    # Filter to only QI columns
    datasets_qi = {name: df[qi_list].copy() for name, df in datasets.items()}

    # Apply label encoding for categorical columns across ALL datasets
    categorical_cols = []
    label_encoders = {}

    console.print("🔄 Checking for categorical columns...\n")

    for col in qi_list:
        # Check if any dataset has this column as non-numeric
        is_categorical = any(
            not pd.api.types.is_numeric_dtype(df[col])
            for df in datasets_qi.values()
        )

        if is_categorical:
            categorical_cols.append(col)
            console.print(f"🔄 Encoding categorical column: {col}")

            # Fit encoder on combined data from all datasets
            le = LabelEncoder()
            combined_values = pd.concat([df[col] for df in datasets_qi.values()]).astype(str)
            le.fit(combined_values)

            # Transform all datasets
            for name in datasets_qi.keys():
                datasets_qi[name][col] = le.transform(datasets_qi[name][col].astype(str))

            label_encoders[col] = le
            console.print(f"  → Encoded {len(le.classes_)} categories: {list(le.classes_)}\n")

    if categorical_cols:
        console.print(f"✅ Encoded {len(categorical_cols)} categorical column(s): {categorical_cols}\n")

    # Compute k-anonymity for each dataset using synthcity
    console.print("🔒 Computing k-anonymity using synthcity kAnonymization...\n")

    k_values = {}
    k_anon_metric = kAnonymization()

    try:
        for name, df_qi in datasets_qi.items():
            with console.status(f"[bold blue]Computing k-anonymity for {name}..."):
                loader = GenericDataLoader(df_qi)
                # Use evaluate_data() for single dataset
                k = k_anon_metric.evaluate_data(loader)
                k_values[name] = int(k)
                console.print(f"✅ {name}: k = {k}")

        console.print(f"\n✅ All k-anonymity computations complete\n", style="green")

    except Exception as e:
        console.print(f"\n❌ [red bold]Error: Synthcity kAnonymization failed[/red bold]", style="red")
        console.print(f"[red]Error details: {e}[/red]\n")
        raise typer.Exit(code=1)

    # Compute k-ratios for relevant pairs
    k_ratios = {}

    ratio_pairs = [
        ("reference", "population", "Reference / Population"),
        ("synthetic", "population", "Synthetic / Population"),
        ("synthetic", "reference", "Synthetic / Reference"),
        ("synthetic", "training", "Synthetic / Training"),
    ]

    for numerator, denominator, label in ratio_pairs:
        if numerator in k_values and denominator in k_values:
            ratio = k_values[numerator] / k_values[denominator] if k_values[denominator] > 0 else float("inf")
            k_ratios[label] = ratio

    # Display K-Anonymity Values Table
    k_values_table = Table(
        title="🔒 K-Anonymity Values",
        show_header=True,
        header_style="bold blue"
    )
    k_values_table.add_column("Dataset", style="cyan", no_wrap=True)
    k_values_table.add_column("k-anonymity", justify="right", style="white")
    k_values_table.add_column("Interpretation", style="yellow")

    for name in ["population", "reference", "training", "synthetic"]:
        if name in k_values:
            k = k_values[name]
            interp, color = interpret_k_value(k)
            k_values_table.add_row(
                name.capitalize(),
                f"[{color}]{k}[/{color}]",
                interp
            )

    console.print(k_values_table)
    console.print()

    # Display K-Ratios Table
    if k_ratios:
        k_ratios_table = Table(
            title="📊 K-Anonymity Ratios",
            show_header=True,
            header_style="bold blue"
        )
        k_ratios_table.add_column("Comparison", style="cyan")
        k_ratios_table.add_column("Ratio", justify="right", style="white")
        k_ratios_table.add_column("Interpretation", style="yellow")

        for label, ratio in k_ratios.items():
            ratio_interp = interpret_ratio(ratio)
            # Color code ratios: green if > 1 (better), red if < 1 (worse)
            color = "green" if ratio > 1.0 else "red" if ratio < 0.9 else "yellow"
            k_ratios_table.add_row(
                label,
                f"[{color}]{ratio:.4f}[/{color}]",
                ratio_interp
            )

        console.print(k_ratios_table)
        console.print()

    # Interpretation guide
    guide_panel = Panel.fit(
        """🔒 K-Anonymity Evaluation Guide:

K-Anonymity Values:
• Higher k values indicate better privacy protection
• k ≥ 10: Excellent privacy protection
• k ≥ 5:  Good privacy protection
• k ≥ 3:  Moderate privacy protection
• k < 3:  Poor privacy protection

K-Ratios:
• Ratio > 1.0: Numerator dataset has better privacy than denominator
• Ratio < 1.0: Numerator dataset has worse privacy than denominator
• Synthetic / Population: Overall privacy improvement from synthesis
• Synthetic / Reference: Privacy protection on holdout data
• Synthetic / Training: Direct comparison with training data

Note: Synthcity uses a clustering-based approach, not traditional equivalence classes.
Categorical columns are automatically label-encoded before evaluation.""",
        title="📖 Interpretation Guide",
        border_style="blue"
    )
    console.print(guide_panel)

    # Prepare results for JSON export
    results_dict = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "datasets": {
                name: {
                    "path": str(path),
                    "shape": list(datasets[name].shape)
                }
                for name, path in provided_datasets.items()
            },
            "qi_columns": qi_list,
            "categorical_columns": categorical_cols,
            "encoded_categories": {
                col: list(label_encoders[col].classes_)
                for col in categorical_cols
            } if categorical_cols else {},
            "evaluation_type": "k_anonymity",
            "method": "synthcity_kAnonymization"
        },
        "metrics": {
            "k_values": {
                name: {
                    "k": int(k),
                    "interpretation": interpret_k_value(k)[0]
                }
                for name, k in k_values.items()
            },
            "k_ratios": {
                label: {
                    "ratio": float(ratio),
                    "interpretation": interpret_ratio(ratio)
                }
                for label, ratio in k_ratios.items()
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
