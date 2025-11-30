"""
CLI for downstream task training
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Create downstream app
downstream_app = typer.Typer(help="Train downstream task models on synthetic data")

console = Console()


@downstream_app.command(name="mimic-iii-readmission")
def train_mimic_iii_readmission(
    train_data: Path = typer.Option(
        ...,
        "--train-data",
        "-t",
        help="Path to training data CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    test_data: Path = typer.Option(
        ...,
        "--test-data",
        "-e",
        help="Path to test data CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target_column: str = typer.Option(
        "IS_READMISSION_30D",
        "--target",
        "-c",
        help="Name of target column for readmission prediction",
    ),
    encoding_config: Optional[Path] = typer.Option(
        None,
        "--encoding-config",
        help="Path to RDT encoding config YAML (same format as SDG pipeline)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    n_trials: int = typer.Option(
        100,
        "--n-trials",
        "-n",
        help="Number of Bayesian optimization trials",
        min=1,
    ),
    n_folds: int = typer.Option(
        5,
        "--n-folds",
        "-k",
        help="Number of cross-validation folds",
        min=2,
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="Maximum time in seconds for optimization (None = no limit)",
        min=1,
    ),
    val_split: float = typer.Option(
        0.2,
        "--val-split",
        "-v",
        help="Fraction of training data for validation in final model",
        min=0.0,
        max=0.5,
    ),
    random_state: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    output_dir: Path = typer.Option(
        Path("experiments/models/downstream"),
        "--output-dir",
        "-o",
        help="Output directory for models and metrics",
    ),
):
    """
    Train LGBM binary classifier for MIMIC-III 30-day readmission prediction

    This command trains a LightGBM model with Bayesian hyperparameter optimization
    using Optuna. The model predicts 30-day hospital readmission risk.

    Example usage:
        sdpype downstream mimic-iii-readmission \\
            --train-data experiments/data/processed/train.csv \\
            --test-data experiments/data/processed/test.csv \\
            --n-trials 100 \\
            --timeout 3600

    The optimization uses:
    - Tree-structured Parzen Estimator (TPE) for Bayesian optimization
    - Stratified K-fold cross-validation
    - AUROC as the primary optimization metric
    - Early stopping to prevent overfitting
    """
    from sdpype.core.downstream import train_readmission_model

    try:
        console.print("\n[bold]MIMIC-III 30-Day Readmission Prediction[/bold]")
        console.print("=" * 60)

        # Train model
        results = train_readmission_model(
            train_file=train_data,
            test_file=test_data,
            target_column=target_column,
            n_trials=n_trials,
            n_folds=n_folds,
            timeout=timeout,
            random_state=random_state,
            output_dir=output_dir,
            val_split=val_split,
            encoding_config=encoding_config,
        )

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Training completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@downstream_app.command(name="mimic-iii-valuation")
def data_valuation_mimic_iii(
    train_data: Path = typer.Option(
        ...,
        "--train-data",
        "-t",
        help="Path to synthetic training data CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    test_data: Path = typer.Option(
        ...,
        "--test-data",
        "-e",
        help="Path to real test data CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target_column: str = typer.Option(
        "IS_READMISSION_30D",
        "--target",
        "-c",
        help="Name of target column for readmission prediction",
    ),
    encoding_config: Optional[Path] = typer.Option(
        None,
        "--encoding-config",
        help="Path to RDT encoding config YAML (same format as SDG pipeline)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    num_samples: int = typer.Option(
        100,
        "--num-samples",
        "-n",
        help="Number of Monte Carlo samples for Shapley approximation (higher=more accurate but slower)",
        min=10,
    ),
    random_state: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    output_dir: Path = typer.Option(
        Path("experiments/data_valuation"),
        "--output-dir",
        "-o",
        help="Output directory for valuation results",
    ),
    include_features: bool = typer.Option(
        True,
        "--include-features/--no-include-features",
        help="Include all features in output CSV (useful for analysis)",
    ),
):
    """
    Data Shapley valuation for synthetic training data

    This command computes Data Shapley values for each synthetic training sample
    to identify potential hallucinations. A negative Shapley value indicates that
    a sample hurts model performance on real test data.

    The output CSV contains each synthetic record with its Shapley value, sorted
    by harmfulness (most harmful first). You can use this to:
    - Identify and remove hallucinations
    - Understand which synthetic samples are valuable
    - Analyze data quality at a granular level

    Example usage:
        sdpype downstream mimic-iii-valuation \\
            --train-data experiments/data/synthetic/train.csv \\
            --test-data experiments/data/real/test.csv \\
            --num-samples 100 \\
            --output-dir experiments/data_valuation

    The method uses:
    - Data Shapley with Truncated Monte Carlo Sampling (TMCS)
    - LightGBM as the base model
    - AUROC on real test data as the utility metric
    """
    from sdpype.core.data_valuation import run_data_valuation

    try:
        console.print("\n[bold]MIMIC-III Data Shapley Valuation[/bold]")
        console.print("=" * 60)

        # Run data valuation
        results = run_data_valuation(
            train_file=train_data,
            test_file=test_data,
            target_column=target_column,
            num_samples=num_samples,
            random_state=random_state,
            output_dir=output_dir,
            lgbm_params=None,  # Use defaults
            encoding_config=encoding_config,
            include_features=include_features,
        )

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Data valuation completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)
