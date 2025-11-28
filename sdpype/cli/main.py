"""
SDPype - Synthetic Data Pipeline CLI Entry Point
"""

import typer
from rich.console import Console
from typing import Optional
from pathlib import Path

console = Console()

app = typer.Typer(
    help="🚀 Synthetic Data Pipeline - Monolithic Repository with Experiment Versioning",
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Import and add model subcommands
from sdpype.cli.model import model_app
from sdpype.cli.metrics import metrics_app
from sdpype.cli.downstream import downstream_app
app.add_typer(model_app, name="model")
app.add_typer(metrics_app, name="metrics")
app.add_typer(downstream_app, name="downstream")


@app.command()
def setup():
    """🏗️ Setup repository for experiments"""
    from sdpype.core.setup import setup_repository_command
    setup_repository_command()

@app.command()
def purge(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    keep_raw_data: bool = typer.Option(True, "--keep-raw-data", help="Preserve raw data files"),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Preserve DVC cache"),
):
    """🧹 Purge all experiments, models, and DVC state (DESTRUCTIVE!)"""
    from sdpype.core.experiment import purge_repository
    purge_repository(confirm, keep_raw_data, keep_cache)

@app.command()
def pipeline(
    force: bool = typer.Option(False, "--force", help="Force rerun all stages"),
):
    """🚀 Run the complete DVC pipeline (train → generate → statistical_similarity → detection_evaluation)"""
    from sdpype.core.pipeline import run_pipeline_command
    run_pipeline_command(force)

@app.command()
def stage(
    stage_name: str = typer.Argument(..., help="Stage to run (e.g., train_sdg, preprocess)"),
    force: bool = typer.Option(False, "--force", help="Force rerun this stage"),
):
    """🎯 Run a specific pipeline stage"""
    from sdpype.core.pipeline import run_stage_command
    run_stage_command(stage_name, force)

@app.command()
def status():
    """📊 Show repository status and experiment summary"""
    from sdpype.core.status import show_repository_status
    show_repository_status()

@app.command()
def models(
    library: Optional[str] = typer.Option(None, "--library", help="Filter by library (sdv, synthcity)"),
    show_params: Optional[str] = typer.Option(None, "--params", help="Show hyperparameters for specific model (e.g., ctgan, sdv/ctgan)"),
):
    """🤖 Show available synthetic data generation models"""
    from sdpype.core.models import show_available_models
    show_available_models(library, show_params)

@app.command()
def metadata(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save metadata to file"),
    table_name: str = typer.Option("data", "--table-name", help="Name for the table"),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip displaying summary"),
):
    """
    🔍 Auto-detect and save SDV metadata for a dataset.

    Uses SDV's auto-detection to analyze your dataset and generate
    the metadata required for synthetic data generation.
    """
    from sdpype.metadata import detect_metadata

    if not data_path.exists():
        console.print(f"❌ Data file not found: {data_path}", style="red")
        raise typer.Exit(1)

    # Default output path if not specified
    if output is None:
        output = data_path.parent / f"{data_path.stem}_metadata.json"

    detect_metadata(
        data_path=data_path,
        output_path=output,
        table_name=table_name,
        show_summary=not no_summary
    )

if __name__ == "__main__":
    app()
