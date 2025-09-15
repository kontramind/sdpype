# sdpype/cli/main.py - Just setup + purge for testing
"""
SDPype - Synthetic Data Pipeline CLI Entry Point
"""

import typer
from rich.console import Console
from typing import Optional

console = Console()

app = typer.Typer(
    help="🚀 Synthetic Data Pipeline - Monolithic Repository with Experiment Versioning",
    rich_markup_mode="rich",
    no_args_is_help=True
)

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
    """🚀 Run the complete DVC pipeline"""
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


if __name__ == "__main__":
    app()
