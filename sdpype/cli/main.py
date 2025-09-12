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

if __name__ == "__main__":
    app()
