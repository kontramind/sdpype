# sdpype/cli/metrics.py
"""Metrics management CLI commands"""

from omegaconf import OmegaConf, DictConfig
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
from pathlib import Path

console = Console()

metrics_app = typer.Typer(
    help="📊 Manage evaluation metrics",
    rich_markup_mode="rich",
    no_args_is_help=False
)

@metrics_app.callback(invoke_without_command=True)
def metrics_main(ctx: typer.Context):
    """📋 Show available evaluation metrics"""
    if ctx.invoked_subcommand is None:
        # Default behavior - show metrics
        from sdpype.core.metrics import show_available_metrics
        show_available_metrics()

@metrics_app.command("info") 
def metric_info(
    metric_name: str = typer.Argument(..., help="Metric name (e.g., alpha_precision, prdc_score)")
):
    """📊 Show detailed information about a specific metric"""
    from sdpype.core.metrics import show_metric_details
    show_metric_details(metric_name)
