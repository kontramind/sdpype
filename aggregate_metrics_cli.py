"""
Aggregate Metrics CLI for SDPype

Processes multiple experiment folders, merges their metric CSVs, computes
aggregated statistics with 95% bootstrap confidence intervals for each
generation, and generates visualizations with double-banded CI plots.

Usage:
    python aggregate_metrics_cli.py "./mimic_iii_baseline_dseed*_synthcity_arf_mseed*/"
    python aggregate_metrics_cli.py "./mimic_iii_baseline_dseed*_synthcity_arf_mseed*/" -s results.html -c metrics.csv
    python aggregate_metrics_cli.py "./exp_dseed*_mseed*/" --summary-output /path/to/results.html
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import typer
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(help="Aggregate metrics from multiple experiment folders")
console = Console()


def validate_folder(folder: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate folder structure and check for metrics CSV.

    Expected structure: {folder}/{folder.name}.csv

    Args:
        folder: Path to experiment folder

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not folder.exists():
        return False, f"Folder does not exist: {folder}"

    if not folder.is_dir():
        return False, f"Not a directory: {folder}"

    csv_path = folder / f"{folder.name}.csv"
    if not csv_path.exists():
        return False, f"Missing metrics CSV: {csv_path}"

    return True, None


def load_metrics_csv(folder: Path) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """
    Load metrics CSV from folder.

    Args:
        folder: Path to experiment folder

    Returns:
        Tuple of (success, dataframe, error_message)
    """
    csv_path = folder / f"{folder.name}.csv"

    try:
        df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = {"generation", "factual_total", "ddr_novel_factual"}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            error = f"Missing columns in {csv_path.name}: {', '.join(missing_cols)}"
            return False, None, error

        return True, df, None

    except Exception as e:
        return False, None, f"Error reading {csv_path.name}: {str(e)}"


def compute_percentile_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute 95% percentile bootstrap confidence interval for the mean.

    Args:
        data: Array of metric values
        n_bootstrap: Number of bootstrap resamples (default 10000)
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    n = len(data)
    rng = np.random.RandomState(42)

    # Generate bootstrap resamples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # Percentile-based CI
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


def aggregate_metrics(
    df: pd.DataFrame,
    metrics: List[str] = None,
    n_bootstrap: int = 10000
) -> pd.DataFrame:
    """
    Aggregate metrics by generation with bootstrap CIs.

    Args:
        df: Merged dataframe with all observations
        metrics: List of metric columns to aggregate (default: ['factual_total', 'ddr_novel_factual'])
        n_bootstrap: Number of bootstrap resamples

    Returns:
        DataFrame with aggregated statistics per generation
    """
    if metrics is None:
        metrics = ["factual_total", "ddr_novel_factual"]

    results = []
    generations = sorted(df["generation"].unique())

    for gen in generations:
        gen_data = df[df["generation"] == gen]
        row = {"generation": gen}

        for metric in metrics:
            values = gen_data[metric].values
            n_obs = len(values)

            if n_obs > 0:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if n_obs > 1 else 0.0
                ci_lower, ci_upper = compute_percentile_ci(values, n_bootstrap)

                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
                row[f"{metric}_ci_lower"] = ci_lower
                row[f"{metric}_ci_upper"] = ci_upper
                row[f"{metric}_n_obs"] = n_obs
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci_lower"] = np.nan
                row[f"{metric}_ci_upper"] = np.nan
                row[f"{metric}_n_obs"] = 0

        results.append(row)

    return pd.DataFrame(results)


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """
    Convert hex color to rgba format with alpha transparency.

    Args:
        hex_color: Hex color string (e.g., '#1f77b4')
        alpha: Transparency value 0-1 (default 0.2)

    Returns:
        RGBA color string (e.g., 'rgba(31, 119, 180, 0.2)')
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def create_plotly_visualization(
    summary_df: pd.DataFrame,
    metrics: List[str] = None
) -> go.Figure:
    """
    Create Plotly figure with double-banded CI visualization for each metric.

    Inner band: 95% Bootstrap CI
    Outer band: Mean ± 1.96 × SD (approximately 95% prediction interval)

    Args:
        summary_df: Aggregated metrics dataframe
        metrics: List of metrics to plot

    Returns:
        Plotly Figure object
    """
    if metrics is None:
        metrics = ["factual_total", "ddr_novel_factual"]

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{m.replace('_', ' ').title()}" for m in metrics],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        horizontal_spacing=0.12,
    )

    colors = ["#1f77b4", "#ff7f0e"]  # Blue, Orange

    for col_idx, metric in enumerate(metrics, start=1):
        col = col_idx

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        ci_lower_col = f"{metric}_ci_lower"
        ci_upper_col = f"{metric}_ci_upper"

        df_sorted = summary_df.sort_values("generation")

        # Extract values
        generations = df_sorted["generation"].values
        means = df_sorted[mean_col].values
        stds = df_sorted[std_col].values
        ci_lowers = df_sorted[ci_lower_col].values
        ci_uppers = df_sorted[ci_upper_col].values

        # Compute prediction interval (mean ± 1.96*std)
        pred_lowers = means - 1.96 * stds
        pred_uppers = means + 1.96 * stds

        color = colors[col_idx - 1]
        color_light = hex_to_rgba(color, alpha=0.2)  # Light transparent version

        # Outer band (prediction interval / std band)
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=pred_uppers,
                fill=None,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=pred_lowers,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name=f"{metric}: Mean ± 1.96×SD",
                fillcolor=color_light,
                hovertemplate="<b>Pred. Interval</b><br>Gen %{x}<extra></extra>",
            ),
            row=1,
            col=col,
        )

        # Inner band (bootstrap CI)
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=ci_uppers,
                fill=None,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=ci_lowers,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name=f"{metric}: 95% Bootstrap CI",
                fillcolor=color,
                hovertemplate="<b>95% CI</b><br>Gen %{x}<extra></extra>",
            ),
            row=1,
            col=col,
        )

        # Mean line (drawn last to ensure it's on top)
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=means,
                mode="lines+markers",
                name=f"{metric}: Mean",
                line=dict(color=color, width=4),
                marker=dict(size=8, color=color, line=dict(color="white", width=1)),
                hovertemplate="<b>Mean</b><br>Gen %{x}<br>Value: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=col,
        )

        # Update axes
        fig.update_xaxes(title_text="Generation", row=1, col=col)
        fig.update_yaxes(
            title_text=metric.replace("_", " ").title(),
            row=1,
            col=col,
        )

    # Update layout
    fig.update_layout(
        title_text="Aggregated Metrics with 95% Bootstrap Confidence Intervals",
        height=500,
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
    )

    return fig


def generate_summary_html(
    summary_df: pd.DataFrame,
    fig: go.Figure,
    num_folders: int,
    output_file: Optional[str] = None
) -> str:
    """
    Generate summary HTML with metrics table and Plotly visualization.

    Args:
        summary_df: Aggregated metrics dataframe
        fig: Plotly figure
        num_folders: Number of folders processed
        output_file: Output filename

    Returns:
        Path to generated HTML file
    """
    if output_file is None:
        output_file = "aggregate_results.html"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if output_path.exists():
        output_path.unlink()

    # Get total observations
    total_obs = summary_df[[col for col in summary_df.columns if col.endswith("_n_obs")]].iloc[:, 0].sum()

    # Build summary table HTML
    summary_table_html = summary_df.to_html(
        index=False,
        float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "N/A",
        border=0,
    )

    # Get Plotly figure as HTML
    plot_html = fig.to_html(include_plotlyjs="cdn", div_id="metrics_plot")

    # Build complete HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <title>Aggregate Metrics Results</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }",
        "    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "    h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }",
        "    h2 { color: #555; margin-top: 30px; border-left: 4px solid #007bff; padding-left: 10px; }",
        "    .summary { background: #f0f8ff; padding: 20px; border-radius: 4px; border-left: 4px solid #007bff; margin: 20px 0; }",
        "    .summary p { margin: 10px 0; font-size: 16px; }",
        "    .stat-label { font-weight: 600; color: #333; }",
        "    .stat-value { color: #007bff; font-weight: bold; }",
        "    table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }",
        "    th { background: #f0f0f0; font-weight: 600; padding: 12px; text-align: left; border-bottom: 2px solid #ddd; }",
        "    td { padding: 10px 12px; border-bottom: 1px solid #eee; }",
        "    tr:hover { background: #f9f9f9; }",
        "    .metric-col { font-family: monospace; font-size: 13px; }",
        "    .plot-container { margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 4px; }",
        "    .note { font-size: 13px; color: #666; margin-top: 20px; padding: 15px; background: #fffbea; border-left: 3px solid #ffc107; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h1>Aggregate Metrics Analysis Results</h1>",
        "    <div class='summary'>",
        f"      <p><span class='stat-label'>Folders processed:</span> <span class='stat-value'>{num_folders}</span></p>",
        f"      <p><span class='stat-label'>Total observations:</span> <span class='stat-value'>{int(total_obs)}</span></p>",
        f"      <p><span class='stat-label'>Generations analyzed:</span> <span class='stat-value'>{len(summary_df)}</span></p>",
        "    </div>",
        "    <h2>Metrics Overview</h2>",
        summary_table_html,
        "    <h2>Visualization</h2>",
        "    <div class='plot-container'>",
        plot_html,
        "    </div>",
        "    <div class='note'>",
        "      <strong>Visualization Details:</strong><br>",
        "      <ul>",
        "        <li><strong>Inner band (solid):</strong> 95% percentile bootstrap confidence interval (CI) for the mean</li>",
        "        <li><strong>Outer band (light):</strong> Mean ± 1.96 × standard deviation (approximately 95% prediction interval)</li>",
        "        <li><strong>Line with markers:</strong> Mean value at each generation</li>",
        "      </ul>",
        "      <strong>Bootstrap Method:</strong> 10,000 resamples with empirical percentile-based CI computation",
        "    </div>",
        "  </div>",
        "</body>",
        "</html>",
    ]

    html_content = "\n".join(html_parts)
    output_path.write_text(html_content)

    return str(output_path)


@app.command()
def main(
    pattern: str = typer.Argument(
        ...,
        help="Glob pattern for experiment folders (e.g., './mimic_iii_baseline_dseed*_synthcity_arf_mseed*/')"
    ),
    summary_output: Optional[str] = typer.Option(
        "aggregate_results.html",
        "--summary-output",
        "-s",
        help="Output path for summary HTML"
    ),
    csv_output: Optional[str] = typer.Option(
        "aggregate_results.csv",
        "--csv-output",
        "-c",
        help="Output path for aggregated metrics CSV"
    ),
) -> None:
    """
    Aggregate metrics from multiple experiment folders.

    Discovers all folders matching the glob pattern, loads their metric CSVs,
    merges them, computes aggregated statistics with 95% bootstrap confidence
    intervals for each generation, and generates visualizations.

    Examples:
        python aggregate_metrics_cli.py "./mimic_iii_baseline_dseed*_synthcity_arf_mseed*/"
        python aggregate_metrics_cli.py "./mimic_iii_baseline_dseed*_synthcity_arf_mseed*/" -s results.html -c metrics.csv
        python aggregate_metrics_cli.py "./exp_dseed*_mseed*/" --summary-output /path/to/results.html
    """

    console.print(f"[cyan]Pattern:[/cyan] {pattern}")

    # Expand glob pattern
    folders = sorted(Path().glob(pattern))

    if not folders:
        console.print(f"[red]Error: No folders matched pattern: {pattern}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(folders)} folder(s)[/cyan]\n")

    # Collect dataframes from all folders
    all_dfs = []
    loaded_folders = 0

    with Progress() as progress:
        task = progress.add_task("[cyan]Loading metrics...", total=len(folders))

        for folder in folders:
            progress.update(task, description=f"[cyan]Loading {folder.name}...")

            # Validate folder
            is_valid, error = validate_folder(folder)
            if not is_valid:
                progress.print(f"[red]✗[/red] {folder.name}: {error}")
                raise typer.Exit(1)

            # Load CSV
            success, df, error = load_metrics_csv(folder)
            if not success:
                progress.print(f"[red]✗[/red] {folder.name}: {error}")
                raise typer.Exit(1)

            all_dfs.append(df)
            loaded_folders += 1
            progress.print(f"[green]✓[/green] {folder.name} ({len(df)} rows)")
            progress.advance(task)

    console.print()

    # Merge all dataframes
    console.print("[cyan]Merging data...[/cyan]")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    console.print(f"[green]✓[/green] Merged {len(merged_df)} total observations")

    # Aggregate metrics
    console.print("[cyan]Computing aggregated statistics and bootstrap CIs...[/cyan]")
    summary_df = aggregate_metrics(
        merged_df,
        metrics=["factual_total", "ddr_novel_factual"],
        n_bootstrap=10000
    )
    console.print(f"[green]✓[/green] Computed statistics for {len(summary_df)} generations")

    # Create visualization
    console.print("[cyan]Creating Plotly visualization...[/cyan]")
    fig = create_plotly_visualization(summary_df)
    console.print("[green]✓[/green] Visualization created")

    # Generate summary HTML
    console.print("[cyan]Generating summary HTML...[/cyan]")
    summary_path = generate_summary_html(summary_df, fig, loaded_folders, summary_output)
    console.print(f"[green]✓[/green] Summary HTML generated: {summary_path}")

    # Export CSV
    console.print("[cyan]Exporting metrics CSV...[/cyan]")
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
    summary_df.to_csv(csv_path, index=False)
    console.print(f"[green]✓[/green] Metrics CSV exported: {csv_path}")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [cyan]Folders processed:[/cyan] {loaded_folders}")
    console.print(f"  [cyan]Total observations:[/cyan] {len(merged_df)}")
    console.print(f"  [cyan]Generations analyzed:[/cyan] {len(summary_df)}")
    console.print(f"  [cyan]Metrics analyzed:[/cyan] factual_total, ddr_novel_factual")


if __name__ == "__main__":
    app()
