"""
Aggregate Hybrid Stratified Metrics CLI for SDPype

Processes multiple experiment folders using a hybrid stratified aggregation strategy
to properly handle hierarchical data structure (dseed → mseed → generation).

This approach:
1. Averages across model seeds (mseed) within each data subset (dseed)
2. Computes t-based confidence intervals on dseed averages
3. Handles small sample sizes correctly using t-distribution

Usage:
    python aggregate_hybrid_stratified_metrics_cli.py "./mimic_iii_baseline_dseed*_synthcity_arf_mseed*/"
    python aggregate_hybrid_stratified_metrics_cli.py "./mimic_iii_baseline_dseed*_synthcity_arf_mseed*/" -s results.html -c metrics.csv -i dseed_avgs.csv
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import typer
from rich.console import Console
from rich.progress import Progress
from scipy import stats

app = typer.Typer(help="Aggregate metrics using hybrid stratified strategy")
console = Console()


def parse_folder_name(folder_name: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Parse dseed and mseed from folder name.

    Expected pattern: {prefix}_dseed{NUMBER}_{infix}_mseed{NUMBER}
    Example: mimic_iii_baseline_dseed001_synthcity_arf_mseed042

    Args:
        folder_name: Name of the experiment folder

    Returns:
        Tuple of (dseed, mseed, error_message)
    """
    pattern = r'.*_dseed(\d+)_.*_mseed(\d+).*'
    match = re.match(pattern, folder_name)

    if not match:
        return None, None, f"Folder name does not match pattern 'dseed{{N}}_mseed{{N}}': {folder_name}"

    dseed = int(match.group(1))
    mseed = int(match.group(2))

    return dseed, mseed, None


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
        required_cols = {"generation", "factual_total", "ddr_novel_factual", "ks_complement", "tv_complement",
                         "wasserstein_dist", "jsd_syndat", "mmd", "alpha_delta_precision_OC",
                         "alpha_delta_coverage_OC", "alpha_authenticity_OC", "prdc_precision",
                         "prdc_recall", "prdc_density", "prdc_coverage", "detection_gmm",
                         "detection_xgb", "detection_mlp", "detection_linear"}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            error = f"Missing required columns in {csv_path.name}: {', '.join(missing_cols)}"
            return False, None, error

        return True, df, None

    except Exception as e:
        return False, None, f"Error reading {csv_path.name}: {str(e)}"


def validate_hierarchical_structure(
    data_structure: Dict[int, Dict[int, pd.DataFrame]]
) -> Tuple[bool, Optional[str]]:
    """
    Validate that all dseeds have the same number of mseeds and complete generation coverage.

    Args:
        data_structure: Nested dict {dseed: {mseed: dataframe}}

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data_structure:
        return False, "No data loaded"

    # Check: All dseeds have same number of mseeds
    dseeds = sorted(data_structure.keys())
    mseed_counts = {dseed: len(data_structure[dseed]) for dseed in dseeds}

    expected_mseeds = mseed_counts[dseeds[0]]
    for dseed, count in mseed_counts.items():
        if count != expected_mseeds:
            return False, f"Inconsistent mseeds: expected {expected_mseeds}, but dseed {dseed} has {count}"

    # Check: All dseed-mseed combinations have the same generations
    all_generations = set()
    first_key = None

    for dseed in dseeds:
        for mseed, df in data_structure[dseed].items():
            generations = set(df["generation"].unique())

            if first_key is None:
                first_key = (dseed, mseed)
                all_generations = generations
            else:
                missing = all_generations - generations
                if missing:
                    return False, f"Missing generation {sorted(missing)[0]} in folder: dseed{dseed}_mseed{mseed}"

                extra = generations - all_generations
                if extra:
                    return False, f"Unexpected generation {sorted(extra)[0]} in folder: dseed{dseed}_mseed{mseed}"

    return True, None


def compute_dseed_averages(
    data_structure: Dict[int, Dict[int, pd.DataFrame]],
    metrics: List[str]
) -> pd.DataFrame:
    """
    Step 1: Average mseed values within each dseed for each generation.

    Args:
        data_structure: Nested dict {dseed: {mseed: dataframe}}
        metrics: List of metric columns to process

    Returns:
        DataFrame in long format: generation, dseed, metric_name, mean, n_mseeds
    """
    results = []

    for dseed, mseed_dict in data_structure.items():
        # Get all generations (same across all mseeds due to validation)
        first_df = next(iter(mseed_dict.values()))
        generations = sorted(first_df["generation"].unique())

        for gen in generations:
            for metric in metrics:
                # Collect all mseed values for this dseed, generation, metric
                values = []
                for mseed, df in mseed_dict.items():
                    gen_data = df[df["generation"] == gen]
                    if len(gen_data) > 0:
                        values.append(gen_data[metric].iloc[0])

                if values:
                    results.append({
                        "generation": gen,
                        "dseed": dseed,
                        "metric_name": metric,
                        "mean": np.mean(values),
                        "n_mseeds": len(values)
                    })

    return pd.DataFrame(results)


def compute_stratified_statistics(
    dseed_averages: pd.DataFrame,
    metrics: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Step 2: Compute t-based confidence intervals on dseed averages.

    Args:
        dseed_averages: DataFrame with dseed-level averages (from compute_dseed_averages)
        metrics: List of metrics to process
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        DataFrame with aggregated statistics per generation
    """
    results = []
    generations = sorted(dseed_averages["generation"].unique())

    for gen in generations:
        gen_data = dseed_averages[dseed_averages["generation"] == gen]
        row = {"generation": gen}

        for metric in metrics:
            metric_data = gen_data[gen_data["metric_name"] == metric]
            values = metric_data["mean"].values
            n_dseeds = len(values)

            if n_dseeds > 0:
                mean_val = np.mean(values)

                if n_dseeds > 1:
                    std_val = np.std(values, ddof=1)
                    se = std_val / np.sqrt(n_dseeds)
                    df = n_dseeds - 1

                    # T-critical value for 95% CI
                    t_crit = stats.t.ppf(1 - alpha / 2, df)
                    ci_lower = mean_val - t_crit * se
                    ci_upper = mean_val + t_crit * se
                else:
                    std_val = 0.0
                    se = 0.0
                    df = 0
                    t_crit = np.nan
                    ci_lower = mean_val
                    ci_upper = mean_val

                # Get total observations (sum of all n_mseeds)
                total_obs = metric_data["n_mseeds"].sum()

                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
                row[f"{metric}_se"] = se
                row[f"{metric}_ci_lower"] = ci_lower
                row[f"{metric}_ci_upper"] = ci_upper
                row[f"{metric}_n_dseeds"] = n_dseeds
                row[f"{metric}_df"] = df
                row[f"{metric}_t_crit"] = t_crit
                row[f"{metric}_total_obs"] = total_obs
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_se"] = np.nan
                row[f"{metric}_ci_lower"] = np.nan
                row[f"{metric}_ci_upper"] = np.nan
                row[f"{metric}_n_dseeds"] = 0
                row[f"{metric}_df"] = 0
                row[f"{metric}_t_crit"] = np.nan
                row[f"{metric}_total_obs"] = 0

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
    metrics: List[str] = None,
    n_dseeds: int = None
) -> List[go.Figure]:
    """
    Create individual Plotly figures with double-banded CI visualization for each metric.

    Inner band: 95% t-based CI for the mean
    Outer band: 95% Prediction Interval (where a new dseed would likely fall)

    Args:
        summary_df: Aggregated metrics dataframe
        metrics: List of metrics to plot
        n_dseeds: Number of data subsets (needed for prediction interval calculation)

    Returns:
        List of Plotly Figure objects (one per metric)
    """
    if metrics is None:
        metrics = ["factual_total", "ddr_novel_factual", "ks_complement", "tv_complement",
                   "wasserstein_dist", "jsd_syndat", "mmd", "alpha_delta_precision_OC",
                   "alpha_delta_coverage_OC", "alpha_authenticity_OC", "prdc_precision",
                   "prdc_recall", "prdc_density", "prdc_coverage", "detection_gmm",
                   "detection_xgb", "detection_mlp", "detection_linear"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#ff9896", "#9edae5", "#c5b0d5", "#c49c94", "#f1c40f", "#16a085", "#e91e63", "#808000"]
    # Blue, Orange, Green, Purple, Red, Cyan, Brown, Pink, Gray, Lime, Light Red, Light Cyan, Light Purple, Light Brown, Gold, Teal, Magenta, Olive
    line_colors = ["#0d3b7a", "#d64a0a", "#1a6b1a", "#5a3a7a", "#8b1a1a", "#0a7a8a", "#5a3525", "#a03a82", "#4a4a4a", "#7a7d15", "#cc4c4c", "#5a9aa8", "#8a7aa8", "#8a6a5a", "#c29d0b", "#117a65", "#ad1457", "#5a5a00"]
    # Dark versions for lines
    gray_color = "#a6a6a6"  # Gray for outer bands

    figures = []

    for metric_idx, metric in enumerate(metrics):
        fig = go.Figure()

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        ci_lower_col = f"{metric}_ci_lower"
        ci_upper_col = f"{metric}_ci_upper"
        t_crit_col = f"{metric}_t_crit"

        df_sorted = summary_df.sort_values("generation")

        # Extract values
        generations = df_sorted["generation"].values
        means = df_sorted[mean_col].values
        stds = df_sorted[std_col].values
        ci_lowers = df_sorted[ci_lower_col].values
        ci_uppers = df_sorted[ci_upper_col].values
        t_crits = df_sorted[t_crit_col].values

        # Compute 95% Prediction Interval
        # PI = mean ± t_crit × SD × sqrt(1 + 1/n)
        # This is where a NEW dseed average would likely fall
        pi_multiplier = np.sqrt(1 + 1/n_dseeds) if n_dseeds else 1.0
        pred_lowers = means - t_crits * stds * pi_multiplier
        pred_uppers = means + t_crits * stds * pi_multiplier

        color = colors[metric_idx]
        line_color = line_colors[metric_idx]

        # Convert colors to rgba with transparency
        outer_band_color = hex_to_rgba(gray_color, alpha=0.75)  # Gray outer band
        inner_band_color = hex_to_rgba(color, alpha=0.50)  # Medium inner band

        # Outer band (Prediction Interval) - gray color
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
        )

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=pred_lowers,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name="95% Prediction Interval",
                fillcolor=outer_band_color,
                hovertemplate="<b>Prediction Interval</b><br>Gen %{x}<extra></extra>",
            ),
        )

        # Inner band (t-based CI) - using metric color with transparency
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
        )

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=ci_lowers,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name="95% t-based CI",
                fillcolor=inner_band_color,
                hovertemplate="<b>95% CI</b><br>Gen %{x}<extra></extra>",
            ),
        )

        # Mean line (drawn last to ensure it's on top)
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=means,
                mode="lines+markers",
                name="Mean",
                line=dict(color=line_color, width=4),
                marker=dict(size=8, color=line_color, line=dict(color="white", width=1)),
                hovertemplate="<b>Mean</b><br>Gen %{x}<br>Value: %{y:.4f}<extra></extra>",
            ),
        )

        # Update layout for individual figure
        fig.update_layout(
            title_text=f"{metric.replace('_', ' ').title()} with 95% t-based Confidence Intervals",
            xaxis_title="Generation",
            yaxis_title=metric.replace("_", " ").title(),
            height=500,
            hovermode="x unified",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=11),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
            ),
        )

        figures.append(fig)

    return figures


def generate_summary_html(
    summary_df: pd.DataFrame,
    figures: List[go.Figure],
    n_dseeds: int,
    n_mseeds: int,
    total_runs: int,
    output_file: Optional[str] = None
) -> str:
    """
    Generate summary HTML with metrics table and Plotly visualizations.

    Args:
        summary_df: Aggregated metrics dataframe
        figures: List of Plotly figures (one per metric)
        n_dseeds: Number of data subsets
        n_mseeds: Number of model seeds per subset
        total_runs: Total number of runs
        output_file: Output filename

    Returns:
        Path to generated HTML file
    """
    if output_file is None:
        output_file = "aggregate_hybrid_results.html"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if output_path.exists():
        output_path.unlink()

    # Get degrees of freedom (same for all generations)
    df_value = n_dseeds - 1

    # Convert summary_df to long format for better readability
    long_format_data = []
    for _, row in summary_df.iterrows():
        generation = row['generation']
        for col in summary_df.columns:
            if col == 'generation':
                continue
            # Extract metric name and stat type
            if col.endswith('_mean'):
                metric_name = col.replace('_mean', '')
                long_format_data.append({
                    'Generation': int(generation),
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Mean': row[f'{metric_name}_mean'],
                    'Std': row[f'{metric_name}_std'],
                    'SE': row[f'{metric_name}_se'],
                    'CI Lower': row[f'{metric_name}_ci_lower'],
                    'CI Upper': row[f'{metric_name}_ci_upper'],
                    'N Dseeds': int(row[f'{metric_name}_n_dseeds']),
                    'DF': int(row[f'{metric_name}_df']),
                    'Total Obs': int(row[f'{metric_name}_total_obs'])
                })

    long_df = pd.DataFrame(long_format_data)

    # Build summary table HTML from long format
    summary_table_html = long_df.to_html(
        index=False,
        float_format=lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, float) else (f"{int(x)}" if isinstance(x, (int, float)) and x == int(x) else str(x)),
        border=0,
    )

    # Get Plotly figures as HTML (only include plotlyjs once)
    plots_html = []
    for idx, fig in enumerate(figures):
        plot_div_id = f"metrics_plot_{idx}"
        plot_html = fig.to_html(
            include_plotlyjs="cdn" if idx == 0 else False,
            div_id=plot_div_id
        )
        plots_html.append(plot_html)

    # Build complete HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <title>Aggregate Hybrid Stratified Metrics Results</title>",
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
        "    .method { font-size: 13px; color: #666; margin-top: 20px; padding: 15px; background: #e8f4f8; border-left: 3px solid #17a2b8; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h1>Aggregate Hybrid Stratified Metrics Results</h1>",
        "    <div class='summary'>",
        f"      <p><span class='stat-label'>Data subsets (dseeds):</span> <span class='stat-value'>{n_dseeds}</span></p>",
        f"      <p><span class='stat-label'>Model seeds per subset:</span> <span class='stat-value'>{n_mseeds}</span></p>",
        f"      <p><span class='stat-label'>Total runs:</span> <span class='stat-value'>{total_runs}</span></p>",
        f"      <p><span class='stat-label'>Degrees of freedom:</span> <span class='stat-value'>{df_value}</span></p>",
        f"      <p><span class='stat-label'>Generations analyzed:</span> <span class='stat-value'>{len(summary_df)}</span></p>",
        "    </div>",
        "    <div class='method'>",
        "      <strong>Hybrid Stratified Aggregation Method:</strong><br>",
        "      <ul>",
        "        <li><strong>Step 1:</strong> Average model seeds (mseed) within each data subset (dseed)</li>",
        "        <li><strong>Step 2:</strong> Compute t-based confidence intervals on dseed averages</li>",
        "        <li><strong>Statistical approach:</strong> Properly handles hierarchical structure and small sample sizes</li>",
        f"        <li><strong>T-distribution:</strong> Used instead of normal approximation (df = {df_value})</li>",
        "      </ul>",
        "    </div>",
        "    <h2>Metrics Overview</h2>",
        summary_table_html,
        "    <h2>Visualizations</h2>",
    ]

    # Add each plot in its own container
    for plot_html in plots_html:
        html_parts.extend([
            "    <div class='plot-container'>",
            plot_html,
            "    </div>",
        ])

    # Add visualization note
    html_parts.extend([
        "    <div class='note'>",
        "      <strong>Visualization Details:</strong><br>",
        "      <ul>",
        "        <li><strong>Inner band (colored):</strong> 95% t-based confidence interval (CI) for the mean - \"Where the true population mean likely is\"</li>",
        "        <li><strong>Outer band (gray):</strong> 95% Prediction Interval (PI) - \"Where a NEW data subset (dseed) average would likely fall\"</li>",
        "        <li><strong>Mean line:</strong> Colored line showing mean value at each generation</li>",
        "        <li><strong>Legend:</strong> Each figure has its own legend with clickable items</li>",
        "      </ul>",
        "      <strong>Statistical Details:</strong><br>",
        "      <ul>",
        f"        <li><strong>Confidence Interval:</strong> Mean ± t<sub>crit</sub> × SE, where SE = SD / √{n_dseeds}</li>",
        f"        <li><strong>Prediction Interval:</strong> Mean ± t<sub>crit</sub> × SD × √(1 + 1/{n_dseeds})</li>",
        "        <li><strong>Interpretation:</strong> CI narrows as n increases (uncertainty about mean decreases), but PI remains wider (natural variation persists)</li>",
        "      </ul>",
        "    </div>",
        "  </div>",
        "</body>",
        "</html>",
    ])

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
        "aggregate_hybrid_results.html",
        "--summary-output",
        "-s",
        help="Output path for summary HTML"
    ),
    csv_output: Optional[str] = typer.Option(
        "aggregate_hybrid_results.csv",
        "--csv-output",
        "-c",
        help="Output path for aggregated metrics CSV"
    ),
    intermediate_output: Optional[str] = typer.Option(
        "dseed_averages.csv",
        "--intermediate-output",
        "-i",
        help="Output path for intermediate dseed averages CSV"
    ),
) -> None:
    """
    Aggregate metrics using hybrid stratified strategy.

    This tool properly handles hierarchical experimental data (dseed → mseed → generation)
    by first averaging across model seeds within each data subset, then computing
    t-based confidence intervals on the dseed averages.

    Examples:
        python aggregate_hybrid_stratified_metrics_cli.py "./mimic_iii_baseline_dseed*_synthcity_arf_mseed*/"
        python aggregate_hybrid_stratified_metrics_cli.py "./exp_dseed*_mseed*/" -s results.html -c metrics.csv -i dseed_avgs.csv
    """

    console.print(f"[cyan]Pattern:[/cyan] {pattern}")

    # Expand glob pattern
    folders = sorted(Path().glob(pattern))

    if not folders:
        console.print(f"[red]Error: No folders matched pattern: {pattern}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(folders)} folder(s)[/cyan]\n")

    # Parse folder names and organize by dseed/mseed
    data_structure = defaultdict(dict)  # {dseed: {mseed: dataframe}}
    metrics = ["factual_total", "ddr_novel_factual", "ks_complement", "tv_complement",
               "wasserstein_dist", "jsd_syndat", "mmd", "alpha_delta_precision_OC",
               "alpha_delta_coverage_OC", "alpha_authenticity_OC", "prdc_precision",
               "prdc_recall", "prdc_density", "prdc_coverage", "detection_gmm",
               "detection_xgb", "detection_mlp", "detection_linear"]

    with Progress() as progress:
        task = progress.add_task("[cyan]Loading and parsing folders...", total=len(folders))

        for folder in folders:
            progress.update(task, description=f"[cyan]Processing {folder.name}...")

            # Parse dseed and mseed from folder name
            dseed, mseed, error = parse_folder_name(folder.name)
            if error:
                progress.print(f"[red]✗[/red] {folder.name}: {error}")
                raise typer.Exit(1)

            # Validate folder structure
            is_valid, error = validate_folder(folder)
            if not is_valid:
                progress.print(f"[red]✗[/red] {folder.name}: {error}")
                raise typer.Exit(1)

            # Load CSV
            success, df, error = load_metrics_csv(folder)
            if not success:
                progress.print(f"[red]✗[/red] {folder.name}: {error}")
                raise typer.Exit(1)

            # Store in hierarchical structure
            data_structure[dseed][mseed] = df
            progress.print(f"[green]✓[/green] {folder.name} (dseed={dseed}, mseed={mseed}, {len(df)} rows)")
            progress.advance(task)

    console.print()

    # Validate hierarchical structure
    console.print("[cyan]Validating hierarchical structure...[/cyan]")
    is_valid, error = validate_hierarchical_structure(data_structure)
    if not is_valid:
        console.print(f"[red]✗[/red] Validation failed: {error}")
        raise typer.Exit(1)

    n_dseeds = len(data_structure)
    n_mseeds = len(next(iter(data_structure.values())))
    total_runs = n_dseeds * n_mseeds

    console.print(f"[green]✓[/green] Structure validated: {n_dseeds} dseeds × {n_mseeds} mseeds = {total_runs} runs")

    # Issue warnings for small sample sizes
    if n_dseeds < 3:
        console.print(f"[yellow]⚠[/yellow] Warning: Very small number of dseeds ({n_dseeds}). CI may be unreliable.")

    if n_mseeds < 3:
        console.print(f"[yellow]⚠[/yellow] Warning: Small number of mseeds ({n_mseeds}). May not fully capture model variance.")

    console.print()

    # Step 1: Compute dseed averages
    console.print("[cyan]Step 1: Computing dseed averages (averaging mseeds within each dseed)...[/cyan]")
    dseed_averages = compute_dseed_averages(data_structure, metrics)
    console.print(f"[green]✓[/green] Computed {len(dseed_averages)} dseed-level averages")

    # Step 2: Compute stratified statistics with t-based CI
    console.print("[cyan]Step 2: Computing t-based confidence intervals on dseed averages...[/cyan]")
    summary_df = compute_stratified_statistics(dseed_averages, metrics)
    console.print(f"[green]✓[/green] Computed statistics for {len(summary_df)} generations")

    # Create visualizations
    console.print("[cyan]Creating Plotly visualizations...[/cyan]")
    figures = create_plotly_visualization(summary_df, metrics, n_dseeds)
    console.print(f"[green]✓[/green] Created {len(figures)} visualizations")

    # Generate summary HTML
    console.print("[cyan]Generating summary HTML...[/cyan]")
    summary_path = generate_summary_html(
        summary_df, figures, n_dseeds, n_mseeds, total_runs, summary_output
    )
    console.print(f"[green]✓[/green] Summary HTML generated: {summary_path}")

    # Export aggregated metrics CSV
    console.print("[cyan]Exporting aggregated metrics CSV...[/cyan]")
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
    summary_df.to_csv(csv_path, index=False)
    console.print(f"[green]✓[/green] Aggregated metrics CSV exported: {csv_path}")

    # Export intermediate dseed averages CSV
    console.print("[cyan]Exporting intermediate dseed averages CSV...[/cyan]")
    intermediate_path = Path(intermediate_output)
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    if intermediate_path.exists():
        intermediate_path.unlink()
    dseed_averages.to_csv(intermediate_path, index=False)
    console.print(f"[green]✓[/green] Intermediate dseed averages CSV exported: {intermediate_path}")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [cyan]Data subsets (dseeds):[/cyan] {n_dseeds}")
    console.print(f"  [cyan]Model seeds per subset:[/cyan] {n_mseeds}")
    console.print(f"  [cyan]Total runs:[/cyan] {total_runs}")
    console.print(f"  [cyan]Degrees of freedom:[/cyan] {n_dseeds - 1}")
    console.print(f"  [cyan]Generations analyzed:[/cyan] {len(summary_df)}")
    console.print(f"  [cyan]Metrics analyzed:[/cyan] {', '.join(metrics)}")
    console.print(f"\n  [cyan]Outputs:[/cyan]")
    console.print(f"    - HTML report: {summary_path}")
    console.print(f"    - Aggregated metrics: {csv_path}")
    console.print(f"    - Dseed averages: {intermediate_path}")


if __name__ == "__main__":
    app()
