#!/usr/bin/env python
"""
SDMetrics Comprehensive Quality Report CLI Tool

Comprehensive synthesis quality analysis with pairwise correlation matrices.
Generates three correlation matrices:
1. Real data correlations (with pairwise deletion and confidence metrics)
2. Synthetic data correlations (complete data analysis)
3. Quality scores (SDMetrics similarity assessment)

Uses SDMetrics QualityReport for distribution similarity and correlation preservation analysis.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from sdv.metadata import SingleTableMetadata

from sdmetrics.reports.single_table import QualityReport
from sdpype.metadata import load_csv_with_metadata

console = Console()
app = typer.Typer(add_completion=False)


def get_score_color(score: float) -> str:
    """Get Rich color for score value."""
    if pd.isna(score):
        return "dim"
    elif score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    else:
        return "red"


def get_confidence_flag(n_samples: int) -> str:
    """Get confidence flag based on sample size."""
    if n_samples >= 1000:
        return '✓'
    elif n_samples >= 500:
        return '⚠'
    elif n_samples >= 50:
        return '✗'
    else:
        return '-'


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1

    if min_dim == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim)) if n > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Calculate correlation ratio (eta) for categorical-numerical association."""
    categories = categories.fillna('Missing')
    values = values.fillna(values.mean())

    categories = pd.Categorical(categories)
    grouped_mean = values.groupby(categories, observed=True).mean()
    overall_mean = values.mean()
    group_counts = values.groupby(categories, observed=True).size()

    ss_between = ((grouped_mean - overall_mean) ** 2 * group_counts).sum()
    ss_total = ((values - overall_mean) ** 2).sum()

    if ss_total == 0:
        return 0.0

    return np.sqrt(ss_between / ss_total)


def compute_correlation_matrix(
    data: pd.DataFrame,
    metadata: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix using pairwise deletion.

    Args:
        data: DataFrame to analyze
        metadata: SDV metadata dictionary

    Returns:
        Tuple of (correlation_matrix, sample_size_matrix)
    """
    columns = list(data.columns)
    corr_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    sample_matrix = pd.DataFrame(index=columns, columns=columns, dtype=int)

    sdtypes = {col: metadata['columns'][col]['sdtype'] for col in columns}

    for i, col1 in enumerate(columns):
        for col2 in columns[i:]:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
                sample_matrix.loc[col1, col2] = len(data)
                continue

            sdtype1 = sdtypes[col1].lower()
            sdtype2 = sdtypes[col2].lower()

            # Skip datetime columns
            if sdtype1 == 'datetime' or sdtype2 == 'datetime':
                corr_matrix.loc[col1, col2] = np.nan
                corr_matrix.loc[col2, col1] = np.nan
                sample_matrix.loc[col1, col2] = 0
                sample_matrix.loc[col2, col1] = 0
                continue

            try:
                # Pairwise deletion: only rows where BOTH columns are not null
                valid_mask = data[[col1, col2]].notna().all(axis=1)
                n_valid = valid_mask.sum()

                if n_valid < 2:  # Need at least 2 samples
                    corr_matrix.loc[col1, col2] = np.nan
                    corr_matrix.loc[col2, col1] = np.nan
                    sample_matrix.loc[col1, col2] = n_valid
                    sample_matrix.loc[col2, col1] = n_valid
                    continue

                data1 = data.loc[valid_mask, col1]
                data2 = data.loc[valid_mask, col2]

                # Both numerical - Pearson
                if sdtype1 == 'numerical' and sdtype2 == 'numerical':
                    corr, _ = pearsonr(data1, data2)
                    corr_matrix.loc[col1, col2] = abs(corr)
                    corr_matrix.loc[col2, col1] = abs(corr)

                # Both categorical - Cramér's V
                elif sdtype1 in ['categorical', 'boolean'] and sdtype2 in ['categorical', 'boolean']:
                    v = cramers_v(data1, data2)
                    corr_matrix.loc[col1, col2] = v
                    corr_matrix.loc[col2, col1] = v

                # Mixed - Correlation ratio
                else:
                    if sdtype1 in ['categorical', 'boolean']:
                        eta = correlation_ratio(data1, data2)
                    else:
                        eta = correlation_ratio(data2, data1)

                    corr_matrix.loc[col1, col2] = eta
                    corr_matrix.loc[col2, col1] = eta

                sample_matrix.loc[col1, col2] = n_valid
                sample_matrix.loc[col2, col1] = n_valid

            except Exception as e:
                console.print(f"[yellow]⚠ Error computing {col1}-{col2}: {str(e)}[/yellow]")
                corr_matrix.loc[col1, col2] = np.nan
                corr_matrix.loc[col2, col1] = np.nan
                sample_matrix.loc[col1, col2] = 0
                sample_matrix.loc[col2, col1] = 0

    return corr_matrix, sample_matrix


def build_quality_pair_matrix(report: QualityReport) -> pd.DataFrame:
    """Extract pairwise quality scores from QualityReport."""
    details = report.get_details('Column Pair Trends')

    all_cols = sorted(set(details['Column 1'].unique()) | set(details['Column 2'].unique()))
    matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

    for _, row in details.iterrows():
        col1, col2 = row['Column 1'], row['Column 2']
        score = row['Score']
        matrix.loc[col1, col2] = score
        matrix.loc[col2, col1] = score

    for col in all_cols:
        matrix.loc[col, col] = 1.0

    return matrix


def display_data_quality_report(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
):
    """Display data quality and null value analysis."""
    console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
    console.print("[bold cyan]Data Quality Report[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

    # Null value analysis
    real_nulls = real_data.isnull().sum()
    synth_nulls = synthetic_data.isnull().sum()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Column", style="dim white")
    table.add_column("Real Nulls", justify="right")
    table.add_column("% Missing", justify="right")
    table.add_column("Synth Nulls", justify="right")
    table.add_column("Note", style="dim")

    for col in real_data.columns:
        real_null_count = real_nulls[col]
        real_pct = (real_null_count / len(real_data)) * 100
        synth_null_count = synth_nulls[col]

        note = ""
        if real_pct > 50:
            note = "⚠ Highly sparse"
        elif real_pct > 20:
            note = "⚠ Sparse"

        table.add_row(
            col,
            str(real_null_count),
            f"{real_pct:.1f}%",
            str(synth_null_count),
            note,
        )

    console.print(table)

    # Sample size interpretation
    console.print("\n[bold cyan]Correlation Confidence Levels:[/bold cyan]")
    console.print("  ✓ = ≥1000 samples (high confidence)")
    console.print("  ⚠ = 500-1000 samples (moderate confidence)")
    console.print("  ✗ = 50-499 samples (low confidence)")
    console.print("  - = <50 samples (insufficient data)")


def display_column_shapes_details(report: QualityReport):
    """Display per-column shape scores breakdown."""
    console.print(f"\n[bold cyan]{'═' * 100}[/bold cyan]")
    console.print("[bold cyan]Column Shapes Details (Distribution Similarity)[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 100}[/bold cyan]\n")

    details = report.get_details('Column Shapes')

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Column", style="dim white", width=25)
    table.add_column("Metric", style="magenta", width=15)
    table.add_column("Score", justify="center", width=12)
    table.add_column("Interpretation", style="dim", width=60)

    for _, row in details.iterrows():
        col_name = row['Column']
        metric = row['Metric']
        score = row['Score']

        if pd.isna(score):
            interpretation = "Insufficient data"
            color = "dim"
        elif score >= 0.9:
            interpretation = "Excellent - distributions very similar"
            color = "green"
        elif score >= 0.8:
            interpretation = "Good - distributions similar"
            color = "green"
        elif score >= 0.7:
            interpretation = "Acceptable - distributions somewhat similar"
            color = "yellow"
        elif score >= 0.5:
            interpretation = "Poor - distributions quite different"
            color = "yellow"
        else:
            interpretation = "Very poor - distributions very different"
            color = "red"

        table.add_row(
            col_name,
            metric,
            f"[{color}]{score:.3f}[/{color}]",
            interpretation,
        )

    console.print(table)


def display_property_scores(report: QualityReport):
    """Display overall quality property scores."""
    console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
    console.print("[bold cyan]Quality Report Scores (Real vs Synthetic)[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim white")
    table.add_column("Score", justify="center")

    for prop_name in ["Column Shapes", "Column Pair Trends"]:
        score = report._properties[prop_name]._compute_average()
        color = get_score_color(score)
        table.add_row(
            prop_name,
            f"[{color}]{score:.2%}[/{color}]",
        )

    overall = report.get_score()
    overall_color = get_score_color(overall)
    table.add_row(
        "[bold]Overall Quality Score[/bold]",
        f"[bold {overall_color}]{overall:.2%}[/bold {overall_color}]",
    )

    console.print(table)


def display_matrix_with_samples(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    title: str,
    max_cols: int = 10,
):
    """Display correlation matrix with sample sizes and confidence flags."""
    console.print(f"\n[bold cyan]{'═' * 100}[/bold cyan]")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 100}[/bold cyan]\n")

    cols = sorted(matrix.columns)
    display_cols = cols[:max_cols]

    table = Table(show_header=True, header_style="dim cyan", width=160)
    table.add_column("", style="bold dim white", width=18)

    for col in display_cols:
        table.add_column(col[:11], width=13, justify="center")

    for row_col in display_cols:
        row_data = [Text(row_col[:11], style="bold dim white")]

        for col_col in display_cols:
            corr = matrix.loc[row_col, col_col]
            n_samples = samples.loc[row_col, col_col]

            if pd.isna(corr):
                row_data.append(Text("    -     ", style="dim"))
            else:
                color = get_score_color(corr)
                confidence = get_confidence_flag(n_samples)
                # Format: correlation (confidence_flag)
                row_data.append(
                    Text(f"{corr:.3f}({confidence})", style=color, justify="center")
                )

        table.add_row(*row_data)

    console.print(table)

    if len(cols) > max_cols:
        console.print(f"[dim]... and {len(cols) - max_cols} more columns[/dim]")


def display_lowest_scoring_pairs(report: QualityReport, n_pairs: int = 10):
    """Display the lowest scoring column pairs."""
    console.print(f"\n[bold cyan]{'═' * 100}[/bold cyan]")
    console.print(f"[bold cyan]Lowest Scoring Pairs (Bottom {n_pairs})[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 100}[/bold cyan]\n")

    details = report.get_details('Column Pair Trends')

    # Get lowest scoring pairs
    lowest = details.nsmallest(n_pairs, 'Score')[['Column 1', 'Column 2', 'Metric', 'Score']]

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Column 1", style="dim white", width=20)
    table.add_column("Column 2", style="dim white", width=25)
    table.add_column("Metric", style="magenta", width=25)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Status", width=8)

    for _, row in lowest.iterrows():
        score = row['Score']

        if pd.isna(score):
            status = "-"
            color = "dim"
        elif score < 0.3:
            status = "✗✗"
            color = "red"
        elif score < 0.5:
            status = "✗"
            color = "red"
        elif score < 0.7:
            status = "⚠"
            color = "yellow"
        else:
            status = "⚠"
            color = "yellow"

        table.add_row(
            row['Column 1'],
            row['Column 2'],
            row['Metric'],
            f"[{color}]{score:.3f}[/{color}]",
            Text(status, style=f"bold {color}"),
        )

    console.print(table)


def get_diagnostic_flag(real_corr: float, synth_corr: float, quality_score: float) -> str:
    """
    Generate diagnostic flag based on three metrics.

    Returns: '✓' (good), '⚠' (warning), '✗' (bad), or '-' (insufficient data)
    """
    if pd.isna(real_corr) or pd.isna(synth_corr) or pd.isna(quality_score):
        return '-'

    # Strong correlation in real, well preserved
    if real_corr >= 0.70 and quality_score >= 0.80:
        return '✓'

    # Weak correlation in real, correctly reproduced as weak in synthetic
    if real_corr < 0.30 and synth_corr < 0.30:
        return '✓'

    # Moderate correlation, decent preservation
    if real_corr >= 0.50 and quality_score >= 0.65:
        return '✓'

    # Strong relationship in real but lost in synthetic
    if real_corr >= 0.70 and quality_score < 0.60:
        return '✗'

    # Warning: moderate issues
    return '⚠'


def get_diagnosis(real_corr: float, synth_corr: float, quality_score: float) -> str:
    """Generate diagnostic message for a pair."""
    if pd.isna(real_corr) or pd.isna(synth_corr) or pd.isna(quality_score):
        return "Insufficient data"

    # Excellent
    if real_corr >= 0.80 and quality_score >= 0.85:
        return "Strong correlation well preserved"

    # Good
    if real_corr >= 0.70 and quality_score >= 0.75:
        return "Strong correlation preserved"

    # Correct noise
    if real_corr < 0.25 and synth_corr < 0.25:
        return "Weak correlation correctly reproduced"

    # Learning failure
    if real_corr >= 0.70 and quality_score < 0.50:
        return "Model failed to capture strong trend"

    # Drift
    if synth_corr >= 0.70 and real_corr < 0.40 and quality_score < 0.50:
        return "Synthetic created spurious correlation"

    # Moderate
    if real_corr >= 0.50 and quality_score >= 0.65:
        return "Moderate correlation preserved"

    # Weak relationship reproduced
    if real_corr < 0.30:
        return "Weak correlation correctly reproduced"

    return "Mixed quality"


def display_diagnostic_analysis(
    real_matrix: pd.DataFrame,
    real_samples: pd.DataFrame,
    synth_matrix: pd.DataFrame,
    quality_matrix: pd.DataFrame,
):
    """Display diagnostic comparison of three matrices."""
    console.print(f"\n[bold cyan]{'═' * 140}[/bold cyan]")
    console.print("[bold cyan]Diagnostic Analysis: Correlation Preservation[/bold cyan]")
    console.print(f"[bold cyan]{'═' * 140}[/bold cyan]\n")

    cols = sorted(real_matrix.columns)

    table = Table(show_header=True, header_style="bold cyan", width=160)
    table.add_column("Column Pair", style="dim white", width=28)
    table.add_column("Real Corr", justify="center", width=12)
    table.add_column("n_samples", justify="center", width=10)
    table.add_column("Synth Corr", justify="center", width=12)
    table.add_column("Quality", justify="center", width=10)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Diagnosis", style="dim", width=70)

    for i, col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            real_corr = real_matrix.loc[col1, col2]
            n_samp = real_samples.loc[col1, col2]
            synth_corr = synth_matrix.loc[col1, col2]
            quality = quality_matrix.loc[col1, col2]

            if pd.isna(real_corr) and pd.isna(synth_corr) and pd.isna(quality):
                continue

            confidence = get_confidence_flag(n_samp)
            flag = get_diagnostic_flag(real_corr, synth_corr, quality)
            flag_color = "green" if flag == "✓" else "yellow" if flag == "⚠" else "red" if flag == "✗" else "dim"

            real_str = f"{real_corr:.3f}" if not pd.isna(real_corr) else "-"
            synth_str = f"{synth_corr:.3f}" if not pd.isna(synth_corr) else "-"
            quality_str = f"{quality:.3f}" if not pd.isna(quality) else "-"

            diagnosis = ""
            if pd.isna(real_corr) and confidence == '-':
                diagnosis = "Insufficient data"
            elif real_corr >= 0.75 and quality >= 0.75:
                diagnosis = "Strong relationship preserved"
            elif real_corr < 0.25 and synth_corr < 0.25:
                diagnosis = "Weak relationship reproduced"
            elif real_corr >= 0.70 and quality < 0.50:
                diagnosis = "Strong trend lost in synthesis"
            elif confidence in ['✗', '-']:
                diagnosis = f"Low confidence ({n_samp} samples)"
            else:
                diagnosis = "Moderate preservation"

            table.add_row(
                f"{col1[:12]} × {col2[:12]}",
                real_str,
                f"{n_samp}" if n_samp > 0 else "-",
                synth_str,
                quality_str,
                Text(flag, style=f"bold {flag_color}"),
                diagnosis[:65] + ("..." if len(diagnosis) > 65 else ""),
            )

    console.print(table)


def generate_comprehensive_json(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata: dict,
    real_matrix: pd.DataFrame,
    real_samples: pd.DataFrame,
    synth_matrix: pd.DataFrame,
    quality_matrix: pd.DataFrame,
    report: QualityReport,
) -> dict:
    """Generate comprehensive JSON output with full diagnostics."""
    cols = sorted(real_matrix.columns)
    diagnostics = {}
    all_pairs = []

    for i, col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            pair_key = f"{col1}×{col2}"

            real_corr = float(real_matrix.loc[col1, col2]) if not pd.isna(real_matrix.loc[col1, col2]) else None
            n_samples = int(real_samples.loc[col1, col2])
            synth_corr = float(synth_matrix.loc[col1, col2]) if not pd.isna(synth_matrix.loc[col1, col2]) else None
            quality_score = float(quality_matrix.loc[col1, col2]) if not pd.isna(quality_matrix.loc[col1, col2]) else None

            if all(x is not None for x in [real_corr, synth_corr, quality_score]):
                flag = get_diagnostic_flag(real_corr, synth_corr, quality_score)
                diagnosis = get_diagnosis(real_corr, synth_corr, quality_score)
                confidence = get_confidence_flag(n_samples)

                pair_data = {
                    "column_1": col1,
                    "column_2": col2,
                    "real_correlation": real_corr,
                    "real_n_samples": n_samples,
                    "real_confidence": confidence,
                    "synthetic_correlation": synth_corr,
                    "quality_score": quality_score,
                    "diagnostic_flag": flag,
                    "diagnosis": diagnosis,
                }

                diagnostics[pair_key] = pair_data
                all_pairs.append(pair_data)

    # Sort for lowest and highest scoring pairs
    all_pairs_sorted = sorted(all_pairs, key=lambda x: x['quality_score'])
    lowest_pairs = all_pairs_sorted[:10]
    highest_pairs = all_pairs_sorted[-10:]
    highest_pairs.reverse()

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "real_data_rows": len(real_data),
            "synthetic_data_rows": len(synthetic_data),
            "columns": len(metadata['columns']),
            "total_pairs": len(all_pairs),
            "methodology": "Pairwise deletion - correlations computed using only non-null pairs for real data",
        },
        "null_values": {
            "real_data": real_data.isnull().sum().to_dict(),
            "synthetic_data": synthetic_data.isnull().sum().to_dict(),
        },
        "property_scores": {
            "column_shapes": float(report._properties["Column Shapes"]._compute_average()),
            "column_pair_trends": float(report._properties["Column Pair Trends"]._compute_average()),
            "overall": float(report.get_score()),
        },
        "matrices": {
            "real_correlations": real_matrix.to_dict(),
            "real_sample_sizes": real_samples.to_dict(),
            "synthetic_correlations": synth_matrix.to_dict(),
            "quality_scores": quality_matrix.to_dict(),
        },
        "diagnostics": {
            "all_pairs": diagnostics,
            "lowest_scoring_pairs": [
                {
                    "rank": i + 1,
                    "column_1": pair["column_1"],
                    "column_2": pair["column_2"],
                    "quality_score": pair["quality_score"],
                    "real_correlation": pair["real_correlation"],
                    "synthetic_correlation": pair["synthetic_correlation"],
                    "diagnostic_flag": pair["diagnostic_flag"],
                    "diagnosis": pair["diagnosis"],
                    "real_confidence": pair["real_confidence"],
                }
                for i, pair in enumerate(lowest_pairs)
            ],
            "highest_scoring_pairs": [
                {
                    "rank": i + 1,
                    "column_1": pair["column_1"],
                    "column_2": pair["column_2"],
                    "quality_score": pair["quality_score"],
                    "real_correlation": pair["real_correlation"],
                    "synthetic_correlation": pair["synthetic_correlation"],
                    "diagnostic_flag": pair["diagnostic_flag"],
                    "diagnosis": pair["diagnosis"],
                    "real_confidence": pair["real_confidence"],
                }
                for i, pair in enumerate(highest_pairs)
            ],
        },
        "column_shapes": report.get_details("Column Shapes").to_dict(orient='records'),
        "column_pair_details": report.get_details("Column Pair Trends").to_dict(orient='records'),
    }


@app.command()
def main(
    real: Path = typer.Option(
        ...,
        "--real",
        help="Path to real/original data CSV.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    synthetic: Path = typer.Option(
        ...,
        "--synthetic",
        help="Path to synthetic data CSV.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    metadata: Path = typer.Option(
        ...,
        "--metadata",
        help="Path to SDV metadata.json file (required for proper type enforcement).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save comprehensive JSON output (optional).",
    ),
    max_display_cols: int = typer.Option(
        10,
        "--max-display-cols",
        help="Maximum number of columns to display in matrices.",
    ),
):
    """
    📊 Comprehensive synthesis quality analysis using SDMetrics.

    Generates three correlation matrices with pairwise deletion:
    1. Real data correlations (with sample sizes and confidence)
    2. Synthetic data correlations (complete data, no nulls)
    3. Quality scores (how well synthesis preserved relationships)

    Uses SDMetrics QualityReport for distribution similarity assessment.

    Example:
        sdmetrics_quality_cli \\
          --real real.csv \\
          --synthetic synthetic.csv \\
          --metadata metadata.json \\
          --output results.json
    """
    console.print(Panel(
        "[bold cyan]📊 SDMetrics Comprehensive Quality Analysis[/bold cyan]",
        expand=False,
        border_style="cyan"
    ))

    # Load metadata
    console.print("\n[bold]Loading metadata...[/bold]")
    try:
        metadata_obj = SingleTableMetadata.load_from_json(str(metadata))
        metadata_dict = metadata_obj.to_dict()
        console.print(f"✅ Loaded metadata with {len(metadata_obj.columns)} columns\n")
    except Exception as e:
        console.print(f"❌ Failed to load metadata: {e}", style="red")
        raise typer.Exit(code=1)

    # Load datasets using metadata for type enforcement
    console.print("[bold]Loading datasets with metadata-enforced dtypes...[/bold]")
    try:
        real_data = load_csv_with_metadata(real, metadata)
        console.print(f"✅ Loaded real data: {real_data.shape[0]:,} rows × {real_data.shape[1]} columns")

        synthetic_data = load_csv_with_metadata(synthetic, metadata)
        console.print(f"✅ Loaded synthetic data: {synthetic_data.shape[0]:,} rows × {synthetic_data.shape[1]} columns\n")
    except Exception as e:
        console.print(f"❌ Failed to load data: {e}", style="red")
        raise typer.Exit(code=1)

    # Validate column compatibility
    metadata_columns = set(metadata_dict['columns'].keys())
    real_columns = set(real_data.columns)
    synthetic_columns = set(synthetic_data.columns)

    if not (metadata_columns == real_columns == synthetic_columns):
        console.print("❌ Column mismatch between metadata and datasets", style="red")
        console.print(f"Metadata columns: {len(metadata_columns)}")
        console.print(f"Real columns: {len(real_columns)}")
        console.print(f"Synthetic columns: {len(synthetic_columns)}")
        raise typer.Exit(code=1)

    if len(real_data) == 0 or len(synthetic_data) == 0:
        console.print("❌ Empty dataset detected", style="red")
        raise typer.Exit(code=1)

    console.print("✅ All inputs validated\n")

    # Compute correlation matrices and quality report
    console.print("[bold]Computing correlation matrices and quality metrics...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing real data correlations (pairwise deletion)...", total=None)
        real_matrix, real_samples = compute_correlation_matrix(real_data, metadata_dict)
        progress.update(task, completed=True)
        console.print("✅ Real data correlations computed")

        task = progress.add_task("Computing synthetic data correlations...", total=None)
        synth_matrix, synth_samples = compute_correlation_matrix(synthetic_data, metadata_dict)
        progress.update(task, completed=True)
        console.print("✅ Synthetic data correlations computed")

        task = progress.add_task("Running SDMetrics quality report...", total=None)
        report = QualityReport()
        report.generate(real_data, synthetic_data, metadata_dict, verbose=False)
        quality_matrix = build_quality_pair_matrix(report)
        progress.update(task, completed=True)
        console.print("✅ Quality report generated")

    console.print("\n✅ [green bold]All computations complete![/green bold]\n")

    # Display results
    display_data_quality_report(real_data, synthetic_data)
    display_column_shapes_details(report)
    display_property_scores(report)

    display_matrix_with_samples(
        real_matrix,
        real_samples,
        "Matrix 1: Real Data Correlations (Pairwise Deletion)",
        max_cols=max_display_cols
    )

    display_matrix_with_samples(
        synth_matrix,
        synth_samples,
        "Matrix 2: Synthetic Data Correlations",
        max_cols=max_display_cols
    )

    display_matrix_with_samples(
        quality_matrix,
        synth_samples,
        "Matrix 3: Quality Scores (Real vs Synthetic)",
        max_cols=max_display_cols
    )

    display_diagnostic_analysis(real_matrix, real_samples, synth_matrix, quality_matrix)
    display_lowest_scoring_pairs(report, n_pairs=10)

    # Save JSON if requested
    if output:
        console.print(f"\n[bold]Saving results to {output.name}...[/bold]")
        json_data = generate_comprehensive_json(
            real_data,
            synthetic_data,
            metadata_dict,
            real_matrix,
            real_samples,
            synth_matrix,
            quality_matrix,
            report,
        )

        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            console.print(f"✅ Results saved to: {output}", style="green")
        except Exception as e:
            console.print(f"❌ Error saving output: {str(e)}", style="red")
            raise typer.Exit(code=1)

    # Interpretation guide
    guide_panel = Panel.fit(
        """📊 SDMetrics Quality Analysis Guide:

Quality Scores:
• Score ≥ 0.8: Excellent - distributions/correlations well preserved
• Score ≥ 0.6: Good - reasonable preservation
• Score < 0.6: Poor - significant differences detected

Correlation Matrices:
• Matrix 1 (Real): Uses pairwise deletion to handle missing values
• Matrix 2 (Synthetic): Complete correlation analysis
• Matrix 3 (Quality): SDMetrics similarity scores (0-1 scale)

Confidence Flags (based on sample size):
• ✓ = High confidence (≥1000 valid pairs)
• ⚠ = Moderate confidence (500-999 pairs)
• ✗ = Low confidence (50-499 pairs)
• - = Insufficient data (<50 pairs)

Diagnostic Analysis:
• ✓ = Good preservation of correlations
• ⚠ = Mixed quality or moderate issues
• ✗ = Poor preservation or learning failure

Implementation:
• Metadata is the single source of truth for column types
• Pairwise deletion: each correlation uses only rows where both columns are non-null
• Supports Pearson (numerical-numerical), Cramér's V (categorical-categorical),
  and correlation ratio (mixed types)""",
        title="📖 Interpretation Guide",
        border_style="blue"
    )
    console.print(guide_panel)

    console.print("\n🎉 [bold green]SDMetrics quality analysis complete![/bold green]\n")


if __name__ == "__main__":
    app()
