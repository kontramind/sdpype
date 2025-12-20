#!/usr/bin/env python3
"""
Post-Training Metrics Computation CLI

Compute metrics on existing experiment folders without re-running the DVC pipeline.
Useful for:
- Adding new metrics to old experiments
- Re-computing metrics with different configurations
- Filling missing metrics after pipeline interruptions
- Experimenting with metric parameters
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

import pandas as pd
import yaml
import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sdv.metadata import SingleTableMetadata
from joblib import Parallel, delayed

from sdpype.evaluation.statistical import evaluate_statistical_metrics, generate_statistical_report, evaluate_privacy_metrics, generate_privacy_report
from sdpype.evaluation.detection import evaluate_detection_metrics, generate_detection_report, ensure_json_serializable
from sdpype.evaluation.hallucination import evaluate_hallucination_metrics, generate_hallucination_report
from sdpype.metadata import load_csv_with_metadata
from sdpype.encoding import RDTDatasetEncoder
from sdpype.encoding import load_encoding_config

# Import display functions from pipeline scripts
import sys
import importlib.util

def import_display_functions():
    """Import display functions from pipeline scripts"""
    functions = {}

    # Import detection display
    try:
        spec = importlib.util.spec_from_file_location("detect_module", "sdpype/detect.py")
        detect_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detect_module)
        functions['detection'] = detect_module._display_detection_tables
    except Exception as e:
        console.print(f"[dim]Could not import detection display: {e}[/dim]")
        functions['detection'] = None

    # Import hallucination display
    try:
        spec = importlib.util.spec_from_file_location("halluc_module", "sdpype/hallucination_evaluation.py")
        halluc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(halluc_module)
        functions['hallucination'] = halluc_module._display_hallucination_tables
    except Exception as e:
        console.print(f"[dim]Could not import hallucination display: {e}[/dim]")
        functions['hallucination'] = None

    return functions

# Load display functions once
DISPLAY_FUNCTIONS = import_display_functions()

console = Console()

def display_statistical_metrics(results: Dict[str, Any]):
    """
    Display statistical metrics results in terminal.
    Only displays metrics that were actually configured and computed.
    """
    console.print()
    metrics = results.get("metrics", {})

    # TableStructure results table
    if "table_structure" in metrics and metrics["table_structure"]["status"] == "success":
        ts_result = metrics["table_structure"]

        # Get parameters info for display
        params_info = ts_result["parameters"]
        params_display = str(params_info) if params_info else "no parameters"

        # Create TableStructure results table
        ts_table = Table(title=f"✅ TableStructure Results ({params_display})", show_header=True, header_style="bold blue")
        ts_table.add_column("Metric", style="cyan", no_wrap=True)
        ts_table.add_column("Value", style="bright_green", justify="right")

        ts_table.add_row("Overall Score", f"{ts_result['score']:.3f}")

        # Add summary information if available (backward compatible)
        if 'summary' in ts_result:
            summary = ts_result['summary']
            ts_table.add_row("Matching Columns", str(summary['matching_columns']))
            ts_table.add_row("Dtype Mismatches", str(summary['dtype_mismatches']))
            ts_table.add_row("Missing in Synthetic", str(summary['missing_in_synthetic']))
            ts_table.add_row("Only in Synthetic", str(summary['only_in_synthetic']))

        console.print(ts_table)

        # Show full column-by-column comparison table if available
        if 'comparison_table' in ts_result and ts_result['comparison_table']:
            comparison_table = ts_result['comparison_table']

            # Create full comparison table
            full_comparison = Table(
                title="Column-by-Column Comparison",
                show_header=True,
                header_style="bold blue",
                show_lines=False
            )
            full_comparison.add_column("Column", style="cyan", no_wrap=False)
            full_comparison.add_column("Real dtype", style="green")
            full_comparison.add_column("Synthetic dtype", style="magenta")
            full_comparison.add_column("Status", style="white")

            # Status display mapping
            status_display = {
                "match": "✓ Match",
                "dtype_mismatch": "⚠ Dtype mismatch",
                "missing_in_synthetic": "✗ Missing in synth",
                "only_in_synthetic": "⚠ Only in synth"
            }

            # Status colors
            status_colors = {
                "match": "bright_green",
                "dtype_mismatch": "yellow",
                "missing_in_synthetic": "red",
                "only_in_synthetic": "yellow"
            }

            # Add all columns to the table
            for item in comparison_table:
                status_text = status_display.get(item['status'], item['status'])
                status_color = status_colors.get(item['status'], "white")

                full_comparison.add_row(
                    item['column'],
                    str(item['real_dtype']) if item['real_dtype'] else '-',
                    str(item['synthetic_dtype']) if item['synthetic_dtype'] else '-',
                    f"[{status_color}]{status_text}[/{status_color}]"
                )

            console.print("\n")
            console.print(full_comparison)

    # SemanticStructure results table
    if "semantic_structure" in metrics and metrics["semantic_structure"]["status"] == "success":
        ss_result = metrics["semantic_structure"]

        # Get parameters info for display
        params_info = ss_result["parameters"]
        params_display = str(params_info) if params_info else "no parameters"

        # Create SemanticStructure results table
        ss_table = Table(title=f"✅ SemanticStructure Results ({params_display})", show_header=True, header_style="bold blue")
        ss_table.add_column("Metric", style="cyan", no_wrap=True)
        ss_table.add_column("Value", style="bright_green", justify="right")

        ss_table.add_row("Overall Score", f"{ss_result['score']:.3f}")

        # Add summary information if available
        if 'summary' in ss_result:
            summary = ss_result['summary']
            ss_table.add_row("Matching Columns", str(summary['matching_columns']))
            ss_table.add_row("Sdtype Mismatches", str(summary['sdtype_mismatches']))
            ss_table.add_row("Missing in Synthetic", str(summary['missing_in_synthetic']))
            ss_table.add_row("Only in Synthetic", str(summary['only_in_synthetic']))

        console.print(ss_table)

        # Show full column-by-column comparison table if available
        if 'comparison_table' in ss_result and ss_result['comparison_table']:
            comparison_table = ss_result['comparison_table']

            # Create full comparison table
            full_comparison = Table(
                title="Column-by-Column Comparison (Semantic Types)",
                show_header=True,
                header_style="bold blue",
                show_lines=False
            )
            full_comparison.add_column("Column", style="cyan", no_wrap=False)
            full_comparison.add_column("Real sdtype", style="green")
            full_comparison.add_column("Synthetic sdtype", style="magenta")
            full_comparison.add_column("Status", style="white")

            # Status display mapping
            status_display = {
                "match": "✓ Match",
                "sdtype_mismatch": "⚠ Sdtype mismatch",
                "missing_in_synthetic": "✗ Missing in synth",
                "only_in_synthetic": "⚠ Only in synth"
            }

            # Status colors
            status_colors = {
                "match": "bright_green",
                "sdtype_mismatch": "yellow",
                "missing_in_synthetic": "red",
                "only_in_synthetic": "yellow"
            }

            # Add all columns to the table
            for item in comparison_table:
                status_text = status_display.get(item['status'], item['status'])
                status_color = status_colors.get(item['status'], "white")

                full_comparison.add_row(
                    item['column'],
                    str(item['real_sdtype']) if item['real_sdtype'] else '-',
                    str(item['synthetic_sdtype']) if item['synthetic_sdtype'] else '-',
                    f"[{status_color}]{status_text}[/{status_color}]"
                )

            console.print("\n")
            console.print(full_comparison)

    # NewRowSynthesis results table
    if "new_row_synthesis" in metrics and metrics["new_row_synthesis"]["status"] == "success":
        nrs_result = metrics["new_row_synthesis"]

        # Get parameters info for display
        params_info = nrs_result["parameters"]
        tolerance = params_info.get("numerical_match_tolerance", 0.01)
        sample_size = params_info.get("synthetic_sample_size", "all rows")
        params_display = f"tolerance={tolerance}, sample_size={sample_size}"

        # Create NewRowSynthesis results table
        nrs_table = Table(title=f"✅ NewRowSynthesis Results ({params_display})", show_header=True, header_style="bold blue")
        nrs_table.add_column("Metric", style="cyan", no_wrap=True)
        nrs_table.add_column("Value", style="bright_green", justify="right")

        nrs_table.add_row("New Row Score", f"{nrs_result['score']:.3f}")
        nrs_table.add_row("New Rows", f"{nrs_result['num_new_rows']:,}")
        nrs_table.add_row("Matched Rows", f"{nrs_result['num_matched_rows']:,}")

        console.print(nrs_table)

    # BoundaryAdherence results table
    if "boundary_adherence" in metrics and metrics["boundary_adherence"]["status"] == "success":
        ba_result = metrics["boundary_adherence"]

        # Get parameters info for display
        params_info = ba_result["parameters"]
        target_cols = params_info.get("target_columns", "all numerical/datetime")
        params_display = f"target_columns={target_cols}"

        # Create BoundaryAdherence results table
        ba_table = Table(title=f"✅ BoundaryAdherence Results ({params_display})", show_header=True, header_style="bold blue")
        ba_table.add_column("Column", style="cyan", no_wrap=True)
        ba_table.add_column("Boundary Score", style="bright_green", justify="right")
        ba_table.add_column("Status", style="yellow", justify="center")

        # Add aggregate score as first row
        if ba_result['aggregate_score'] is not None:
            ba_table.add_row("AGGREGATE", f"{ba_result['aggregate_score']:.3f}", "✓")
        else:
            ba_table.add_row("AGGREGATE", "n/a", "ℹ️")

        ba_table.add_section()

        # Get all columns from column_scores and compatible_columns
        column_scores = ba_result['column_scores']
        compatible_columns = ba_result.get('compatible_columns', [])
        all_columns = sorted(set(list(column_scores.keys()) + list(compatible_columns)))

        for col in all_columns:
            if col in column_scores:
                ba_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
            elif col in compatible_columns:
                ba_table.add_row(col, "error", "⚠️")
            else:
                ba_table.add_row(col, "n/a", "—")

        # Add message at the bottom if no compatible columns found
        if ba_result.get("message"):
            ba_table.add_section()
            ba_table.add_row("INFO", ba_result["message"], "ℹ️")

        console.print(ba_table)

    # CategoryAdherence results table
    if "category_adherence" in metrics and metrics["category_adherence"]["status"] == "success":
        ca_result = metrics["category_adherence"]

        # Get parameters info for display
        params_info = ca_result["parameters"]
        target_cols = params_info.get("target_columns", None)
        params_display = f"target_columns={target_cols}"

        if ca_result.get("message"):
            # Handle case where no compatible columns found
            console.print(f"⚠️  CategoryAdherence Results ({params_display})", style="bold yellow")
            console.print(f"   Status: {ca_result['message']}", style="yellow")
        else:
            # Create CategoryAdherence results table
            ca_table = Table(title=f"✅ CategoryAdherence Results ({params_display})", show_header=True, header_style="bold blue")
            ca_table.add_column("Column", style="cyan", no_wrap=True)
            ca_table.add_column("Category Score", style="bright_green", justify="right")
            ca_table.add_column("Status", style="yellow", justify="center")

            # Add aggregate score first
            ca_table.add_row("AGGREGATE", f"{ca_result['aggregate_score']:.3f}", "✓")

            ca_table.add_section()

            # Get all columns from column_scores and compatible_columns
            column_scores = ca_result['column_scores']
            compatible_columns = ca_result.get('compatible_columns', [])
            all_columns = sorted(set(list(column_scores.keys()) + list(compatible_columns)))

            for col in all_columns:
                if col in column_scores:
                    ca_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
                elif col in compatible_columns:
                    ca_table.add_row(col, "error", "⚠️")
                else:
                    ca_table.add_row(col, "n/a", "—")

            # Add message at the bottom if no compatible columns found
            if ca_result.get("message"):
                ca_table.add_section()
                ca_table.add_row("INFO", ca_result["message"], "ℹ️")

            console.print(ca_table)

    # Alpha Precision
    if "alpha_precision" in metrics and metrics["alpha_precision"]["status"] == "success":
        scores = metrics["alpha_precision"]["scores"]
        params_info = metrics["alpha_precision"]["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        table = Table(title=f"✅ Alpha Precision Results ({params_display})", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("OC Variant", style="green", justify="right")
        table.add_column("Naive Variant", style="yellow", justify="right")

        table.add_row("Delta Precision Alpha", f"{scores['delta_precision_alpha_OC']:.3f}", f"{scores['delta_precision_alpha_naive']:.3f}")
        table.add_row("Delta Coverage Beta", f"{scores['delta_coverage_beta_OC']:.3f}", f"{scores['delta_coverage_beta_naive']:.3f}")
        table.add_row("Authenticity", f"{scores['authenticity_OC']:.3f}", f"{scores['authenticity_naive']:.3f}")

        console.print(table)

    # PRDC Score
    if "prdc_score" in metrics and metrics["prdc_score"]["status"] == "success":
        prdc_result = metrics["prdc_score"]
        params_info = prdc_result["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        prdc_table = Table(title=f"✅ PRDC Score Results ({params_display})", show_header=True, header_style="bold blue")
        prdc_table.add_column("Metric", style="cyan", no_wrap=True)
        prdc_table.add_column("Score", style="bright_green", justify="right")

        prdc_table.add_row("Precision", f"{prdc_result['precision']:.3f}")
        prdc_table.add_row("Recall", f"{prdc_result['recall']:.3f}")
        prdc_table.add_row("Density", f"{prdc_result['density']:.3f}")
        prdc_table.add_row("Coverage", f"{prdc_result['coverage']:.3f}")

        console.print(prdc_table)

    # Wasserstein Distance
    if "wasserstein_distance" in metrics and metrics["wasserstein_distance"]["status"] == "success":
        wd_result = metrics["wasserstein_distance"]
        params_info = wd_result["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        wd_table = Table(title=f"✅ Wasserstein Distance Results ({params_display})", show_header=True, header_style="bold blue")
        wd_table.add_column("Metric", style="cyan", no_wrap=True)
        wd_table.add_column("Score", style="bright_green", justify="right")
        wd_table.add_column("Interpretation", style="yellow")

        distance = wd_result['joint_distance']
        interpretation = "Identical" if distance < 0.01 else "Very Similar" if distance < 0.05 else "Similar" if distance < 0.1 else "Different"

        wd_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
        wd_table.add_row("", "", "Lower is better")

        console.print(wd_table)

    # Maximum Mean Discrepancy
    if "maximum_mean_discrepancy" in metrics and metrics["maximum_mean_discrepancy"]["status"] == "success":
        mmd_result = metrics["maximum_mean_discrepancy"]
        params_info = mmd_result["parameters"]
        kernel = mmd_result.get("kernel", "rbf")
        params_display = f"kernel={kernel}"

        mmd_table = Table(title=f"✅ Maximum Mean Discrepancy Results ({params_display})", show_header=True, header_style="bold blue")
        mmd_table.add_column("Metric", style="cyan", no_wrap=True)
        mmd_table.add_column("Score", style="bright_green", justify="right")
        mmd_table.add_column("Interpretation", style="yellow")

        distance = mmd_result['joint_distance']
        interpretation = "Identical" if distance < 0.001 else "Very Similar" if distance < 0.01 else "Similar" if distance < 0.1 else "Different"

        mmd_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
        mmd_table.add_row("", "", "Lower is better")

        console.print(mmd_table)

    # Jensen-Shannon Distance variants
    for js_variant in ["jensenshannon_synthcity", "jensenshannon_syndat", "jensenshannon_nannyml"]:
        if js_variant in metrics and metrics[js_variant]["status"] == "success":
            jsd_result = metrics[js_variant]
            params_info = jsd_result["parameters"]

            if js_variant == "jensenshannon_synthcity":
                normalize = jsd_result.get("normalize", True)
                n_histogram_bins = jsd_result.get("n_histogram_bins", 10)
                params_display = f"normalize={normalize}, n_histogram_bins={n_histogram_bins}"
                title_name = "Jensen-Shannon Distance (Synthcity)"
            elif js_variant == "jensenshannon_syndat":
                n_unique_threshold = jsd_result.get("n_unique_threshold", 10)
                params_display = f"n_unique_threshold={n_unique_threshold}"
                title_name = "Jensen-Shannon Distance (SYNDAT)"
            else:  # nannyml
                params_display = str(params_info) if params_info else "adaptive binning (Doane's formula)"
                title_name = "Jensen-Shannon Distance (NannyML)"

            jsd_table = Table(title=f"✅ {title_name} Results ({params_display})", show_header=True, header_style="bold blue")
            jsd_table.add_column("Metric", style="cyan", no_wrap=True)
            jsd_table.add_column("Score", style="bright_green", justify="right")
            jsd_table.add_column("Interpretation", style="yellow")

            distance = jsd_result['distance_score']
            interpretation = "Identical" if distance < 0.01 else "Very Similar" if distance < 0.05 else "Similar" if distance < 0.1 else "Different"

            jsd_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
            jsd_table.add_row("", "", "Lower is better")

            console.print(jsd_table)

    # KS Complement
    if "ks_complement" in metrics and metrics["ks_complement"]["status"] == "success":
        ks_result = metrics["ks_complement"]
        params_info = ks_result["parameters"]
        target_cols = params_info.get("target_columns", "all numerical/datetime")
        params_display = f"target_columns={target_cols}"

        ks_table = Table(title=f"✅ KSComplement Results ({params_display})", show_header=True, header_style="bold blue")
        ks_table.add_column("Column", style="cyan", no_wrap=True)
        ks_table.add_column("KS Score", style="bright_green", justify="right")
        ks_table.add_column("Status", style="yellow", justify="center")

        if ks_result['aggregate_score'] is not None:
            ks_table.add_row("AGGREGATE", f"{ks_result['aggregate_score']:.3f}", "✓")
        else:
            ks_table.add_row("AGGREGATE", "n/a", "ℹ️")

        ks_table.add_section()

        column_scores = ks_result['column_scores']
        for col in sorted(column_scores.keys()):
            score = column_scores[col]
            if score is not None:
                ks_table.add_row(col, f"{score:.3f}", "✓")
            else:
                ks_table.add_row(col, "error", "⚠️")

        console.print(ks_table)

    # TV Complement
    if "tv_complement" in metrics and metrics["tv_complement"]["status"] == "success":
        tv_result = metrics["tv_complement"]
        params_info = tv_result["parameters"]
        target_cols = params_info.get("target_columns", "all categorical/boolean")
        params_display = f"target_columns={target_cols}"

        tv_table = Table(title=f"✅ TVComplement Results ({params_display})", show_header=True, header_style="bold blue")
        tv_table.add_column("Column", style="cyan", no_wrap=True)
        tv_table.add_column("TV Score", style="bright_green", justify="right")
        tv_table.add_column("Status", style="yellow", justify="center")

        if tv_result['aggregate_score'] is not None:
            tv_table.add_row("AGGREGATE", f"{tv_result['aggregate_score']:.3f}", "✓")
        else:
            tv_table.add_row("AGGREGATE", "n/a", "ℹ️")

        tv_table.add_section()

        column_scores = tv_result['column_scores']
        for col in sorted(column_scores.keys()):
            score = column_scores[col]
            if score is not None:
                tv_table.add_row(col, f"{score:.3f}", "✓")
            else:
                tv_table.add_row(col, "error", "⚠️")

        console.print(tv_table)

    console.print("\n✅ Statistical metrics evaluation completed", style="bold green")


def display_privacy_metrics(results: Dict[str, Any]):
    """
    Display privacy metrics results in terminal.
    Only displays metrics that were actually configured and computed.
    """
    console.print()
    metrics = results.get("metrics", {})

    # DCR Baseline Protection
    if "dcr_baseline_protection" in metrics and metrics["dcr_baseline_protection"]["status"] == "success":
        dcr_result = metrics["dcr_baseline_protection"]
        params_info = dcr_result["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        dcr_table = Table(title=f"✅ DCR Baseline Protection Results ({params_display})", show_header=True, header_style="bold blue")
        dcr_table.add_column("Metric", style="cyan", no_wrap=True)
        dcr_table.add_column("Score", style="bright_green", justify="right")
        dcr_table.add_column("Interpretation", style="yellow")

        # Overall privacy score
        score = dcr_result['score']
        interpretation = "Excellent" if score > 0.8 else "Good" if score > 0.6 else "Moderate" if score > 0.4 else "Needs Improvement"
        dcr_table.add_row("Privacy Score", f"{score:.3f}", interpretation)

        dcr_table.add_section()

        # Breakdown: DCR values
        dcr_table.add_row("Median DCR (Synthetic)", f"{dcr_result['median_dcr_synthetic']:.6f}", "")
        dcr_table.add_row("Median DCR (Random)", f"{dcr_result['median_dcr_random']:.6f}", "")
        dcr_table.add_row("", "", "Higher is better")

        console.print(dcr_table)

    # K-Anonymization
    if "k_anonymization" in metrics and metrics["k_anonymization"]["status"] == "success":
        k_anon_result = metrics["k_anonymization"]
        params_info = k_anon_result["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        # K-Anonymity Values Table
        k_values_table = Table(
            title=f"✅ K-Anonymity Values ({params_display})",
            show_header=True,
            header_style="bold blue"
        )
        k_values_table.add_column("Dataset", style="cyan", no_wrap=True)
        k_values_table.add_column("k-anonymity", justify="right", style="bright_green")
        k_values_table.add_column("Interpretation", style="yellow")

        k_values = k_anon_result.get("k_values", {})
        for dataset_name in ["population", "reference", "training", "synthetic"]:
            if dataset_name in k_values:
                k_data = k_values[dataset_name]
                k_val = k_data["k"]
                interp = k_data["interpretation"]
                k_values_table.add_row(dataset_name.capitalize(), str(k_val), interp)

        console.print(k_values_table)

        # K-Anonymity Ratios Table (if available)
        k_ratios = k_anon_result.get("k_ratios", {})
        if k_ratios:
            k_ratios_table = Table(
                title="📊 K-Anonymity Ratios",
                show_header=True,
                header_style="bold blue"
            )
            k_ratios_table.add_column("Comparison", style="cyan")
            k_ratios_table.add_column("Ratio", justify="right", style="bright_green")
            k_ratios_table.add_column("Interpretation", style="yellow")

            for label, ratio_data in k_ratios.items():
                ratio_val = ratio_data["ratio"]
                interp = ratio_data["interpretation"]
                k_ratios_table.add_row(label, f"{ratio_val:.4f}", interp)

            console.print(k_ratios_table)

        # Display QI columns info
        qi_cols = k_anon_result.get("qi_columns", [])
        cat_cols = k_anon_result.get("categorical_columns", [])
        datasets_eval = k_anon_result.get("datasets_evaluated", [])

        info_lines = [
            f"QI Columns: {', '.join(qi_cols)}",
            f"Datasets: {', '.join(datasets_eval)}"
        ]
        if cat_cols:
            info_lines.append(f"Categorical (Encoded): {', '.join(cat_cols)}")

        console.print("\n" + " | ".join(info_lines), style="dim")

    console.print("\n✅ Privacy metrics evaluation completed", style="bold green")


app = typer.Typer(
    name="compute_metrics_cli",
    help="Compute metrics on experiment folders post-training",
    add_completion=True,
    rich_markup_mode="rich"
)


class MetricType(str, Enum):
    """Available metric types"""
    statistical = "statistical"
    detection = "detection"
    hallucination = "hallucination"
    privacy = "privacy"
    all = "all"


def parse_model_id(model_id: str) -> Dict[str, str]:
    """
    Parse model ID from filename.

    Format: library_modeltype_refhash_roothash_trnhash_gen_N_cfghash_seed
    Example: sdv_gaussiancopula_906d6c18_0f363a5b_0f363a5b_gen_10_mimic_iii_baseline_6cb21f5b_24157817

    Args:
        model_id: Model identifier string

    Returns:
        Dictionary with parsed components
    """
    # Updated pattern to handle experiment name with underscores
    pattern = r'^([^_]+)_([^_]+)_([a-f0-9]{8})_([a-f0-9]{8})_([a-f0-9]{8})_gen_(\d+)_(.+)_([a-f0-9]{8})_(\d+)$'
    match = re.match(pattern, model_id)

    if not match:
        raise ValueError(f"Could not parse model_id: {model_id}")

    return {
        'library': match.group(1),
        'model_type': match.group(2),
        'ref_hash': match.group(3),
        'root_hash': match.group(4),
        'trn_hash': match.group(5),
        'generation': int(match.group(6)),
        'experiment_name': match.group(7),
        'cfg_hash': match.group(8),
        'seed': int(match.group(9))
    }


def discover_generation_files(folder: Path, generation: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Discover all generation files in an experiment folder.

    Args:
        folder: Experiment folder path
        generation: If specified, only return this generation

    Returns:
        List of dictionaries with generation information
    """
    generations = []

    # Look for synthetic data files in data/synthetic/
    synthetic_dir = folder / "data" / "synthetic"
    if not synthetic_dir.exists():
        console.print(f"[yellow]Warning: No data/synthetic/ directory found in {folder}[/yellow]")
        return generations

    # Find all synthetic data files (decoded)
    for synthetic_file in synthetic_dir.glob("synthetic_data_*_decoded.csv"):
        filename = synthetic_file.stem
        # Extract model_id from filename: synthetic_data_<MODEL_ID>_decoded
        match = re.match(r'synthetic_data_(.+)_decoded', filename)
        if not match:
            continue

        model_id = match.group(1)

        try:
            parsed = parse_model_id(model_id)

            # Filter by generation if specified
            if generation is not None and parsed['generation'] != generation:
                continue

            generations.append({
                'generation': parsed['generation'],
                'model_id': model_id,
                'parsed': parsed,
                'folder': folder
            })
        except ValueError as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")
            continue

    return sorted(generations, key=lambda x: x['generation'])


def find_metadata_file(folder: Path, config: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """
    Find metadata.json file by checking common locations.

    Priority order:
    1. From config['data']['metadata_file'] (if config provided)
    2. Auto-discovery in common locations

    Args:
        folder: Experiment folder path
        config: Optional configuration dictionary (from params.yaml)

    Returns:
        Path to metadata.json or None
    """
    # 1. Try to get from config first
    if config and 'data' in config and config['data'].get('metadata_file'):
        metadata_path = Path(config['data']['metadata_file'])

        # If path is relative, try resolving it relative to the experiment folder
        if not metadata_path.is_absolute() and not metadata_path.exists():
            relative_to_folder = folder / metadata_path
            if relative_to_folder.exists():
                console.print(f"[dim]Using metadata from config (relative to folder): {relative_to_folder}[/dim]")
                return relative_to_folder

        # Check if absolute path or relative to CWD exists
        if metadata_path.exists():
            console.print(f"[dim]Using metadata from config: {metadata_path}[/dim]")
            return metadata_path
        else:
            console.print(f"[yellow]Warning: Metadata file from config not found: {metadata_path}[/yellow]")

    # 2. Fall back to auto-discovery
    candidates = [
        folder / "metadata.json",
        folder / "data" / "metadata.json",
        folder / ".." / "downloads" / "metadata.json",  # Common location for downloads
        Path("data") / "metadata.json",  # Look in repo data/
        Path("metadata.json")  # Look in current directory
    ]

    for candidate in candidates:
        if candidate.exists():
            console.print(f"[dim]Auto-discovered metadata: {candidate}[/dim]")
            return candidate

    return None


def load_generation_data(
    folder: Path,
    model_id: str,
    metadata_path: Path,
    population_file: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    use_original_files: bool = False,
    load_encoded: bool = False
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
           Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load all necessary data files for a generation (decoded and optionally encoded).

    Args:
        folder: Experiment folder path
        model_id: Model identifier
        metadata_path: Path to metadata file
        population_file: Optional path to population data file (decoded)
        config: Optional config dict to get original training/reference files
        use_original_files: If True, use original files from config (for hallucination metrics)
                           If False, use decoded files from experiments folder (for statistical/detection)
        load_encoded: If True, also load pre-encoded files from encode_evaluation stage

    Returns:
        If load_encoded=False: (reference_decoded, synthetic_decoded, training_decoded, population_decoded)
        If load_encoded=True: (reference_decoded, synthetic_decoded, training_decoded, population_decoded,
                               reference_encoded, synthetic_encoded)
    """
    data_root = folder / "data"

    # For hallucination metrics: use original files (matches DVC hallucination_evaluation)
    # For statistical/detection: use decoded files from encode_evaluation stage (matches DVC)
    reference_file = None
    training_file = None

    if use_original_files and config and 'data' in config:
        # Try to get original files from config
        ref_from_config = config['data'].get('reference_file')
        trn_from_config = config['data'].get('training_file')

        if ref_from_config:
            reference_file = Path(ref_from_config)
            if not reference_file.is_absolute() and not reference_file.exists():
                # Try relative to folder
                reference_file = folder / ref_from_config
            if reference_file.exists():
                console.print(f"[dim]Using original reference file from config: {reference_file}[/dim]")
            else:
                console.print(f"[yellow]Warning: Reference file from config not found: {reference_file}[/yellow]")
                reference_file = None

        if trn_from_config:
            training_file = Path(trn_from_config)
            if not training_file.is_absolute() and not training_file.exists():
                # Try relative to folder
                training_file = folder / trn_from_config
            if training_file.exists():
                console.print(f"[dim]Using original training file from config: {training_file}[/dim]")
            else:
                console.print(f"[yellow]Warning: Training file from config not found: {training_file}[/yellow]")
                training_file = None

    # Fallback to decoded files if original files not available
    if not reference_file or not reference_file.exists():
        reference_file = data_root / "decoded" / f"reference_{model_id}.csv"
        console.print(f"[dim]Using decoded reference file: {reference_file}[/dim]")

    if not training_file or not training_file.exists():
        training_file = data_root / "decoded" / f"training_{model_id}.csv"
        console.print(f"[dim]Using decoded training file: {training_file}[/dim]")

    # Synthetic always uses decoded version
    synthetic_decoded = data_root / "synthetic" / f"synthetic_data_{model_id}_decoded.csv"

    # Load data files
    ref_dec = load_csv_with_metadata(reference_file, metadata_path) if reference_file.exists() else None
    syn_dec = load_csv_with_metadata(synthetic_decoded, metadata_path) if synthetic_decoded.exists() else None
    trn_dec = load_csv_with_metadata(training_file, metadata_path) if training_file.exists() else None

    # Load population data (decoded) from provided path or try to find it
    pop_dec = None
    if population_file and population_file.exists():
        console.print(f"[dim]Loading population data: {population_file}[/dim]")
        pop_dec = load_csv_with_metadata(population_file, metadata_path, low_memory=False)
    elif population_file:
        console.print(f"[yellow]Warning: Population file not found: {population_file}[/yellow]")

    # If encoded files requested, load them from encode_evaluation stage
    if load_encoded:
        # Load pre-encoded files created by DVC encode_evaluation stage
        reference_encoded_file = data_root / "encoded" / f"reference_{model_id}.csv"
        synthetic_encoded_file = data_root / "encoded" / f"synthetic_{model_id}.csv"

        ref_enc = None
        syn_enc = None

        if reference_encoded_file.exists():
            console.print(f"[dim]Loading pre-encoded reference file: {reference_encoded_file}[/dim]")
            ref_enc = pd.read_csv(reference_encoded_file)
        else:
            console.print(f"[yellow]Warning: Pre-encoded reference file not found: {reference_encoded_file}[/yellow]")

        if synthetic_encoded_file.exists():
            console.print(f"[dim]Loading pre-encoded synthetic file: {synthetic_encoded_file}[/dim]")
            syn_enc = pd.read_csv(synthetic_encoded_file)
        else:
            console.print(f"[yellow]Warning: Pre-encoded synthetic file not found: {synthetic_encoded_file}[/yellow]")

        return ref_dec, syn_dec, trn_dec, pop_dec, ref_enc, syn_enc

    return ref_dec, syn_dec, trn_dec, pop_dec


def load_encoder(folder: Path, model_id: str) -> Optional[RDTDatasetEncoder]:
    """
    Load fitted encoder for a generation.

    Args:
        folder: Experiment folder path
        model_id: Model identifier

    Returns:
        Fitted RDTDatasetEncoder or None if not found
    """
    models_dir = folder / "models"

    # Try evaluation_encoder first (used for metrics evaluation)
    encoder_path = models_dir / f"evaluation_encoder_{model_id}.pkl"

    if not encoder_path.exists():
        # Fallback to training_encoder if evaluation encoder not found
        encoder_path = models_dir / f"training_encoder_{model_id}.pkl"

    if not encoder_path.exists():
        console.print(f"[yellow]Warning: Encoder not found in {models_dir}/[/yellow]")
        console.print(f"[yellow]  Tried: evaluation_encoder_{model_id}.pkl, training_encoder_{model_id}.pkl[/yellow]")
        return None

    try:
        console.print(f"[dim]Loading encoder: {encoder_path.name}[/dim]")
        return RDTDatasetEncoder.load(encoder_path)
    except Exception as e:
        console.print(f"[red]Error loading encoder: {e}[/red]")
        return None



def deep_merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    The override dict takes precedence over base dict. For nested dicts, merge recursively.
    For lists and other types, override completely replaces base.

    Args:
        base: Base configuration (lower priority)
        override: Override configuration (higher priority)

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_configs(result[key], value)
        else:
            # Override completely replaces (for lists, primitives, etc.)
            result[key] = value

    return result


def encode_data(
    encoder: RDTDatasetEncoder,
    data: pd.DataFrame,
    data_name: str = "data"
) -> pd.DataFrame:
    """
    Encode decoded data using fitted encoder.

    Args:
        encoder: Fitted RDTDatasetEncoder
        data: Decoded dataframe
        data_name: Name for logging

    Returns:
        Encoded dataframe
    """
    console.print(f"[dim]Encoding {data_name}...[/dim]")
    return encoder.transform(data)


def load_generation_config(folder: Path, generation: int) -> Optional[Dict[str, Any]]:
    """
    Load configuration for a specific generation from checkpoints/params_*.yaml

    Args:
        folder: Experiment folder path
        generation: Generation number

    Returns:
        Configuration dictionary or None
    """
    checkpoints_dir = folder / "checkpoints"
    if not checkpoints_dir.exists():
        console.print(f"[dim]No checkpoints directory found in {folder}[/dim]")
        return None

    # Find params file for this generation
    params_files = list(checkpoints_dir.glob(f"params_*_gen_{generation}.yaml"))
    if not params_files:
        console.print(f"[dim]No checkpoint params file found for generation {generation}[/dim]")
        console.print(f"[dim]Searched pattern: {checkpoints_dir}/params_*_gen_{generation}.yaml[/dim]")
        return None

    # Load the first matching file
    console.print(f"[dim]Loading checkpoint config: {params_files[0].name}[/dim]")
    with open(params_files[0], 'r') as f:
        return yaml.safe_load(f)


def compute_statistical_metrics_post_training(
    folder: Path,
    model_id: str,
    parsed: Dict[str, str],
    metadata: SingleTableMetadata,
    metadata_path: Path,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    show_tables: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Compute statistical metrics for a generation post-training.

    Args:
        folder: Experiment folder path
        model_id: Model identifier
        parsed: Parsed model ID components
        metadata: SDV metadata object
        metadata_path: Path to metadata file
        config: Optional configuration (from params.yaml)
        force: If True, overwrite existing metrics

    Returns:
        Metrics dictionary or None if skipped
    """
    # Check if metrics already exist
    metrics_file = folder / "metrics" / f"statistical_similarity_{model_id}.json"
    if metrics_file.exists() and not force:
        console.print(f"[yellow]Skipping statistical metrics (already exists): {metrics_file.name}[/yellow]")
        return None

    console.print(f"[cyan]Computing statistical metrics for generation {parsed['generation']}...[/cyan]")

    # Define encoded vs decoded metrics
    ENCODED_METRICS = {
        'alpha_precision', 'prdc_score', 'jensenshannon_synthcity',
        'jensenshannon_syndat', 'jensenshannon_nannyml',
        'wasserstein_distance', 'maximum_mean_discrepancy', 'ks_complement'
    }

    DECODED_METRICS = {
        'tv_complement', 'table_structure', 'semantic_structure',
        'boundary_adherence', 'category_adherence', 'new_row_synthesis'
    }

    # Get metrics config first (use default if not provided)
    if config and 'evaluation' in config and 'statistical_similarity' in config['evaluation']:
        metrics_config = config['evaluation']['statistical_similarity'].get('metrics', [])
    else:
        # Default metrics config
        metrics_config = [
            {'name': 'alpha_precision'},
            {'name': 'prdc_score'},
            {'name': 'wasserstein_distance'},
            {'name': 'ks_complement'},
            {'name': 'tv_complement'}
        ]

    # Check if any metrics need encoding
    needs_encoding = any(m.get('name') in ENCODED_METRICS for m in metrics_config)

    # Load data files - use pre-encoded files from encode_evaluation stage if needed (matches DVC)
    if needs_encoding:
        # Load both decoded and pre-encoded files
        ref_dec, syn_dec, trn_dec, _, ref_enc, syn_enc = load_generation_data(
            folder, model_id, metadata_path, population_file=None, config=config,
            use_original_files=False, load_encoded=True
        )

        if ref_enc is None or syn_enc is None:
            console.print("[yellow]Warning: Pre-encoded files not found, some encoded metrics may fail[/yellow]")
    else:
        # Only load decoded files
        ref_dec, syn_dec, trn_dec, _ = load_generation_data(
            folder, model_id, metadata_path, population_file=None, config=config,
            use_original_files=False, load_encoded=False
        )
        ref_enc = None
        syn_enc = None

    if ref_dec is None or syn_dec is None:
        console.print("[red]Error: Missing required data files for statistical metrics[/red]")
        return None

    # Load encoding config if available
    encoding_config = None
    if config and 'encoding' in config and config['encoding'].get('config_file'):
        encoding_config_path = Path(config['encoding']['config_file'])

        # Resolve relative paths (handles ../downloads/... paths)
        if not encoding_config_path.is_absolute():
            if not encoding_config_path.exists():
                # Try common locations with just filename
                filename = encoding_config_path.name
                candidates = [
                    Path("..") / "downloads" / filename,
                    Path("downloads") / filename,
                    Path("experiments") / "configs" / "encoding" / filename,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        encoding_config_path = candidate.resolve()
                        break

        if encoding_config_path.exists():
            console.print(f"[dim]Loading encoding config: {encoding_config_path}[/dim]")
            encoding_config = load_encoding_config(encoding_config_path)
        else:
            console.print(f"[yellow]Warning: Encoding config file not found: {config['encoding']['config_file']}[/yellow]")

    # Call evaluation function
    try:
        results = evaluate_statistical_metrics(
            ref_enc if ref_enc is not None else ref_dec,
            syn_enc if syn_enc is not None else syn_dec,
            metrics_config,
            experiment_name=f"{parsed['experiment_name']}_seed_{parsed['seed']}",
            metadata=metadata,
            reference_data_decoded=ref_dec,
            synthetic_data_decoded=syn_dec,
            reference_data_encoded=ref_enc,
            synthetic_data_encoded=syn_enc,
            encoded_metrics=ENCODED_METRICS,
            decoded_metrics=DECODED_METRICS,
            encoding_config=encoding_config
        )

        # Display results in terminal with Rich tables
        if show_tables:
            display_statistical_metrics(results)

        return results
    except Exception as e:
        console.print(f"[red]Error computing statistical metrics: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def compute_detection_metrics_post_training(
    folder: Path,
    model_id: str,
    parsed: Dict[str, str],
    metadata: SingleTableMetadata,
    metadata_path: Path,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    show_tables: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Compute detection metrics for a generation post-training.
    """
    # Check if metrics already exist
    metrics_file = folder / "metrics" / f"detection_evaluation_{model_id}.json"
    if metrics_file.exists() and not force:
        console.print(f"[yellow]Skipping detection metrics (already exists): {metrics_file.name}[/yellow]")
        return None

    console.print(f"[cyan]Computing detection metrics for generation {parsed['generation']}...[/cyan]")

    # Load pre-encoded data from encode_evaluation stage (matches DVC)
    # Detection metrics always use encoded data
    ref_dec, syn_dec, trn_dec, _, ref_enc, syn_enc = load_generation_data(
        folder, model_id, metadata_path, population_file=None, config=config,
        use_original_files=False, load_encoded=True
    )

    if ref_enc is None or syn_enc is None:
        console.print("[red]Error: Missing required pre-encoded data files for detection metrics[/red]")
        return None

    # Get detection config
    if config and 'evaluation' in config and 'detection_evaluation' in config['evaluation']:
        methods_config = config['evaluation']['detection_evaluation'].get('methods', [])
        common_params = config['evaluation']['detection_evaluation'].get('common_params', {
            'n_folds': 5, 'random_state': 42, 'reduction': 'mean'
        })
    else:
        # Default detection config
        methods_config = [
            {'name': 'detection_gmm'},
            {'name': 'detection_xgb'},
            {'name': 'detection_mlp'},
            {'name': 'detection_linear'}
        ]
        common_params = {'n_folds': 5, 'random_state': 42, 'reduction': 'mean'}

    # Load encoding config
    encoding_config = None
    if config and 'encoding' in config and config['encoding'].get('config_file'):
        encoding_config_path = Path(config['encoding']['config_file'])

        # Resolve relative paths (handles ../downloads/... paths)
        if not encoding_config_path.is_absolute():
            if not encoding_config_path.exists():
                # Try common locations with just filename
                filename = encoding_config_path.name
                candidates = [
                    Path("..") / "downloads" / filename,
                    Path("downloads") / filename,
                    Path("experiments") / "configs" / "encoding" / filename,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        encoding_config_path = candidate.resolve()
                        break

        if encoding_config_path.exists():
            console.print(f"[dim]Loading encoding config: {encoding_config_path}[/dim]")
            encoding_config = load_encoding_config(encoding_config_path)
        else:
            console.print(f"[yellow]Warning: Encoding config file not found: {config['encoding']['config_file']}[/yellow]")

    # Call evaluation function
    try:
        results = evaluate_detection_metrics(
            ref_enc,
            syn_enc,
            metadata,
            methods_config,
            common_params,
            parsed['experiment_name'],
            encoding_config
        )

        # Ensure JSON serializable
        results = ensure_json_serializable(results)

        # Display results in terminal
        if show_tables and DISPLAY_FUNCTIONS['detection']:
            DISPLAY_FUNCTIONS['detection'](results)

        return results
    except Exception as e:
        console.print(f"[red]Error computing detection metrics: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def compute_hallucination_metrics_post_training(
    folder: Path,
    model_id: str,
    parsed: Dict[str, str],
    metadata: SingleTableMetadata,
    metadata_path: Path,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    show_tables: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Compute hallucination metrics for a generation post-training.
    """
    # Check if metrics already exist
    metrics_file = folder / "metrics" / f"hallucination_{model_id}.json"
    if metrics_file.exists() and not force:
        console.print(f"[yellow]Skipping hallucination metrics (already exists): {metrics_file.name}[/yellow]")
        return None

    console.print(f"[cyan]Computing hallucination metrics for generation {parsed['generation']}...[/cyan]")

    # Get population file path from config
    population_file = None
    console.print(f"[dim]Debug: config is None: {config is None}[/dim]")
    if config:
        console.print(f"[dim]Debug: 'data' in config: {'data' in config}[/dim]")

    if config and 'data' in config:
        pop_file_str = config['data'].get('population_file')
        console.print(f"[dim]Debug: population_file from config: {pop_file_str}[/dim]")
        if pop_file_str:
            population_file_path = Path(pop_file_str)
            console.print(f"[dim]Debug: Path object created: {population_file_path}[/dim]")
            console.print(f"[dim]Debug: is_absolute: {population_file_path.is_absolute()}[/dim]")

            # Try multiple resolution strategies for relative paths
            if not population_file_path.is_absolute():
                console.print(f"[dim]Debug: Trying path.exists(): {population_file_path.exists()}[/dim]")
                # Try 1: As-is (relative to CWD) - handles ../downloads/... paths
                if population_file_path.exists():
                    population_file = population_file_path.resolve()
                    console.print(f"[dim]Debug: Found via Try 1 (as-is): {population_file}[/dim]")
                # Try 2: Relative to experiment folder (for paths like data/file.csv)
                elif not str(population_file_path).startswith('..') and (folder / population_file_path).exists():
                    population_file = (folder / population_file_path).resolve()
                    console.print(f"[dim]Debug: Found via Try 2 (relative to folder): {population_file}[/dim]")
                # Try 3: Just filename in common locations
                else:
                    console.print(f"[dim]Debug: Trying candidate paths...[/dim]")
                    filename = population_file_path.name
                    candidates = [
                        Path("..") / "downloads" / filename,
                        Path("downloads") / filename,
                        folder / "data" / filename,
                        folder / ".." / "downloads" / filename,
                        Path("data") / filename,
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            population_file = candidate.resolve()
                            break
            else:
                # Absolute path - use directly
                if population_file_path.exists():
                    population_file = population_file_path

    if not population_file or not population_file.exists():
        console.print("[red]Error: Population file not found. Please specify 'data.population_file' in config.[/red]")
        if config and 'data' in config:
            console.print(f"[yellow]Config specifies: {config['data'].get('population_file')}[/yellow]")
        console.print("[yellow]Hint: Use --metadata option or check that paths in checkpoint params are correct[/yellow]")
        return None

    console.print(f"[dim]Using population file: {population_file}[/dim]")

    # Load data with population file
    # CRITICAL: Hallucination metrics use ORIGINAL files from config (matches DVC hallucination_evaluation)
    ref_dec, syn_dec, trn_dec, pop_dec = load_generation_data(
        folder, model_id, metadata_path, population_file=population_file, config=config, use_original_files=True
    )

    if ref_dec is None or syn_dec is None or trn_dec is None or pop_dec is None:
        console.print("[red]Error: Missing required data files for hallucination metrics[/red]")
        return None

    # Get hallucination config
    if config and 'evaluation' in config and 'hallucination' in config['evaluation']:
        query_file = config['evaluation']['hallucination'].get('query_file')
    else:
        # Default query file
        query_file = "queries/validation.sql"

    query_file_path = Path(query_file)
    if not query_file_path.exists():
        console.print(f"[red]Error: Query file not found: {query_file_path}[/red]")
        return None

    # Call evaluation function with decoded population data
    try:
        results = evaluate_hallucination_metrics(
            population=pop_dec,
            training=trn_dec,
            reference=ref_dec,
            synthetic=syn_dec,
            metadata=metadata,
            query_file=query_file_path,
            experiment_name=parsed['experiment_name']
        )

        # Display results in terminal
        if show_tables and DISPLAY_FUNCTIONS['hallucination']:
            DISPLAY_FUNCTIONS['hallucination'](results)

        return results
    except Exception as e:
        console.print(f"[red]Error computing hallucination metrics: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def compute_privacy_metrics_post_training(
    folder: Path,
    model_id: str,
    parsed: Dict[str, str],
    metadata: SingleTableMetadata,
    metadata_path: Path,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    show_tables: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Compute privacy metrics for a generation post-training.

    Args:
        folder: Experiment folder path
        model_id: Model identifier
        parsed: Parsed model ID components
        metadata: SDV metadata object
        metadata_path: Path to metadata file
        config: Optional configuration (from params.yaml)
        force: If True, overwrite existing metrics
        show_tables: If True, display Rich tables in terminal

    Returns:
        Metrics dictionary or None if skipped
    """
    # Check if metrics already exist
    metrics_file = folder / "metrics" / f"privacy_{model_id}.json"
    if metrics_file.exists() and not force:
        console.print(f"[yellow]Skipping privacy metrics (already exists): {metrics_file.name}[/yellow]")
        return None

    console.print(f"[cyan]Computing privacy metrics for generation {parsed['generation']}...[/cyan]")

    # Define encoded vs decoded privacy metrics
    ENCODED_PRIVACY_METRICS = {'dcr_baseline_protection'}
    DECODED_PRIVACY_METRICS = {'k_anonymization'}
    MULTIDATASET_PRIVACY_METRICS = {'k_anonymization'}

    # Get metrics config first (use default if not provided)
    if config and 'evaluation' in config and 'privacy' in config['evaluation']:
        metrics_config = config['evaluation']['privacy'].get('metrics', [])
    else:
        # Default privacy metrics config
        metrics_config = [
            {'name': 'dcr_baseline_protection'}
        ]

    # Check if any metrics need encoding or multi-dataset support
    needs_encoding = any(m.get('name') in ENCODED_PRIVACY_METRICS for m in metrics_config)
    needs_multidataset = any(m.get('name') in MULTIDATASET_PRIVACY_METRICS for m in metrics_config)

    # Get population file path from config (for k-anonymization and other multi-dataset metrics)
    population_file = None
    if needs_multidataset and config and 'data' in config:
        pop_file_str = config['data'].get('population_file')
        if pop_file_str:
            population_file_path = Path(pop_file_str)

            # Try multiple resolution strategies for relative paths
            if not population_file_path.is_absolute():
                # Try 1: As-is (relative to CWD) - handles ../downloads/... paths
                if population_file_path.exists():
                    population_file = population_file_path.resolve()
                    console.print(f"[dim]Found population file: {population_file}[/dim]")
                # Try 2: Relative to experiment folder (for paths like data/file.csv)
                elif not str(population_file_path).startswith('..') and (folder / population_file_path).exists():
                    population_file = (folder / population_file_path).resolve()
                    console.print(f"[dim]Found population file (relative to folder): {population_file}[/dim]")
                # Try 3: Just filename in common locations
                else:
                    filename = population_file_path.name
                    candidates = [
                        Path("..") / "downloads" / filename,
                        Path("downloads") / filename,
                        folder / "data" / filename,
                        folder / ".." / "downloads" / filename,
                        Path("data") / filename,
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            population_file = candidate.resolve()
                            console.print(f"[dim]Found population file: {population_file}[/dim]")
                            break
            else:
                # Absolute path - use directly
                if population_file_path.exists():
                    population_file = population_file_path
                    console.print(f"[dim]Using population file: {population_file}[/dim]")

    # Load data files - use pre-encoded files from encode_evaluation stage if needed (matches DVC)
    if needs_encoding:
        # Load both decoded and pre-encoded files
        ref_dec, syn_dec, trn_dec, pop_dec, ref_enc, syn_enc = load_generation_data(
            folder, model_id, metadata_path, population_file=population_file, config=config,
            use_original_files=False, load_encoded=True
        )

        if ref_enc is None or syn_enc is None:
            console.print("[yellow]Warning: Pre-encoded files not found, some encoded privacy metrics may fail[/yellow]")
    else:
        # Only load decoded files
        ref_dec, syn_dec, trn_dec, pop_dec = load_generation_data(
            folder, model_id, metadata_path, population_file=population_file, config=config,
            use_original_files=False, load_encoded=False
        )
        ref_enc = None
        syn_enc = None

    if ref_dec is None or syn_dec is None:
        console.print("[red]Error: Missing required data files for privacy metrics[/red]")
        return None

    # Load encoding config if available
    encoding_config = None
    if config and 'encoding' in config and config['encoding'].get('config_file'):
        encoding_config_path = Path(config['encoding']['config_file'])

        # Resolve relative paths
        if not encoding_config_path.is_absolute():
            if not encoding_config_path.exists():
                filename = encoding_config_path.name
                candidates = [
                    Path("..") / "downloads" / filename,
                    Path("downloads") / filename,
                    Path("experiments") / "configs" / "encoding" / filename,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        encoding_config_path = candidate.resolve()
                        break

        if encoding_config_path.exists():
            console.print(f"[dim]Loading encoding config: {encoding_config_path}[/dim]")
            encoding_config = load_encoding_config(encoding_config_path)
        else:
            console.print(f"[yellow]Warning: Encoding config file not found: {config['encoding']['config_file']}[/yellow]")

    # Call evaluation function
    try:
        results = evaluate_privacy_metrics(
            ref_enc if ref_enc is not None else ref_dec,
            syn_enc if syn_enc is not None else syn_dec,
            metrics_config,
            experiment_name=f"{parsed['experiment_name']}_seed_{parsed['seed']}",
            metadata=metadata,
            reference_data_decoded=ref_dec,
            synthetic_data_decoded=syn_dec,
            reference_data_encoded=ref_enc,
            synthetic_data_encoded=syn_enc,
            encoding_config=encoding_config,
            population_data=pop_dec,
            training_data=trn_dec
        )

        # Display results in terminal with Rich tables
        if show_tables:
            display_privacy_metrics(results)

        return results
    except Exception as e:
        console.print(f"[red]Error computing privacy metrics: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def save_metrics(
    folder: Path,
    model_id: str,
    metric_type: str,
    results: Dict[str, Any],
    merge: bool = True
) -> None:
    """
    Save metrics results to JSON file.

    Args:
        folder: Experiment folder path
        model_id: Model identifier
        metric_type: Type of metric (statistical_similarity, detection_evaluation, hallucination)
        results: Metrics results dictionary
        merge: If True, merge with existing metrics instead of overwriting (default: True)
    """
    metrics_dir = folder / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / f"{metric_type}_{model_id}.json"

    # If merge=True and file exists, load existing metrics and merge
    if merge and metrics_file.exists():
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                existing_metrics = json.load(f)

            # Merge metrics: update existing with new results
            # For statistical_similarity, merge the metrics dict
            if 'metrics' in existing_metrics and 'metrics' in results:
                # Keep existing metrics, update with new ones
                existing_metrics['metrics'].update(results['metrics'])
                # Update metadata with new timestamp
                if 'metadata' in results:
                    existing_metrics['metadata'] = results['metadata']
                results = existing_metrics
                console.print(f"[dim]Merged with existing metrics (kept {len(existing_metrics['metrics'])} total)[/dim]")
            elif 'individual_scores' in existing_metrics and 'individual_scores' in results:
                # For detection, merge individual_scores
                existing_metrics['individual_scores'].update(results['individual_scores'])
                if 'metadata' in results:
                    existing_metrics['metadata'] = results['metadata']
                # Recalculate ensemble score if needed
                if 'ensemble_score' in results:
                    existing_metrics['ensemble_score'] = results['ensemble_score']
                results = existing_metrics
                console.print(f"[dim]Merged with existing metrics (kept {len(existing_metrics['individual_scores'])} total)[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not merge with existing metrics: {e}[/yellow]")
            console.print(f"[yellow]Will overwrite existing file[/yellow]")

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Saved: {metrics_file.name}[/green]")

    # Generate and save report if applicable
    report_file = metrics_dir / f"{metric_type.replace('_similarity', '').replace('_evaluation', '')}_report_{model_id}.txt"

    try:
        if metric_type == "statistical_similarity":
            report = generate_statistical_report(results)
        elif metric_type == "detection_evaluation":
            report = generate_detection_report(results)
        elif metric_type == "hallucination":
            report = generate_hallucination_report(results)
        elif metric_type == "privacy":
            report = generate_privacy_report(results)
        else:
            return

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        console.print(f"[green]✓ Saved: {report_file.name}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate report: {e}[/yellow]")


def process_single_folder(
    exp_folder: Path,
    metadata_path_override: Optional[Path],
    generation_filter: Optional[int],
    config_data: Optional[Dict[str, Any]],
    force: bool,
    show_tables: bool,
    user_specified_metrics: bool,
    metrics: List[MetricType]
) -> Dict[str, Any]:
    """
    Process a single experiment folder.

    This function is extracted to enable parallel processing of multiple folders.

    Args:
        exp_folder: Path to experiment folder
        metadata_path_override: User-specified metadata path (from --metadata)
        generation_filter: If specified, only process this generation
        config_data: Configuration dictionary (from --config)
        force: Whether to overwrite existing metrics
        show_tables: Whether to display Rich tables
        user_specified_metrics: Whether user explicitly specified --metrics
        metrics: List of metric types to compute

    Returns:
        Dictionary with processing results: {
            'folder': exp_folder,
            'success': bool,
            'generations_processed': int,
            'error': Optional[str]
        }
    """
    try:
        console.print(f"\n[bold]Processing folder: {exp_folder}[/bold]")

        # Load config for metadata discovery
        # If user provided partial config (e.g., metrics_only.yaml), try to load checkpoint config
        # for metadata path and other missing data paths
        effective_config = config_data
        checkpoint_config = None

        if not metadata_path_override:
            # Try to load checkpoint config for metadata discovery (even if user provided partial config)
            checkpoint_config = load_generation_config(exp_folder, generation_filter if generation_filter is not None else 0)

            # If no user config, use checkpoint config entirely
            if not effective_config:
                effective_config = checkpoint_config
            # If user config exists, deep merge with checkpoint config
            # User config takes precedence for overlapping keys (including metric lists)
            elif checkpoint_config:
                # Deep merge: checkpoint provides defaults, user config overrides
                effective_config = deep_merge_configs(checkpoint_config, effective_config)
                console.print("[dim]Merged user config with checkpoint config[/dim]")

        # Debug: Show key config paths
        if effective_config:
            if 'data' in effective_config:
                console.print(f"[dim]Config data paths:[/dim]")
                if 'population_file' in effective_config['data']:
                    console.print(f"[dim]  population_file: {effective_config['data']['population_file']}[/dim]")
                if 'training_file' in effective_config['data']:
                    console.print(f"[dim]  training_file: {effective_config['data']['training_file']}[/dim]")
                if 'reference_file' in effective_config['data']:
                    console.print(f"[dim]  reference_file: {effective_config['data']['reference_file']}[/dim]")
            if 'encoding' in effective_config and 'config_file' in effective_config['encoding']:
                console.print(f"[dim]  encoding_config: {effective_config['encoding']['config_file']}[/dim]")
            if 'evaluation' in effective_config and 'statistical_similarity' in effective_config['evaluation']:
                metrics_list = effective_config['evaluation']['statistical_similarity'].get('metrics', [])
                metric_names = [m.get('name') for m in metrics_list if isinstance(m, dict)]
                console.print(f"[dim]  statistical_metrics: {metric_names}[/dim]")

        # Find metadata file
        # Priority: --metadata flag > config['data']['metadata_file'] > checkpoint config > auto-discovery
        metadata_path = metadata_path_override if metadata_path_override else find_metadata_file(exp_folder, effective_config)
        if not metadata_path:
            error_msg = f"Could not find metadata.json for {exp_folder}"
            console.print(f"[red]Error: {error_msg}[/red]")
            console.print("[yellow]Specify with --metadata option or ensure checkpoint params exist[/yellow]")
            return {
                'folder': exp_folder,
                'success': False,
                'generations_processed': 0,
                'error': error_msg
            }

        console.print(f"Using metadata: {metadata_path}")
        metadata_obj = SingleTableMetadata.load_from_json(str(metadata_path))

        # Discover generations
        generations = discover_generation_files(exp_folder, generation_filter)
        if not generations:
            warning_msg = f"No generations found in {exp_folder}"
            console.print(f"[yellow]Warning: {warning_msg}[/yellow]")
            return {
                'folder': exp_folder,
                'success': True,
                'generations_processed': 0,
                'error': warning_msg
            }

        console.print(f"Found {len(generations)} generation(s) to process")

        # Determine which metrics to compute
        # If user explicitly specified --metrics flag, use their choice
        # Otherwise, auto-detect from config structure
        if user_specified_metrics:
            # User explicitly specified metrics via --metrics flag
            metric_values = [m.value for m in metrics]
            compute_all = MetricType.all.value in metric_values
            compute_statistical = compute_all or MetricType.statistical.value in metric_values
            compute_detection = compute_all or MetricType.detection.value in metric_values
            compute_hallucination = compute_all or MetricType.hallucination.value in metric_values
            compute_privacy = compute_all or MetricType.privacy.value in metric_values
        else:
            # Auto-detect from config structure
            # Use the USER'S config (config_data), not effective_config
            # This ensures we only compute what the user explicitly configured
            # (effective_config includes merged checkpoint data which has all metric types)
            compute_statistical = False
            compute_detection = False
            compute_hallucination = False
            compute_privacy = False

            if config_data and 'evaluation' in config_data:
                eval_config = config_data['evaluation']

                # Check for statistical metrics configuration
                if 'statistical_similarity' in eval_config:
                    compute_statistical = True
                    console.print("[dim]Auto-detected: statistical metrics configured[/dim]")

                # Check for detection metrics configuration
                if 'detection_evaluation' in eval_config:
                    compute_detection = True
                    console.print("[dim]Auto-detected: detection metrics configured[/dim]")

                # Check for hallucination metrics configuration
                if 'hallucination' in eval_config:
                    compute_hallucination = True
                    console.print("[dim]Auto-detected: hallucination metrics configured[/dim]")

                # Check for privacy metrics configuration
                if 'privacy' in eval_config:
                    compute_privacy = True
                    console.print("[dim]Auto-detected: privacy metrics configured[/dim]")

            # If no config provided or no evaluation section, fall back to computing all
            if not config_data or not (compute_statistical or compute_detection or compute_hallucination or compute_privacy):
                if not config_data:
                    console.print("[dim]No config provided, computing all metrics[/dim]")
                else:
                    console.print("[dim]No metric configuration found, computing all metrics[/dim]")
                compute_statistical = True
                compute_detection = True
                compute_hallucination = True
                compute_privacy = True

        # Process each generation
        generations_processed = 0
        for gen_info in generations:
            model_id = gen_info['model_id']
            parsed = gen_info['parsed']
            gen_num = gen_info['generation']

            console.print(f"\n[bold yellow]Generation {gen_num}[/bold yellow] ({model_id})")

            # Use effective_config (merged user + checkpoint config) for all metrics
            # This ensures data paths, encoding config, etc. are available

            # Compute metrics
            if compute_statistical:
                results = compute_statistical_metrics_post_training(
                    exp_folder, model_id, parsed, metadata_obj, metadata_path, effective_config, force, show_tables
                )
                if results:
                    save_metrics(exp_folder, model_id, "statistical_similarity", results)

            if compute_detection:
                results = compute_detection_metrics_post_training(
                    exp_folder, model_id, parsed, metadata_obj, metadata_path, effective_config, force, show_tables
                )
                if results:
                    save_metrics(exp_folder, model_id, "detection_evaluation", results)

            if compute_hallucination:
                results = compute_hallucination_metrics_post_training(
                    exp_folder, model_id, parsed, metadata_obj, metadata_path, effective_config, force, show_tables
                )
                if results:
                    save_metrics(exp_folder, model_id, "hallucination", results)

            if compute_privacy:
                results = compute_privacy_metrics_post_training(
                    exp_folder, model_id, parsed, metadata_obj, metadata_path, effective_config, force, show_tables
                )
                if results:
                    save_metrics(exp_folder, model_id, "privacy", results)

            generations_processed += 1

        return {
            'folder': exp_folder,
            'success': True,
            'generations_processed': generations_processed,
            'error': None
        }

    except Exception as e:
        error_msg = f"Error processing folder {exp_folder}: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        import traceback
        traceback.print_exc()
        return {
            'folder': exp_folder,
            'success': False,
            'generations_processed': 0,
            'error': error_msg
        }


@app.command()
def main(
    folder: Annotated[
        Optional[Path],
        typer.Option(
            "--folder",
            help="Experiment folder path",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True
        )
    ] = None,
    folders_pattern: Annotated[
        Optional[str],
        typer.Option(
            "--folders-pattern",
            help='Glob pattern to match multiple folders (e.g., "./mimic_*dseed233*/")'
        )
    ] = None,
    generation: Annotated[
        Optional[int],
        typer.Option(
            "--generation",
            help="Specific generation to compute (default: all)",
            min=0
        )
    ] = None,
    metrics: Annotated[
        List[MetricType],
        typer.Option(
            "--metrics",
            help="Which metrics to compute (can specify multiple)",
            case_sensitive=False
        )
    ] = [MetricType.all],
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="Custom params.yaml config file (optional)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True
        )
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing metrics"
        )
    ] = False,
    metadata: Annotated[
        Optional[Path],
        typer.Option(
            "--metadata",
            help="Path to metadata.json file (auto-discovered if not specified)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True
        )
    ] = None,
    show_tables: Annotated[
        bool,
        typer.Option(
            "--show-tables/--no-show-tables",
            help="Display Rich tables in terminal (default: True)"
        )
    ] = True,
    n_jobs: Annotated[
        int,
        typer.Option(
            "--n-jobs",
            help="Number of parallel jobs for folder processing (1=sequential, -1=all CPUs)"
        )
    ] = 1
):
    """
    [bold cyan]Compute metrics on experiment folders post-training[/bold cyan]

    This tool enables computing metrics on existing experiment folders without
    re-running the DVC pipeline. Useful for:

    • Adding new metrics to old experiments
    • Re-computing metrics with different configurations
    • Filling missing metrics after pipeline interruptions
    • Experimenting with metric parameters

    [bold]Examples:[/bold]

      # Simplest: Use repo params.yaml (reads metadata + encoding paths from config)
      python compute_metrics_cli.py --folder ./exp/ --config params.yaml

      # Compute all metrics on one folder
      python compute_metrics_cli.py --folder ./experiment_folder/

      # Compute specific metrics
      python compute_metrics_cli.py --folder ./exp/ --metrics statistical --metrics detection

      # Batch process multiple folders (sequential)
      python compute_metrics_cli.py --folders-pattern "./mimic_*dseed233*/"

      # Batch process multiple folders in parallel (8 workers)
      python compute_metrics_cli.py --folders-pattern "./mimic_*dseed233*/" --n-jobs 8

      # Use all available CPU cores for parallel processing
      python compute_metrics_cli.py --folders-pattern "./mimic_*/" --n-jobs -1

      # Force recompute
      python compute_metrics_cli.py --folder ./exp/ --force

    [bold]Parallelization:[/bold]

      Use --n-jobs to process multiple folders in parallel:
      • n_jobs=1 (default): Sequential processing
      • n_jobs=N: Use N parallel workers
      • n_jobs=-1: Use all available CPU cores

      Parallelization provides near-linear speedup for batch processing.
      Example: 20 folders on 8-core machine → ~8x faster

    [bold]Config Integration:[/bold]

      When --config is provided, the tool automatically reads:
      • Metadata path from config['data']['metadata_file']
      • Encoding config from config['encoding']['config_file']
      • Metrics configuration from config['evaluation']
    """

    # Validate arguments
    if not folder and not folders_pattern:
        console.print("[red]Error: Either --folder or --folders-pattern must be specified[/red]")
        raise typer.Exit(code=1)

    # Collect folders to process
    folders = []
    if folder:
        folders.append(folder)
    elif folders_pattern:
        folders = list(Path().glob(folders_pattern))
        if not folders:
            console.print(f"[red]Error: No folders matched pattern: {folders_pattern}[/red]")
            raise typer.Exit(code=1)

    # Load config if provided
    config_data = None
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)

    # Check if user explicitly specified metrics (not using default)
    user_specified_metrics = metrics != [MetricType.all]

    # Process each folder
    console.print(f"\n[bold cyan]Post-Training Metrics Computation[/bold cyan]")
    console.print(f"Processing {len(folders)} folder(s)")

    # Determine execution mode
    if n_jobs == 1:
        console.print("[dim]Running in sequential mode (n_jobs=1)[/dim]\n")
        # Sequential execution (original behavior)
        for exp_folder in folders:
            process_single_folder(
                exp_folder=exp_folder,
                metadata_path_override=metadata,
                generation_filter=generation,
                config_data=config_data,
                force=force,
                show_tables=show_tables,
                user_specified_metrics=user_specified_metrics,
                metrics=metrics
            )
    else:
        # Parallel execution
        import multiprocessing

        # Handle n_jobs=-1 (use all CPUs)
        if n_jobs == -1:
            actual_n_jobs = multiprocessing.cpu_count()
        else:
            actual_n_jobs = min(n_jobs, len(folders))  # Don't use more jobs than folders

        console.print(f"[dim]Running in parallel mode with {actual_n_jobs} workers (n_jobs={n_jobs})[/dim]")
        console.print("[dim]Progress will be reported by joblib verbose output[/dim]\n")

        try:
            # Run folders in parallel
            results = Parallel(n_jobs=actual_n_jobs, backend='multiprocessing', verbose=10)(
                delayed(process_single_folder)(
                    exp_folder=exp_folder,
                    metadata_path_override=metadata,
                    generation_filter=generation,
                    config_data=config_data,
                    force=force,
                    show_tables=show_tables,
                    user_specified_metrics=user_specified_metrics,
                    metrics=metrics
                )
                for exp_folder in folders
            )

            # Print summary of results
            console.print("\n[bold cyan]Parallel Processing Summary[/bold cyan]")
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            total_generations = sum(r['generations_processed'] for r in results)

            console.print(f"  Folders processed: {len(results)}")
            console.print(f"  Successful: {successful}")
            console.print(f"  Failed: {failed}")
            console.print(f"  Total generations processed: {total_generations}")

            if failed > 0:
                console.print("\n[yellow]Failed folders:[/yellow]")
                for r in results:
                    if not r['success']:
                        console.print(f"  - {r['folder']}: {r['error']}")

        except Exception as e:
            console.print(f"\n[red]Parallel execution failed: {e}[/red]")
            console.print("[yellow]Falling back to sequential execution...[/yellow]\n")

            # Fallback to sequential execution
            for exp_folder in folders:
                process_single_folder(
                    exp_folder=exp_folder,
                    metadata_path_override=metadata,
                    generation_filter=generation,
                    config_data=config_data,
                    force=force,
                    show_tables=show_tables,
                    user_specified_metrics=user_specified_metrics,
                    metrics=metrics
                )

    console.print("\n[bold green]✓ Post-training metrics computation complete![/bold green]\n")


if __name__ == "__main__":
    app()
