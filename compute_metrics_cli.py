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

from sdpype.evaluation.statistical import evaluate_statistical_metrics, generate_statistical_report
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
    """Display statistical metrics results in terminal - matches DVC pipeline output"""
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
    else:
        console.print("❌ TableStructure failed", style="bold red")

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
    else:
        console.print("❌ SemanticStructure failed", style="bold red")

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
    else:
        console.print("❌ NewRowSynthesis failed", style="bold red")

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
    else:
        console.print("❌ BoundaryAdherence failed", style="bold red")

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
    else:
        console.print("❌ CategoryAdherence failed", style="bold red")

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
    elif "alpha_precision" in metrics:
        console.print("❌ Alpha Precision failed", style="bold red")

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
    elif "prdc_score" in metrics:
        console.print("❌ PRDC Score failed", style="bold red")

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
    elif "wasserstein_distance" in metrics:
        console.print("❌ Wasserstein Distance failed", style="bold red")

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
    elif "maximum_mean_discrepancy" in metrics:
        console.print("❌ Maximum Mean Discrepancy failed", style="bold red")

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
        elif js_variant in metrics:
            console.print(f"❌ {js_variant.replace('_', ' ').title()} failed", style="bold red")

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
    elif "ks_complement" in metrics:
        console.print("❌ KSComplement failed", style="bold red")

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
    elif "tv_complement" in metrics:
        console.print("❌ TVComplement failed", style="bold red")

    console.print("\n✅ Statistical metrics evaluation completed", style="bold green")

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

        # Resolve relative paths
        if not encoding_config_path.is_absolute() and not encoding_config_path.exists():
            # Try relative to parent directory
            parent_path = Path("..") / encoding_config_path
            if parent_path.exists():
                encoding_config_path = parent_path.resolve()

        if encoding_config_path.exists():
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

        # Resolve relative paths
        if not encoding_config_path.is_absolute() and not encoding_config_path.exists():
            # Try relative to current directory first
            if not encoding_config_path.exists():
                # Try relative to parent directory
                parent_path = Path("..") / encoding_config_path
                if parent_path.exists():
                    encoding_config_path = parent_path.resolve()

        if encoding_config_path.exists():
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
    if config and 'data' in config:
        pop_file_str = config['data'].get('population_file')
        if pop_file_str:
            population_file_path = Path(pop_file_str)

            # Try multiple resolution strategies for relative paths
            if not population_file_path.is_absolute():
                # Try 1: Absolute path or relative to CWD
                if population_file_path.exists():
                    population_file = population_file_path
                # Try 2: Relative to experiment folder
                elif (folder / population_file_path).exists():
                    population_file = folder / population_file_path
                # Try 3: Relative to parent directory (for ../downloads/... paths)
                elif (Path("..") / population_file_path).exists():
                    population_file = (Path("..") / population_file_path).resolve()
                # Try 4: Just filename in common locations
                else:
                    filename = population_file_path.name
                    candidates = [
                        folder / "data" / filename,
                        folder / ".." / "downloads" / filename,
                        Path("downloads") / filename,
                        Path("data") / filename,
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            population_file = candidate
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


def save_metrics(
    folder: Path,
    model_id: str,
    metric_type: str,
    results: Dict[str, Any]
) -> None:
    """
    Save metrics results to JSON file.

    Args:
        folder: Experiment folder path
        model_id: Model identifier
        metric_type: Type of metric (statistical_similarity, detection_evaluation, hallucination)
        results: Metrics results dictionary
    """
    metrics_dir = folder / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / f"{metric_type}_{model_id}.json"

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
        else:
            return

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        console.print(f"[green]✓ Saved: {report_file.name}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate report: {e}[/yellow]")


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
    ] = True
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

      # Batch process multiple folders
      python compute_metrics_cli.py --folders-pattern "./mimic_*dseed233*/"

      # Force recompute
      python compute_metrics_cli.py --folder ./exp/ --force

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

    # Determine which metrics to compute
    metric_values = [m.value for m in metrics]
    compute_all = MetricType.all.value in metric_values
    compute_statistical = compute_all or MetricType.statistical.value in metric_values
    compute_detection = compute_all or MetricType.detection.value in metric_values
    compute_hallucination = compute_all or MetricType.hallucination.value in metric_values

    # Process each folder
    console.print(f"\n[bold cyan]Post-Training Metrics Computation[/bold cyan]")
    console.print(f"Processing {len(folders)} folder(s)\n")

    for exp_folder in folders:
        console.print(f"\n[bold]Processing folder: {exp_folder}[/bold]")

        # Load config for metadata discovery
        # If user provided partial config (e.g., metrics_only.yaml), try to load checkpoint config
        # for metadata path and other missing data paths
        effective_config = config_data
        checkpoint_config = None

        if not metadata:
            # Try to load checkpoint config for metadata discovery (even if user provided partial config)
            checkpoint_config = load_generation_config(exp_folder, generation if generation is not None else 0)

            # If no user config, use checkpoint config entirely
            if not effective_config:
                effective_config = checkpoint_config
            # If user config exists, deep merge with checkpoint config
            # User config takes precedence for overlapping keys (including metric lists)
            elif checkpoint_config:
                # Deep merge: checkpoint provides defaults, user config overrides
                effective_config = deep_merge_configs(checkpoint_config, effective_config)

        # Find metadata file
        # Priority: --metadata flag > config['data']['metadata_file'] > checkpoint config > auto-discovery
        metadata_path = metadata if metadata else find_metadata_file(exp_folder, effective_config)
        if not metadata_path:
            console.print(f"[red]Error: Could not find metadata.json for {exp_folder}[/red]")
            console.print("[yellow]Specify with --metadata option or ensure checkpoint params exist[/yellow]")
            continue

        console.print(f"Using metadata: {metadata_path}")
        metadata_obj = SingleTableMetadata.load_from_json(str(metadata_path))

        # Discover generations
        generations = discover_generation_files(exp_folder, generation)
        if not generations:
            console.print(f"[yellow]Warning: No generations found in {exp_folder}[/yellow]")
            continue

        console.print(f"Found {len(generations)} generation(s) to process")

        # Process each generation
        for gen_info in generations:
            model_id = gen_info['model_id']
            parsed = gen_info['parsed']
            gen_num = gen_info['generation']

            console.print(f"\n[bold yellow]Generation {gen_num}[/bold yellow] ({model_id})")

            # Load generation config
            gen_config = config_data if config_data else load_generation_config(exp_folder, gen_num)

            # Compute metrics
            if compute_statistical:
                results = compute_statistical_metrics_post_training(
                    exp_folder, model_id, parsed, metadata_obj, metadata_path, gen_config, force
                )
                if results:
                    save_metrics(exp_folder, model_id, "statistical_similarity", results)

            if compute_detection:
                results = compute_detection_metrics_post_training(
                    exp_folder, model_id, parsed, metadata_obj, metadata_path, gen_config, force
                )
                if results:
                    save_metrics(exp_folder, model_id, "detection_evaluation", results)

            if compute_hallucination:
                results = compute_hallucination_metrics_post_training(
                    exp_folder, model_id, parsed, metadata_obj, metadata_path, gen_config, force
                )
                if results:
                    save_metrics(exp_folder, model_id, "hallucination", results)

    console.print("\n[bold green]✓ Post-training metrics computation complete![/bold green]\n")


if __name__ == "__main__":
    app()
