#!/usr/bin/env python3
"""
Post-Training Metrics Computation CLI

Compute metrics on existing experiment folders without re-running the DVC pipeline.
Useful for:
- Adding new metrics to old experiments
- Re-computing metrics with different configurations
- Filling missing metrics after pipeline interruptions
- Experimenting with metric parameters

Usage:
    # Compute all metrics on one folder
    python compute_metrics_cli.py --folder ./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817/

    # Compute specific metrics
    python compute_metrics_cli.py --folder <folder> --metrics statistical detection

    # Compute specific generation only
    python compute_metrics_cli.py --folder <folder> --generation 5

    # Batch process multiple folders
    python compute_metrics_cli.py --folders-pattern "./mimic_iii_*dseed233*/"

    # Use custom config for metrics
    python compute_metrics_cli.py --folder <folder> --config custom_params.yaml
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sdv.metadata import SingleTableMetadata

from sdpype.evaluation.statistical import evaluate_statistical_metrics, generate_statistical_report
from sdpype.evaluation.detection import evaluate_detection_metrics, generate_detection_report, ensure_json_serializable
from sdpype.evaluation.hallucination import evaluate_hallucination_metrics, generate_hallucination_report
from sdpype.metadata import load_csv_with_metadata
from sdpype.encoding import load_encoding_config

console = Console()


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


def find_metadata_file(folder: Path) -> Optional[Path]:
    """
    Find metadata.json file by checking common locations.

    Args:
        folder: Experiment folder path

    Returns:
        Path to metadata.json or None
    """
    # Check common locations
    candidates = [
        folder / "metadata.json",
        folder / "data" / "metadata.json",
        Path("data") / "metadata.json",  # Look in repo data/
        Path("metadata.json")  # Look in current directory
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_generation_data(
    folder: Path,
    model_id: str,
    metadata_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all necessary data files for a generation.

    Returns:
        (reference_decoded, reference_encoded, synthetic_decoded, synthetic_encoded,
         training_decoded, training_encoded, population, training_binned)
    """
    data_root = folder / "data"

    # Construct file paths
    reference_decoded = data_root / "decoded" / f"reference_{model_id}.csv"
    reference_encoded = data_root / "encoded" / f"reference_{model_id}.csv"
    synthetic_decoded = data_root / "synthetic" / f"synthetic_data_{model_id}_decoded.csv"
    synthetic_encoded = data_root / "synthetic" / f"synthetic_data_{model_id}_encoded.csv"
    training_decoded = data_root / "decoded" / f"training_{model_id}.csv"
    training_encoded = data_root / "encoded" / f"training_{model_id}.csv"
    population_binned = data_root / "binned" / "population_data_for_hallucinations.csv"
    training_binned = data_root / "binned" / f"training_data_for_hallucinations.csv"

    # Load data
    ref_dec = load_csv_with_metadata(reference_decoded, metadata_path) if reference_decoded.exists() else None
    ref_enc = pd.read_csv(reference_encoded) if reference_encoded.exists() else None
    syn_dec = load_csv_with_metadata(synthetic_decoded, metadata_path) if synthetic_decoded.exists() else None
    syn_enc = pd.read_csv(synthetic_encoded) if synthetic_encoded.exists() else None
    trn_dec = load_csv_with_metadata(training_decoded, metadata_path) if training_decoded.exists() else None
    trn_enc = pd.read_csv(training_encoded) if training_encoded.exists() else None
    pop_bin = load_csv_with_metadata(population_binned, metadata_path, low_memory=False) if population_binned.exists() else None
    trn_bin = load_csv_with_metadata(training_binned, metadata_path, low_memory=False) if training_binned.exists() else None

    return ref_dec, ref_enc, syn_dec, syn_enc, trn_dec, trn_enc, pop_bin, trn_bin


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
        return None

    # Find params file for this generation
    params_files = list(checkpoints_dir.glob(f"params_*_gen_{generation}.yaml"))
    if not params_files:
        return None

    # Load the first matching file
    with open(params_files[0], 'r') as f:
        return yaml.safe_load(f)


def compute_statistical_metrics_post_training(
    folder: Path,
    model_id: str,
    parsed: Dict[str, str],
    metadata: SingleTableMetadata,
    metadata_path: Path,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False
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

    # Load data
    ref_dec, ref_enc, syn_dec, syn_enc, _, _, _, _ = load_generation_data(folder, model_id, metadata_path)

    if ref_dec is None or syn_dec is None:
        console.print("[red]Error: Missing required data files for statistical metrics[/red]")
        return None

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

    # Get metrics config (use default if not provided)
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

    # Load encoding config if available
    encoding_config = None
    if config and 'encoding' in config and config['encoding'].get('config_file'):
        encoding_config_path = Path(config['encoding']['config_file'])
        if encoding_config_path.exists():
            encoding_config = load_encoding_config(encoding_config_path)

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
    force: bool = False
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

    # Load data (detection needs encoded data)
    _, ref_enc, _, syn_enc, _, _, _, _ = load_generation_data(folder, model_id, metadata_path)

    if ref_enc is None or syn_enc is None:
        console.print("[red]Error: Missing encoded data files for detection metrics[/red]")
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
        if encoding_config_path.exists():
            encoding_config = load_encoding_config(encoding_config_path)

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
    force: bool = False
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

    # Load data
    ref_dec, _, syn_dec, _, trn_dec, _, pop_bin, _ = load_generation_data(folder, model_id, metadata_path)

    if ref_dec is None or syn_dec is None or trn_dec is None or pop_bin is None:
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

    # Call evaluation function
    try:
        results = evaluate_hallucination_metrics(
            population=pop_bin,
            training=trn_dec,
            reference=ref_dec,
            synthetic=syn_dec,
            metadata=metadata,
            query_file=query_file_path,
            experiment_name=parsed['experiment_name']
        )

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


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics on experiment folders post-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--folder',
        type=Path,
        help='Experiment folder path'
    )

    parser.add_argument(
        '--folders-pattern',
        type=str,
        help='Glob pattern to match multiple folders (e.g., "./mimic_*dseed233*/")'
    )

    parser.add_argument(
        '--generation',
        type=int,
        help='Specific generation to compute (default: all)'
    )

    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['statistical', 'detection', 'hallucination', 'all'],
        default=['all'],
        help='Which metrics to compute (default: all)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Custom params.yaml config file (optional)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing metrics'
    )

    parser.add_argument(
        '--metadata',
        type=Path,
        help='Path to metadata.json file (auto-discovered if not specified)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.folder and not args.folders_pattern:
        parser.error("Either --folder or --folders-pattern must be specified")

    # Collect folders to process
    folders = []
    if args.folder:
        if not args.folder.exists():
            console.print(f"[red]Error: Folder not found: {args.folder}[/red]")
            sys.exit(1)
        folders.append(args.folder)
    elif args.folders_pattern:
        folders = list(Path().glob(args.folders_pattern))
        if not folders:
            console.print(f"[red]Error: No folders matched pattern: {args.folders_pattern}[/red]")
            sys.exit(1)

    # Load config if provided
    config = None
    if args.config:
        if not args.config.exists():
            console.print(f"[red]Error: Config file not found: {args.config}[/red]")
            sys.exit(1)
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Determine which metrics to compute
    compute_all = 'all' in args.metrics
    compute_statistical = compute_all or 'statistical' in args.metrics
    compute_detection = compute_all or 'detection' in args.metrics
    compute_hallucination = compute_all or 'hallucination' in args.metrics

    # Process each folder
    console.print(f"\n[bold cyan]Post-Training Metrics Computation[/bold cyan]")
    console.print(f"Processing {len(folders)} folder(s)\n")

    for folder in folders:
        console.print(f"\n[bold]Processing folder: {folder}[/bold]")

        # Find metadata file
        metadata_path = args.metadata if args.metadata else find_metadata_file(folder)
        if not metadata_path:
            console.print(f"[red]Error: Could not find metadata.json for {folder}[/red]")
            console.print("[yellow]Specify with --metadata option[/yellow]")
            continue

        console.print(f"Using metadata: {metadata_path}")
        metadata = SingleTableMetadata.load_from_json(str(metadata_path))

        # Discover generations
        generations = discover_generation_files(folder, args.generation)
        if not generations:
            console.print(f"[yellow]Warning: No generations found in {folder}[/yellow]")
            continue

        console.print(f"Found {len(generations)} generation(s) to process")

        # Process each generation
        for gen_info in generations:
            model_id = gen_info['model_id']
            parsed = gen_info['parsed']
            gen_num = gen_info['generation']

            console.print(f"\n[bold yellow]Generation {gen_num}[/bold yellow] ({model_id})")

            # Load generation config
            gen_config = config if config else load_generation_config(folder, gen_num)

            # Compute metrics
            if compute_statistical:
                results = compute_statistical_metrics_post_training(
                    folder, model_id, parsed, metadata, metadata_path, gen_config, args.force
                )
                if results:
                    save_metrics(folder, model_id, "statistical_similarity", results)

            if compute_detection:
                results = compute_detection_metrics_post_training(
                    folder, model_id, parsed, metadata, metadata_path, gen_config, args.force
                )
                if results:
                    save_metrics(folder, model_id, "detection_evaluation", results)

            if compute_hallucination:
                results = compute_hallucination_metrics_post_training(
                    folder, model_id, parsed, metadata, metadata_path, gen_config, args.force
                )
                if results:
                    save_metrics(folder, model_id, "hallucination", results)

    console.print("\n[bold green]✓ Post-training metrics computation complete![/bold green]\n")


if __name__ == "__main__":
    main()
