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
from typing import Dict, Any, List, Optional, Tuple
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

console = Console()
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
    population_file: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all necessary decoded data files for a generation.

    Args:
        folder: Experiment folder path
        model_id: Model identifier
        metadata_path: Path to metadata file
        population_file: Optional path to population data file (decoded)

    Returns:
        (reference_decoded, synthetic_decoded, training_decoded, population_decoded)
    """
    data_root = folder / "data"

    # Construct file paths for decoded data
    reference_decoded = data_root / "decoded" / f"reference_{model_id}.csv"
    synthetic_decoded = data_root / "synthetic" / f"synthetic_data_{model_id}_decoded.csv"
    training_decoded = data_root / "decoded" / f"training_{model_id}.csv"

    # Load decoded data
    ref_dec = load_csv_with_metadata(reference_decoded, metadata_path) if reference_decoded.exists() else None
    syn_dec = load_csv_with_metadata(synthetic_decoded, metadata_path) if synthetic_decoded.exists() else None
    trn_dec = load_csv_with_metadata(training_decoded, metadata_path) if training_decoded.exists() else None

    # Load population data (decoded) from provided path or try to find it
    pop_dec = None
    if population_file and population_file.exists():
        console.print(f"[dim]Loading population data: {population_file}[/dim]")
        pop_dec = load_csv_with_metadata(population_file, metadata_path, low_memory=False)
    elif population_file:
        console.print(f"[yellow]Warning: Population file not found: {population_file}[/yellow]")

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

    # Load decoded data (population not needed for statistical metrics)
    ref_dec, syn_dec, trn_dec, _ = load_generation_data(folder, model_id, metadata_path, population_file=None)

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

    # Load encoder and encode data if needed for encoded metrics
    ref_enc = None
    syn_enc = None
    needs_encoding = any(m.get('name') in ENCODED_METRICS for m in config.get('evaluation', {}).get('statistical', {}).get('metrics', []))

    if needs_encoding:
        encoder = load_encoder(folder, model_id)
        if encoder:
            ref_enc = encode_data(encoder, ref_dec, "reference")
            syn_enc = encode_data(encoder, syn_dec, "synthetic")
        else:
            console.print("[yellow]Warning: Could not load encoder, encoded metrics may fail[/yellow]")

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

    # Load decoded data
    ref_dec, syn_dec, trn_dec, _ = load_generation_data(folder, model_id, metadata_path, population_file=None)

    if ref_dec is None or syn_dec is None:
        console.print("[red]Error: Missing required data files for detection metrics[/red]")
        return None

    # Load encoder and encode data (detection needs encoded data)
    encoder = load_encoder(folder, model_id)
    if not encoder:
        console.print("[red]Error: Could not load encoder for detection metrics[/red]")
        return None

    ref_enc = encode_data(encoder, ref_dec, "reference")
    syn_enc = encode_data(encoder, syn_dec, "synthetic")

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

    # Get population file path from config
    population_file = None
    if config and 'data' in config:
        pop_file_str = config['data'].get('population_file')
        if pop_file_str:
            population_file = Path(pop_file_str)
            # Resolve relative paths from experiment folder
            if not population_file.is_absolute():
                population_file = folder / population_file
                if not population_file.exists():
                    # Try relative to current directory
                    population_file = Path(pop_file_str)

    if not population_file or not population_file.exists():
        console.print("[red]Error: Population file not found. Please specify 'data.population_file' in config.[/red]")
        if population_file:
            console.print(f"[red]Tried: {population_file}[/red]")
        return None

    # Load data with population file
    ref_dec, syn_dec, trn_dec, pop_dec = load_generation_data(
        folder, model_id, metadata_path, population_file=population_file
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
    ] = None
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

        # If no config provided, try to load from checkpoints for metadata discovery
        effective_config = config_data
        if not effective_config and not metadata:
            # Try to load checkpoint config from generation 0 for metadata path
            effective_config = load_generation_config(exp_folder, generation if generation is not None else 0)

        # Find metadata file
        # Priority: --metadata flag > config['data']['metadata_file'] > auto-discovery
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
