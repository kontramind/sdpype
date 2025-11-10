#!/usr/bin/env python3
"""
Recursive Training Script for SDPype

Runs N generations of recursive training where each generation uses
the synthetic data from the previous generation as training input.

Usage:
    python recursive_train.py --generations 5
    python recursive_train.py --generations 10 --params custom_params.yaml
    python recursive_train.py --resume-from MODEL_ID --generations 10
    python recursive_train.py --resume  # Resume from last checkpoint
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Union


import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from ruamel.yaml import YAML

# Import parse_model_id from trace_chain for consistent parsing
from trace_chain import parse_model_id

app = typer.Typer(help="Recursive training for SDPype")
console = Console()


def validate_initial_params(params_file: Path) -> dict:
    """Validate params.yaml is ready for generation 0"""
    yaml = YAML()
    yaml.preserve_quotes = True
    
    with params_file.open() as f:
        params = yaml.load(f)
    
    generation = params.get('experiment', {}).get('generation', None)
    if generation is None:
        console.print("[red]Error: experiment.generation not found in params.yaml[/red]")
        raise typer.Exit(1)
    
    if generation != 0:
        console.print(f"[yellow]Warning: generation={generation}, expected 0 for initial run[/yellow]")
    
    return params


def run_pipeline() -> tuple[bool, float]:
    """Execute sdpype pipeline, return (success, elapsed_time)"""
    start = time.time()
    try:
        result = subprocess.run(
            ["uv", "run", "sdpype", "pipeline"],
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        console.print(f"[red]Pipeline failed with exit code {e.returncode}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr}[/dim]")
        return False, elapsed


def get_latest_model_id(generation: int) -> str:
    """
    Get model ID for a specific generation from the metrics files just created.

    This is more reliable than using filesystem modification times.

    Args:
        generation: The generation number that just completed

    Returns:
        Model ID string

    Raises:
        FileNotFoundError: If no metrics found for this generation
    """
    metrics_dir = Path("experiments/metrics")

    # Look for training metrics file (always created first)
    # Pattern: training_*_gen_{generation}_*.json
    pattern = f"training_*_gen_{generation}_*.json"
    metric_files = list(metrics_dir.glob(pattern))

    if not metric_files:
        raise FileNotFoundError(f"No training metrics found for generation {generation}")

    # Get most recent (in case multiple exist)
    latest_metric = max(metric_files, key=lambda p: p.stat().st_mtime)

    # Extract model_id from filename: training_{model_id}.json
    filename = latest_metric.stem
    model_id = filename.replace("training_", "")

    # Validate by checking if model file exists
    model_file = Path(f"experiments/models/sdg_model_{model_id}.pkl")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found for model_id: {model_id}")

    return model_id


def get_synthetic_data_path(model_id: str) -> Path:
    """Construct synthetic data path from model ID"""
    return Path(f"experiments/data/synthetic/synthetic_data_{model_id}_decoded.csv")


def read_metrics(model_id: str, generation: int) -> dict:
    """Read key metrics from generation's output"""

    # Extract components: last 2 parts are always config_hash and seed
    # Everything before that is the experiment_name (includes generation number)
    parts = model_id.split("_")
    seed = parts[-1]
    config_hash = parts[-2]
    experiment_name = "_".join(parts[:-2])

    stat_file = Path(f"experiments/metrics/statistical_similarity_{experiment_name}_{config_hash}_{seed}.json")
    det_file = Path(f"experiments/metrics/detection_evaluation_{experiment_name}_{config_hash}_{seed}.json")
    halluc_file = Path(f"experiments/metrics/hallucination_{experiment_name}_{config_hash}_{seed}.json")

    console.print(f"[dim]Looking for metrics:[/dim]")
    console.print(f"[dim]  Stat: {stat_file} (exists: {stat_file.exists()})[/dim]")
    console.print(f"[dim]  Det: {det_file} (exists: {det_file.exists()})[/dim]")
    console.print(f"[dim]  Halluc: {halluc_file} (exists: {halluc_file.exists()})[/dim]")
   
    metrics = {}

    # Statistical similarity metrics
    if stat_file.exists():
        with stat_file.open() as f:
            data = json.load(f)
            metrics_data = data.get('metrics', {})
            
            # Alpha precision - use authenticity_OC score
            if 'alpha_precision' in metrics_data:
                ap = metrics_data['alpha_precision']
                if ap.get('status') == 'success':
                    scores = ap.get('scores', {})
                    metrics['α'] = scores.get('authenticity_OC', 0)

            # PRDC - average of precision, recall, density, coverage
            if 'prdc_score' in metrics_data:
               prdc_metric = metrics_data['prdc_score']
               if prdc_metric.get('status') == 'success':
                    prdc_avg = sum([
                        prdc_metric.get('precision', 0),
                        prdc_metric.get('recall', 0),
                        prdc_metric.get('density', 0),
                        prdc_metric.get('coverage', 0)
                    ]) / 4
                    metrics['PRDC'] = prdc_avg
            
            # Wasserstein Distance - lower is better
            if 'wasserstein_distance' in metrics_data:
                wd_metric = metrics_data['wasserstein_distance']
                if wd_metric.get('status') == 'success':
                    # Store distance (lower = more similar distributions)
                    metrics['WD'] = wd_metric.get('joint_distance', 1.0)

            # Maximum Mean Discrepancy - lower is better
            if 'maximum_mean_discrepancy' in metrics_data:
                mmd_metric = metrics_data['maximum_mean_discrepancy']
                if mmd_metric.get('status') == 'success':
                    # Store distance (lower = more similar distributions)
                    metrics['MMD'] = mmd_metric.get('joint_distance', 1.0)

            # Jensen-Shannon (Synthcity) - higher is better (similarity score)
            if 'jensenshannon_synthcity' in metrics_data:
                jsd_sc_metric = metrics_data['jensenshannon_synthcity']
                if jsd_sc_metric.get('status') == 'success':
                    # Store distance score (lower = more similar distributions)
                    metrics['JSD_SC'] = jsd_sc_metric.get('distance_score', 1.0)

            # Jensen-Shannon (SYNDAT) - higher is better (similarity score)
            if 'jensenshannon_syndat' in metrics_data:
                jsd_sd_metric = metrics_data['jensenshannon_syndat']
                if jsd_sd_metric.get('status') == 'success':
                    # Store distance score (lower = more similar distributions)
                    metrics['JSD_SD'] = jsd_sd_metric.get('distance_score', 1.0)

            # Jensen-Shannon (NannyML) - higher is better (similarity score)
            if 'jensenshannon_nannyml' in metrics_data:
                jsd_nm_metric = metrics_data['jensenshannon_nannyml']
                if jsd_nm_metric.get('status') == 'success':
                    # Store distance score (lower = more similar distributions)
                    metrics['JSD_NM'] = jsd_nm_metric.get('distance_score', 1.0)

    # Detection metrics
    if det_file.exists():
        with det_file.open() as f:
            data = json.load(f)
            individual_scores = data.get('individual_scores', {})

            # Average detection score (lower is better - harder to detect synthetic)
            # Average AUC score across detection methods
            if individual_scores:
                det_scores = []
                for method_name, method_data in individual_scores.items():
                    if method_data.get('status') == 'success' and 'auc_score' in method_data:
                        det_scores.append(method_data['auc_score'])

                if det_scores:
                    metrics['Det'] = sum(det_scores) / len(det_scores)

    # Hallucination metrics (DDR + Plausibility)
    if halluc_file.exists():
        with halluc_file.open() as f:
            data = json.load(f)

            # Factual (Total) - records in population
            quality_matrix = data.get('quality_matrix_2x2', {})
            if 'total_factual' in quality_matrix:
                metrics['Factual'] = quality_matrix['total_factual'].get('rate_pct', 0) / 100

            # Novel Factual (DDR) - factual AND not memorized (THE IDEAL!)
            if 'novel_factual' in quality_matrix:
                metrics['DDR'] = quality_matrix['novel_factual'].get('rate_pct', 0) / 100

            # Plausible (Total) - passes validation rules
            if 'total_plausible' in quality_matrix:
                metrics['Plausible'] = quality_matrix['total_plausible'].get('rate_pct', 0) / 100

            # Novel Plausible - plausible AND not memorized
            if 'novel_plausible' in quality_matrix:
                metrics['NovelPlaus'] = quality_matrix['novel_plausible'].get('rate_pct', 0) / 100

    return metrics


def backup_params(params_file: Path, generation: int, checkpoint_dir: Path, model_id: str):
    """
    Backup params.yaml for this generation and chain.
    
    Params are backed up with chain_id to prevent different chains from overwriting each other.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Include chain_id in filename to prevent overwriting between chains
    chain_id = extract_chain_identifier(model_id)
    backup_file = checkpoint_dir / f"params_{chain_id}_gen_{generation}.yaml"

    import shutil
    shutil.copy2(params_file, backup_file)


def update_params_for_next_generation(params_file: Path, synthetic_data_path: Path, next_gen: int):
    """Update params.yaml with new training file and generation number"""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=0)
    
    with params_file.open() as f:
        params = yaml.load(f)
    
    # Update training file to synthetic data from current generation
    if 'data' not in params:
        params['data'] = {}
    params['data']['training_file'] = str(synthetic_data_path)
    
    # Increment generation
    if 'experiment' not in params:
        params['experiment'] = {}
    params['experiment']['generation'] = next_gen
    
    # Write back
    with params_file.open('w') as f:
        yaml.dump(params, f)


def extract_chain_identifier(model_id: str) -> str:
    """
    Extract unique chain identifier from model_id.

    Returns: library_modeltype_roothash_seed
    This uniquely identifies a chain across all invariants.
    """
    parts = model_id.split("_")

    # Fixed positions - simple extraction
    library = parts[0]
    model_type = parts[1]
    root_hash = parts[3]
    seed = parts[8]

    return f"{library}_{model_type}_{root_hash}_{seed}"


def save_checkpoint(checkpoint_dir: Path, model_id: str, generation: int, elapsed: float):
    """
    Save chain-specific checkpoint for resume capability.

    Checkpoint filename format: checkpoint_{library}_{modeltype}_{roothash}_{seed}.json
    This allows multiple chains to coexist without conflicts.
    """
    chain_id = extract_chain_identifier(model_id)
    checkpoint_file = checkpoint_dir / f"checkpoint_{chain_id}.json"

    parsed = parse_model_id(model_id)
    checkpoint_data = {
        "chain_id": chain_id,
        "root_hash": parsed['root_hash'],
        "seed": parsed['seed'],
        "last_completed_generation": generation,
        "last_model_id": model_id,
        "elapsed_time": elapsed,        
        "timestamp": datetime.now().isoformat()
    }
    
    with checkpoint_file.open('w') as f:
        json.dump(checkpoint_data, f, indent=2)


def save_generation_timing(checkpoint_dir: Path, generation: int, elapsed: float):
    """Save timing for a specific generation"""
    timing_file = checkpoint_dir / "timings.json"

    # Load existing timings or create new
    timings = {}
    if timing_file.exists():
        with timing_file.open() as f:
            timings = json.load(f)

    timings[str(generation)] = elapsed

    with timing_file.open('w') as f:
        json.dump(timings, f, indent=2)


def load_checkpoint_for_chain(checkpoint_dir: Path, model_id: str) -> dict:
    """
    Load checkpoint for a specific chain identified by model_id.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_id: Any model_id from the chain to resume

    Returns:
        Checkpoint dict or None if not found
    """
    chain_id = extract_chain_identifier(model_id)
    checkpoint_file = checkpoint_dir / f"checkpoint_{chain_id}.json"

    if checkpoint_file.exists():
        with checkpoint_file.open() as f:
            return json.load(f)
    return None


def find_latest_checkpoint(checkpoint_dir: Path) -> dict:
    """
    Find the most recent checkpoint when no specific chain is given.

    Returns:
        Most recent checkpoint dict or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    # Find all chain-specific checkpoints
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))

    if not checkpoint_files:
        return None

    if len(checkpoint_files) > 1:
        console.print(f"[yellow]Found {len(checkpoint_files)} checkpoints. Please specify which chain to resume with --resume-from MODEL_ID[/yellow]")
        console.print("[yellow]Available chains:[/yellow]")
        for cp_file in checkpoint_files:
            with cp_file.open() as f:
                cp_data = json.load(f)
                console.print(f"  • {cp_data.get('chain_id', 'unknown')}, gen={cp_data['last_completed_generation']}")
        return None

    # Single checkpoint found - use it
    checkpoint_file = checkpoint_files[0]
    with checkpoint_file.open() as f:
        checkpoint = json.load(f)

    console.print(f"[cyan]Found checkpoint: root_hash={checkpoint['root_hash']}, seed={checkpoint['seed']}[/cyan]")
    return checkpoint


def reconstruct_checkpoint_from_chain(model_id: str) -> dict:
    """
    Reconstruct checkpoint by tracing chain from metrics files.
    
    Fallback when checkpoint file doesn't exist but generations have been completed.

    Args:
        model_id: Any model_id from the chain

    Returns:
        Reconstructed checkpoint dict

    Raises:
        ValueError: If no generations found for the chain
    """
    parsed = parse_model_id(model_id)
    root_hash = parsed['root_hash']
    seed = parsed['seed']
    library = parsed['library']
    model_type = parsed['model_type']
    ref_hash = parsed['ref_hash']    

    console.print(f"[yellow]No checkpoint found, reconstructing from metrics...[/yellow]")

    # Find last completed generation
    last_gen = -1
    last_model_id = None

    for gen in range(100):  # Max search
        pattern = f"experiments/metrics/statistical_similarity_*_gen_{gen}_*_{seed}.json"
        metric_files = list(Path().glob(pattern))

        # Filter by all chain invariants
        matching_model_id = None
        for metric_file in metric_files:
            filename = metric_file.stem
            candidate_model_id = filename.replace("statistical_similarity_", "")

            try:
                parsed_candidate = parse_model_id(candidate_model_id)

                if (parsed_candidate['library'] == library and
                    parsed_candidate['model_type'] == model_type and
                    parsed_candidate['ref_hash'] == ref_hash and
                    parsed_candidate['root_hash'] == root_hash and
                    parsed_candidate['seed'] == seed):
                    matching_model_id = candidate_model_id
                    break
            except ValueError:
                continue

        if matching_model_id:
            last_gen = gen
            last_model_id = matching_model_id
        else:
            break

    if last_gen == -1:
        raise ValueError(f"No generations found for chain: root_hash={root_hash}, seed={seed}")

    console.print(f"[green]Reconstructed: found {last_gen + 1} completed generations[/green]")

    return {
        "root_hash": root_hash,
        "seed": seed,
        "last_completed_generation": last_gen,
        "last_model_id": last_model_id,
        "elapsed_time": 0,  # Unknown
        "timestamp": datetime.now().isoformat()
    }


def load_all_generation_results(checkpoint_dir: Path, start_gen: int, end_gen: int, model_id: str) -> list:
    """
    Load results from all completed generations for a specific chain.

    Args:
        checkpoint_dir: Directory containing checkpoints
        start_gen: Starting generation
        end_gen: Ending generation (exclusive)
        model_id: Model ID to identify which chain to load

    Returns:
        List of generation results for the specified chain only
    """
    # Parse model_id to get chain invariants
    parsed = parse_model_id(model_id)
    library = parsed['library']
    model_type = parsed['model_type']
    ref_hash = parsed['ref_hash']
    root_hash = parsed['root_hash']
    seed = parsed['seed']

    results = []

    for gen in range(start_gen, end_gen):
        # Glob for generation + seed
        metrics_pattern = f"experiments/metrics/statistical_similarity_*_gen_{gen}_*_{seed}.json"
        metric_files = list(Path().glob(metrics_pattern))

        # Filter by all chain invariants
        matching_model_id = None
        for metric_file in metric_files:
            filename = metric_file.stem
            candidate_model_id = filename.replace("statistical_similarity_", "")

            try:
                parsed_candidate = parse_model_id(candidate_model_id)

                # All five invariants must match
                if (parsed_candidate['library'] == library and
                    parsed_candidate['model_type'] == model_type and
                    parsed_candidate['ref_hash'] == ref_hash and
                    parsed_candidate['root_hash'] == root_hash and
                    parsed_candidate['seed'] == seed):
                    matching_model_id = candidate_model_id
                    break
            except ValueError:
                continue
        if matching_model_id:
            elapsed = 0  # Will be loaded from timings file below

            metrics = read_metrics(matching_model_id, gen)
            results.append({
                'generation': gen,
                'model_id': matching_model_id,
                'metrics': metrics,
                'elapsed': elapsed
            })

    # Load timing data
    timing_file = checkpoint_dir / "timings.json"
    if timing_file.exists():
        with timing_file.open() as f:
            timings = json.load(f)
            for r in results:
                gen_str = str(r['generation'])
                if gen_str in timings:
                    r['elapsed'] = timings[gen_str]

    return results


@app.command()
def run(
    generations: int = typer.Option(5, "--generations", "-n", help="Number of generations to run"),
    params_file: Path = typer.Option("params.yaml", "--params", "-p", help="Path to params.yaml"),
    checkpoint_dir: Path = typer.Option("checkpoints", "--checkpoint-dir", help="Directory for checkpoints"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last checkpoint"),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Resume specific chain by MODEL_ID (e.g., sdv_gaussiancopula_..._gen_5_..._51)"
    ),    
):
    """
    Run recursive training for N generations.
    
    Use --resume to auto-resume the last checkpoint (if only one exists).
    Use --resume-from MODEL_ID to explicitly resume a specific chain.
    """
    
    # Check if resuming
    start_gen = 0

    # Handle --resume-from (explicit model_id)
    if resume_from:
        console.print(f"[cyan]Resuming from MODEL_ID: {resume_from}[/cyan]")
        
        try:
            # Try to load checkpoint for this chain
            checkpoint = load_checkpoint_for_chain(checkpoint_dir, resume_from)
            
            if not checkpoint:
                # No checkpoint - try to reconstruct from metrics
                checkpoint = reconstruct_checkpoint_from_chain(resume_from)
        
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        
        start_gen = checkpoint['last_completed_generation'] + 1
        console.print(f"[green]Resuming from generation {start_gen}[/green]")
        console.print(f"[dim]Last model: {checkpoint['last_model_id']}[/dim]")

        if start_gen >= generations:
            console.print(f"[yellow]Chain already completed {start_gen} generations (requested: {generations})[/yellow]")
            console.print(f"[yellow]Increase --generations to continue training[/yellow]")
            raise typer.Exit(0)
        
    # Handle --resume (auto-detect)
    elif resume:
        console.print("[cyan]Auto-resuming from last checkpoint...[/cyan]")
        checkpoint = find_latest_checkpoint(checkpoint_dir)
        
        if checkpoint:
            start_gen = checkpoint['last_completed_generation'] + 1
            console.print(f"[green]Resuming from generation {start_gen}[/green]")
            console.print(f"[dim]Last model: {checkpoint['last_model_id']}[/dim]")

            if start_gen >= generations:
                console.print(f"[yellow]Chain already completed {start_gen} generations (requested: {generations})[/yellow]")
                console.print(f"[yellow]Increase --generations to continue training[/yellow]")
                raise typer.Exit(0)
        else:
            console.print("[yellow]No checkpoint found, starting from generation 0[/yellow]")
            checkpoint = None
    else:
        checkpoint = None
    
    # Restore params if resuming
    if checkpoint:
        last_gen = checkpoint['last_completed_generation']
        chain_id = checkpoint.get('chain_id')
        last_params = checkpoint_dir / f"params_{chain_id}_gen_{last_gen}.yaml"
        console.print(f"[dim]Resuming chain: {checkpoint.get('chain_id', 'unknown')}[/dim]")

        if last_params.exists():
            import shutil
            shutil.copy2(last_params, params_file)
            console.print(f"[dim]Restored params from gen {last_gen}[/dim]")
                 
            # Now update for next generation
            last_model_id = checkpoint['last_model_id']
            synthetic_path = get_synthetic_data_path(last_model_id)
            update_params_for_next_generation(params_file, synthetic_path, start_gen)
            console.print(f"[dim]Updated params for generation {start_gen}[/dim]")
        else:
            console.print(f"[yellow]Warning: Params backup not found: {last_params.name}[/yellow]")
            console.print("[yellow]Continuing with current params.yaml[/yellow]")
    
    # Validate initial params if starting fresh
    if start_gen == 0:
        validate_initial_params(params_file)
    
    # Display header
    console.print(f"\nRecursive training: {generations} generations\n")
    
    # Track metrics across generations
    results = []
    overall_start = time.time()
    # Load previous generation results if resuming
    if start_gen > 0:
        results = load_all_generation_results(checkpoint_dir, 0, start_gen, checkpoint['last_model_id'])

    # Run generations
    for gen in range(start_gen, generations):
        gen_start = time.time()
        
        # Run pipeline with spinner
        with console.status(f"[bold green]Gen {gen}/{generations-1}...", spinner="dots"):
            success, elapsed = run_pipeline()
        
        if not success:
            console.print(f"[red]Generation {gen} failed[/red]")
            raise typer.Exit(1)
        
        # Get model ID
        model_id = get_latest_model_id(gen)
        synthetic_path = get_synthetic_data_path(model_id)
        
        # Verify synthetic data exists
        if not synthetic_path.exists():
            console.print(f"[red]Synthetic data not found: {synthetic_path}[/red]")
            raise typer.Exit(1)
        
        # Get file size
        size_mb = synthetic_path.stat().st_size / (1024 * 1024)
        
        # Read metrics
        metrics = read_metrics(model_id, gen)
        
        # Display progress line
        metrics_str = " ".join([
            f"DDR={metrics.get('DDR', 0):.3f}" if 'DDR' in metrics else "",
            f"Factual={metrics.get('Factual', 0):.3f}" if 'Factual' in metrics else "",
            f"Plausible={metrics.get('Plausible', 0):.3f}" if 'Plausible' in metrics else "",
            f"Det={metrics.get('Det', 0):.3f}" if 'Det' in metrics else "",
            f"α={metrics.get('α', 0):.3f}" if 'α' in metrics else "",
        ]).strip()
        
        console.print(f"Gen {gen}: [green]{'█' * 20}[/green] 100% | {elapsed:.0f}s | {metrics_str}")
        
        # Store results
        results.append({
            'generation': gen,
            'model_id': model_id,
            'elapsed': elapsed,
            'metrics': metrics
        })
        
        # Backup params
        backup_params(params_file, gen, checkpoint_dir, model_id)
        
        # Save checkpoint
        save_checkpoint(checkpoint_dir, model_id, gen, elapsed)

        # Save timing separately for easy lookup
        save_generation_timing(checkpoint_dir, gen, elapsed)

        # Update params for next generation (if not last)
        if gen < generations - 1:
            update_params_for_next_generation(params_file, synthetic_path, gen + 1)
    
    # Summary
    total_time = time.time() - overall_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    gens_completed = generations - start_gen
    console.print(f"\n[green]✓ Completed generations {start_gen}-{generations-1} in {minutes}m {seconds}s[/green]")
    if start_gen > 0:
        console.print(f"[dim](Total: {len(results)} generations including previous runs)[/dim]")
    console.print(f"Checkpoints: {checkpoint_dir}/")
    
    # Show metrics table
    if results:
        table = Table(title="Generation Metrics Summary")
        table.add_column("Gen", justify="right", style="cyan")
        table.add_column("DDR\n(Novel Factual)", justify="right", style="green")
        table.add_column("Factual\n(Total)", justify="right")
        table.add_column("Plausible\n(Total)", justify="right")
        table.add_column("Novel\nPlausible", justify="right")
        table.add_column("Detection\nAUC", justify="right")
        table.add_column("Alpha\nPrecision", justify="right")
        table.add_column("Wasserstein\nDist", justify="right")
        table.add_column("MMD", justify="right")
        table.add_column("Time", justify="right", style="dim")

        for r in results:
            metrics = r['metrics']
            table.add_row(
                str(r['generation']),
                f"{metrics.get('DDR', 0):.3f}" if 'DDR' in metrics else "—",
                f"{metrics.get('Factual', 0):.3f}" if 'Factual' in metrics else "—",
                f"{metrics.get('Plausible', 0):.3f}" if 'Plausible' in metrics else "—",
                f"{metrics.get('NovelPlaus', 0):.3f}" if 'NovelPlaus' in metrics else "—",
                f"{metrics.get('Det', 0):.3f}" if 'Det' in metrics else "—",
                f"{metrics.get('α', 0):.3f}" if 'α' in metrics else "—",
                f"{metrics.get('WD', 0):.6f}" if 'WD' in metrics else "—",
                f"{metrics.get('MMD', 0):.6f}" if 'MMD' in metrics else "—",
                f"{r['elapsed']:.0f}s" if r['elapsed'] > 0 else "—"
            )
        
        console.print()
        console.print(table)


if __name__ == "__main__":
    app()
