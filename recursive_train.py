#!/usr/bin/env python3
"""
Recursive Training Script for SDPype

Runs N generations of recursive training where each generation uses
the synthetic data from the previous generation as training input.

Usage:
    python recursive_train.py --generations 5
    python recursive_train.py --generations 10 --params custom_params.yaml
    python recursive_train.py --resume  # Resume from last checkpoint
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from ruamel.yaml import YAML

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


def get_latest_model_id() -> str:
    """Get model ID from most recently created model file"""
    models_dir = Path("experiments/models")
    if not models_dir.exists():
        raise FileNotFoundError("No models directory found")

    # Find all model files
    model_files = list(models_dir.glob("sdg_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No model files found")

    # Get most recent by modification time
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

    # Extract model ID from filename: sdg_model_{model_id}.pkl
    model_id = latest_model.stem.replace("sdg_model_", "")
    return model_id


def get_synthetic_data_path(model_id: str) -> Path:
    """Construct synthetic data path from model ID"""
    return Path(f"experiments/data/synthetic/synthetic_data_{model_id}.csv")


def read_metrics(model_id: str, generation: int) -> dict:
    """Read key metrics from generation's output"""

    # Debug: print what we're looking for
    parts = model_id.split("_")
    seed = parts[-1]
    config_hash = parts[-2]
    experiment_name = "_".join(parts[:-4])
    
    stat_file = Path(f"experiments/metrics/statistical_similarity_{experiment_name}_gen_{generation}_{config_hash}_{seed}.json")
    det_file = Path(f"experiments/metrics/detection_evaluation_{experiment_name}_gen_{generation}_{config_hash}_{seed}.json")
    
    console.print(f"[dim]Looking for metrics:[/dim]")
    console.print(f"[dim]  Stat: {stat_file} (exists: {stat_file.exists()})[/dim]")
    console.print(f"[dim]  Det: {det_file} (exists: {det_file.exists()})[/dim]")
    
    # New model_id format: library_model_refhash_roothash_trnhash_gen_N_cfghash_seed
    parts = model_id.split("_")

    # Extract components from end
    seed = parts[-1]
    config_hash = parts[-2]
    # parts[-3] is generation number
    # parts[-4] is "gen"

    # Experiment name is everything except: _gen_N_cfghash_seed
    experiment_name = "_".join(parts[:-4])
   
    metrics = {}
    
    # Statistical similarity metrics
    stat_file = Path(f"experiments/metrics/statistical_similarity_{experiment_name}_gen_{generation}_{config_hash}_{seed}.json")
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
    
    # Detection metrics
    det_file = Path(f"experiments/metrics/detection_evaluation_{experiment_name}_gen_{generation}_{config_hash}_{seed}.json")
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
    
    return metrics


def backup_params(params_file: Path, generation: int, checkpoint_dir: Path):
    """Backup params.yaml for this generation"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backup_file = checkpoint_dir / f"params_gen_{generation}.yaml"
    
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


def save_checkpoint(checkpoint_dir: Path, generation: int, model_id: str, elapsed: float):
    """Save checkpoint information for resume capability"""
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    checkpoint_data = {
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


def load_checkpoint(checkpoint_dir: Path) -> dict:
    """Load checkpoint to resume training"""
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    if not checkpoint_file.exists():
        return None
    
    with checkpoint_file.open() as f:
        return json.load(f)


def load_all_generation_results(checkpoint_dir: Path, start_gen: int, end_gen: int) -> list:
    """Load results from all completed generations in checkpoints"""
    results = []
    
    for gen in range(start_gen, end_gen):
        # Look specifically for statistical_similarity files (most reliable)
        metrics_pattern = f"experiments/metrics/statistical_similarity_*_gen_{gen}_*.json"
        metric_files = list(Path().glob(metrics_pattern))
        
        if metric_files:
            # Extract model_id from filename by removing prefix and suffix
            # Pattern: statistical_similarity_{model_id}.json
            filename = metric_files[0].stem

            # Remove "statistical_similarity_" prefix (22 chars + 1 underscore)
            if filename.startswith("statistical_similarity_"):
                model_id = filename[len("statistical_similarity_"):]
            else:
                console.print(f"[yellow]Warning: Unexpected filename format for gen {gen}[/yellow]")
                continue

            elapsed = 0  # Will be loaded from timings file below

            metrics = read_metrics(model_id, gen)
            results.append({
                'generation': gen,
                'model_id': model_id,
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
):
    """Run recursive training for N generations"""
    
    # Check if resuming
    start_gen = 0
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            start_gen = checkpoint['last_completed_generation'] + 1
            console.print(f"[yellow]Resuming from generation {start_gen}[/yellow]")
            if start_gen >= generations:
                console.print("[yellow]All generations already completed[/yellow]")
                return

            # Restore params from last checkpoint
            last_params = checkpoint_dir / f"params_gen_{checkpoint['last_completed_generation']}.yaml"
            if last_params.exists():
                import shutil
                shutil.copy2(last_params, params_file)
                console.print(f"[dim]Restored params from gen {checkpoint['last_completed_generation']}[/dim]")
                
                # Now update for next generation
                last_model_id = checkpoint['last_model_id']
                synthetic_path = get_synthetic_data_path(last_model_id)
                update_params_for_next_generation(params_file, synthetic_path, start_gen)
                console.print(f"[dim]Updated params for generation {start_gen}[/dim]")
        else:
            console.print("[yellow]No checkpoint found, starting from generation 0[/yellow]")
    
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
        results = load_all_generation_results(checkpoint_dir, 0, start_gen)

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
        model_id = get_latest_model_id()
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
            f"α={metrics.get('α', 0):.3f}" if 'α' in metrics else "",
            f"PRDC={metrics.get('PRDC', 0):.3f}" if 'PRDC' in metrics else "",
            f"Det={metrics.get('Det', 0):.3f}" if 'Det' in metrics else "",
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
        backup_params(params_file, gen, checkpoint_dir)
        
        # Save checkpoint
        save_checkpoint(checkpoint_dir, gen, model_id, elapsed)

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
        table.add_column("Alpha Precision", justify="right")
        table.add_column("PRDC Avg", justify="right")
        table.add_column("Detection Avg", justify="right")
        table.add_column("Time", justify="right", style="dim")
        
        for r in results:
            metrics = r['metrics']
            table.add_row(
                str(r['generation']),
                f"{metrics.get('α', 0):.3f}" if 'α' in metrics else "—",
                f"{metrics.get('PRDC', 0):.3f}" if 'PRDC' in metrics else "—",
                f"{metrics.get('Det', 0):.3f}" if 'Det' in metrics else "—",
                f"{r['elapsed']:.0f}s" if r['elapsed'] > 0 else "—"
            )
        
        console.print()
        console.print(table)


if __name__ == "__main__":
    app()
