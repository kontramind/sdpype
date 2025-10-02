"""
Chain Tracing Script for SDPype

Traces all generations in a recursive training chain given a model ID.
Extracts metrics from each generation for analysis.

Usage:
    python trace_chain.py sdv_gaussian_copula_0cf8e0f5_852ba944_852ba944_gen_0_9eadbd5d_51
    python trace_chain.py MODEL_ID --format csv
    python trace_chain.py MODEL_ID --format json > chain.json
    python trace_chain.py MODEL_ID --plot
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Trace generation chains in SDPype experiments")
console = Console()


def parse_model_id(model_id: str) -> Dict[str, str]:
    """
    Parse model_id into components for chain tracing.
    
    Format: library_model_refhash_roothash_trnhash_gen_N_cfghash_seed
    Example: sdv_gaussian_copula_0cf8e0f5_852ba944_852ba944_gen_0_9eadbd5d_51
    
    Args:
        model_id: Full model identifier string
        
    Returns:
        Dict with parsed components: root_hash, seed, generation, experiment_name
        
    Raises:
        ValueError: If model_id format is invalid
    """
    parts = model_id.split("_")
    
    # Validate minimum length: library_model_refhash_roothash_trnhash_gen_N_cfghash_seed
    if len(parts) < 9:
        raise ValueError(
            f"Invalid model_id format (expected at least 9 components): {model_id}\n"
            f"Expected format: library_model_refhash_roothash_trnhash_gen_N_cfghash_seed"
        )
    
    try:
        # Extract from the end
        seed = parts[-1]
        config_hash = parts[-2]
        generation_num = int(parts[-3])
        gen_marker = parts[-4]
        
        # Validate "gen" marker
        if gen_marker != "gen":
            raise ValueError(f"Expected 'gen' marker at position -4, got: {gen_marker}")
        
        # Extract root_hash (4th component, index 3)
        root_hash = parts[3]
        
        # Experiment name is everything except: _gen_N_cfghash_seed
        experiment_name = "_".join(parts[:-4])
        
        return {
            "root_hash": root_hash,
            "seed": seed,
            "generation": generation_num,
            "experiment_name": experiment_name,
            "config_hash": config_hash
        }
        
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse model_id '{model_id}': {e}") from e


def read_metrics(model_id: str, generation: int) -> Dict:
    """
    Read key metrics from generation's output files.
    
    Adapted from recursive_train.py to work with any model_id.
    
    Args:
        model_id: Full model identifier
        generation: Generation number to read metrics for
        
    Returns:
        Dict with metrics: α (Alpha Precision), PRDC (avg), Det (avg)
    """
    # Parse model_id to extract components
    parts = model_id.split("_")
    seed = parts[-1]
    config_hash = parts[-2]
    experiment_name = "_".join(parts[:-4])
    
    metrics = {}
    
    # Statistical similarity metrics
    stat_file = Path(
        f"experiments/metrics/statistical_similarity_{experiment_name}_gen_{generation}_{config_hash}_{seed}.json"
    )
    
    if stat_file.exists():
        with stat_file.open() as f:
            data = json.load(f)
            metrics_data = data.get('metrics', {})
            
            # Alpha precision - use authenticity_OC score
            if 'alpha_precision' in metrics_data:
                ap = metrics_data['alpha_precision']
                if ap.get('status') == 'success':
                    scores = ap.get('scores', {})
                    metrics['alpha_precision'] = scores.get('authenticity_OC', 0)
            
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
                    metrics['prdc_avg'] = prdc_avg
                    
                    # Also store individual components
                    metrics['prdc_precision'] = prdc_metric.get('precision', 0)
                    metrics['prdc_recall'] = prdc_metric.get('recall', 0)
                    metrics['prdc_density'] = prdc_metric.get('density', 0)
                    metrics['prdc_coverage'] = prdc_metric.get('coverage', 0)
    
    # Detection metrics
    det_file = Path(
        f"experiments/metrics/detection_evaluation_{experiment_name}_gen_{generation}_{config_hash}_{seed}.json"
    )
    
    if det_file.exists():
        with det_file.open() as f:
            data = json.load(f)
            individual_scores = data.get('individual_scores', {})
            
            # Average detection score across methods
            if individual_scores:
                det_scores = []
                for method_name, method_data in individual_scores.items():
                    if method_data.get('status') == 'success' and 'auc_score' in method_data:
                        det_scores.append(method_data['auc_score'])
                
                if det_scores:
                    metrics['detection_avg'] = sum(det_scores) / len(det_scores)
    
    return metrics


def trace_chain(model_id: str, max_generations: int = 100) -> List[Dict]:
    """
    Trace all generations in a recursive training chain.
    
    Strategy:
    1. Parse input model_id to get root_hash and seed
    2. For each generation (0 to max), glob for metrics files
    3. Extract model_id from metrics filename
    4. Read metrics for that generation
    5. Stop when no more generations found
    
    Args:
        model_id: Any model_id from the chain
        max_generations: Maximum generations to search for (default: 100)
        
    Returns:
        List of dicts with generation data, sorted by generation number
    """
    # Parse input model_id
    try:
        parsed = parse_model_id(model_id)
        root_hash = parsed['root_hash']
        seed = parsed['seed']
    except ValueError as e:
        console.print(f"[red]Error parsing model_id: {e}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Tracing chain:[/cyan] root_hash={root_hash}, seed={seed}")
    
    results = []
    
    # Search for each generation
    for gen in range(max_generations):
        # Look for statistical_similarity metrics (most reliable indicator)
        # Pattern: statistical_similarity_*_{root_hash}_*_gen_{gen}_*_{seed}.json
        pattern = f"experiments/metrics/statistical_similarity_*_{root_hash}_*_gen_{gen}_*_{seed}.json"
        metric_files = list(Path().glob(pattern))
        
        if not metric_files:
            # No more generations found
            break
        
        # Extract full model_id from metrics filename
        # Filename format: statistical_similarity_{model_id}.json
        filename = metric_files[0].stem
        model_id_for_gen = filename.replace("statistical_similarity_", "")
        
        # Read metrics
        metrics = read_metrics(model_id_for_gen, gen)
        
        # Get model file info if exists
        model_file = Path(f"experiments/models/sdg_model_{model_id_for_gen}.pkl")
        model_exists = model_file.exists()
        model_size_mb = model_file.stat().st_size / (1024 * 1024) if model_exists else 0
        
        results.append({
            'generation': gen,
            'model_id': model_id_for_gen,
            'metrics': metrics,
            'model_exists': model_exists,
            'model_size_mb': model_size_mb
        })
    
    console.print(f"[green]Found {len(results)} generations[/green]")
    
    return results


def display_chain_table(results: List[Dict]):
    """Display chain results as a rich table"""
    
    if not results:
        console.print("[yellow]No generations found in chain[/yellow]")
        return
    
    table = Table(title="Generation Chain Metrics", show_header=True, header_style="bold blue")
    table.add_column("Gen", justify="right", style="cyan")
    table.add_column("Alpha Precision", justify="right")
    table.add_column("PRDC Avg", justify="right")
    table.add_column("Detection Avg", justify="right")
    table.add_column("Model Size", justify="right", style="dim")
    table.add_column("Status", justify="center")
    
    for r in results:
        gen = r['generation']
        metrics = r['metrics']
        
        # Format metrics with fallbacks
        alpha = f"{metrics.get('alpha_precision', 0):.3f}" if 'alpha_precision' in metrics else "—"
        prdc = f"{metrics.get('prdc_avg', 0):.3f}" if 'prdc_avg' in metrics else "—"
        det = f"{metrics.get('detection_avg', 0):.3f}" if 'detection_avg' in metrics else "—"
        
        # Model size
        size = f"{r['model_size_mb']:.1f} MB" if r['model_exists'] else "—"
        
        # Status indicator
        status = "✓" if r['model_exists'] else "⚠"
        status_style = "green" if r['model_exists'] else "yellow"
        
        table.add_row(
            str(gen),
            alpha,
            prdc,
            det,
            size,
            f"[{status_style}]{status}[/{status_style}]"
        )
    
    console.print()
    console.print(table)
    console.print()
    console.print(f"[dim]Total generations: {len(results)}[/dim]")
    console.print(f"[dim]Root hash: {parse_model_id(results[0]['model_id'])['root_hash']}[/dim]")
    console.print(f"[dim]Seed: {parse_model_id(results[0]['model_id'])['seed']}[/dim]")


def export_csv(results: List[Dict]) -> str:
    """
    Export chain results as CSV format.
    
    Returns:
        CSV string with headers and data
    """
    if not results:
        return "generation,model_id,alpha_precision,prdc_avg,detection_avg,model_size_mb\n"
    
    lines = ["generation,model_id,alpha_precision,prdc_avg,prdc_precision,prdc_recall,prdc_density,prdc_coverage,detection_avg,model_size_mb"]
    
    for r in results:
        gen = r['generation']
        model_id = r['model_id']
        metrics = r['metrics']
        
        alpha = metrics.get('alpha_precision', '')
        prdc_avg = metrics.get('prdc_avg', '')
        prdc_p = metrics.get('prdc_precision', '')
        prdc_r = metrics.get('prdc_recall', '')
        prdc_d = metrics.get('prdc_density', '')
        prdc_c = metrics.get('prdc_coverage', '')
        det = metrics.get('detection_avg', '')
        size = r['model_size_mb'] if r['model_exists'] else ''
        
        lines.append(f"{gen},{model_id},{alpha},{prdc_avg},{prdc_p},{prdc_r},{prdc_d},{prdc_c},{det},{size}")
    
    return "\n".join(lines)


def export_json(results: List[Dict]) -> str:
    """
    Export chain results as JSON format.
    
    Returns:
        JSON string (pretty-printed)
    """
    # Clean up results for JSON export
    export_data = {
        "chain_summary": {
            "total_generations": len(results),
            "root_hash": parse_model_id(results[0]['model_id'])['root_hash'] if results else None,
            "seed": parse_model_id(results[0]['model_id'])['seed'] if results else None
        },
        "generations": [
            {
                "generation": r['generation'],
                "model_id": r['model_id'],
                "metrics": r['metrics'],
                "model_exists": r['model_exists'],
                "model_size_mb": round(r['model_size_mb'], 2) if r['model_exists'] else None
            }
            for r in results
        ]
    }
    
    return json.dumps(export_data, indent=2)


def plot_chain(results: List[Dict], output_file: Optional[str] = None):
    """
    Plot metrics degradation across generations.
    
    Args:
        results: Chain results from trace_chain()
        output_file: Optional filename to save plot (e.g., 'chain.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[red]matplotlib not installed. Install with: pip install matplotlib[/red]")
        raise typer.Exit(1)
    
    if not results:
        console.print("[yellow]No data to plot[/yellow]")
        return
    
    generations = [r['generation'] for r in results]
    
    # Extract metrics
    alpha = [r['metrics'].get('alpha_precision', None) for r in results]
    prdc = [r['metrics'].get('prdc_avg', None) for r in results]
    det = [r['metrics'].get('detection_avg', None) for r in results]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric if available
    if any(x is not None for x in alpha):
        ax.plot(generations, alpha, marker='o', label='Alpha Precision', linewidth=2)
    
    if any(x is not None for x in prdc):
        ax.plot(generations, prdc, marker='s', label='PRDC Avg', linewidth=2)
    
    if any(x is not None for x in det):
        ax.plot(generations, det, marker='^', label='Detection Avg', linewidth=2)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metric Degradation Across Generations', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(0, 1.05)
    
    # Add annotations for key points
    if alpha and alpha[0] is not None and alpha[-1] is not None:
        degradation = alpha[0] - alpha[-1]
        ax.annotate(
            f'Δ = {degradation:.3f}',
            xy=(generations[-1], alpha[-1]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            alpha=0.7
        )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        console.print(f"[green]Plot saved to: {output_file}[/green]")
    else:
        plt.show()
    
    plt.close()


@app.command()
def main(
    model_id: str = typer.Argument(..., help="Model ID from any generation in the chain"),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format: table (default), csv, json"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (for csv/json export)"
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        "-p",
        help="Generate matplotlib plot of metrics"
    ),
    plot_output: Optional[str] = typer.Option(
        None,
        "--plot-output",
        help="Save plot to file (e.g., chain.png)"
    ),
    max_generations: int = typer.Option(
        100,
        "--max-generations",
        help="Maximum generations to search"
    )
):
    """
    Trace all generations in a recursive training chain.
    
    Examples:
        trace_chain.py MODEL_ID
        trace_chain.py MODEL_ID --format csv --output chain.csv
        trace_chain.py MODEL_ID --format json > chain.json
        trace_chain.py MODEL_ID --plot --plot-output degradation.png
    """
    
    # Trace the chain
    try:
        results = trace_chain(model_id, max_generations)
    except Exception as e:
        console.print(f"[red]Error tracing chain: {e}[/red]")
        raise typer.Exit(1)
    
    if not results:
        console.print("[yellow]No generations found in chain[/yellow]")
        raise typer.Exit(0)
    
    # Handle output format
    if format == "csv":
        csv_output = export_csv(results)
        if output:
            Path(output).write_text(csv_output)
            console.print(f"[green]CSV exported to: {output}[/green]")
        else:
            print(csv_output)
    
    elif format == "json":
        json_output = export_json(results)
        if output:
            Path(output).write_text(json_output)
            console.print(f"[green]JSON exported to: {output}[/green]")
        else:
            print(json_output)
    
    else:
        # Default: table display
        display_chain_table(results)
    
    # Generate plot if requested
    if plot:
        plot_chain(results, plot_output)


if __name__ == "__main__":
    app()