"""
Chain Tracing Script for SDPype

Traces all generations in a recursive training chain given a model ID.
Extracts metrics from each generation for analysis.

Usage:
    python trace_chain.py sdv_gaussiancopula_0cf8e0f5_852ba944_852ba944_gen_0_9eadbd5d_51
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
    
    Format: library_modeltype_refhash_roothash_trnhash_gen_N_cfghash_seed
    Example: sdv_gaussiancopula_0cf8e0f5_852ba944_852ba944_gen_0_9eadbd5d_51
    
    Args:
        model_id: Full model identifier string
        
    Returns:
        Dict with parsed components: root_hash, seed, generation, experiment_name
        
    Raises:
        ValueError: If model_id format is invalid
    """
    parts = model_id.split("_")
    
    # Validate length: library_modeltype_refhash_roothash_trnhash_gen_N_cfghash_seed = 9 parts minimum
    if len(parts) < 9:
        raise ValueError(
            f"Invalid model_id format (expected at least 9 components): {model_id}\n"
            f"Expected format: library_modeltype_refhash_roothash_trnhash_gen_N_cfghash_seed"
        )
    
    try:
        # Fixed positions - no detection needed!
        library = parts[0]
        model_type = parts[1]
        ref_hash = parts[2]
        root_hash = parts[3]
        training_hash = parts[4]
        gen_marker = parts[5]
        generation_num = int(parts[6])
        config_hash = parts[7]
        seed = parts[8]
        
        # Validate "gen" marker
        if gen_marker != "gen":
            raise ValueError(f"Expected 'gen' marker at position 5, got: {gen_marker}")
        
        # Experiment name is everything except: _gen_N_cfghash_seed
        experiment_name = "_".join(parts[:5])

        return {
            "library": library,
            "model_type": model_type,
            "ref_hash": ref_hash,
            "root_hash": root_hash,
            "training_hash": training_hash,
            "generation": generation_num,
            "config_hash": config_hash,
            "seed": seed,
            "experiment_name": experiment_name,
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
    # Extract components using fixed positions
    parts = model_id.split("_")
    seed = parts[8]
    config_hash = parts[7]
    experiment_name = "_".join(parts[:5])  # library_model_ref_root_trn
    
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

        # Wasserstein Distance - lower is better
        if 'wasserstein_distance' in metrics_data:
            wd_metric = metrics_data['wasserstein_distance']
            if wd_metric.get('status') == 'success':
                # Store distance (lower = more similar distributions)
                metrics['wasserstein_dist'] = wd_metric.get('joint_distance', 1.0)

        # Maximum Mean Discrepancy - lower is better
        if 'maximum_mean_discrepancy' in metrics_data:
            mmd_metric = metrics_data['maximum_mean_discrepancy']
            if mmd_metric.get('status') == 'success':
                # Store distance (lower = more similar distributions)
                metrics['mmd'] = mmd_metric.get('joint_distance', 1.0)

        # Jensen-Shannon (Synthcity) - higher is better (similarity score)
        if 'jensenshannon_synthcity' in metrics_data:
            jsd_sc_metric = metrics_data['jensenshannon_synthcity']
            if jsd_sc_metric.get('status') == 'success':
                # Store distance score (lower = more similar distributions)
                metrics['jsd_synthcity'] = jsd_sc_metric.get('distance_score', 1.0)
        
        # Jensen-Shannon (SYNDAT) - higher is better (similarity score)
        if 'jensenshannon_syndat' in metrics_data:
            jsd_sd_metric = metrics_data['jensenshannon_syndat']
            if jsd_sd_metric.get('status') == 'success':
                # Store distance score (lower = more similar distributions)
                metrics['jsd_syndat'] = jsd_sd_metric.get('distance_score', 1.0)

        # Jensen-Shannon (NannyML) - higher is better (similarity score)
        if 'jensenshannon_nannyml' in metrics_data:
            jsd_nm_metric = metrics_data['jensenshannon_nannyml']
            if jsd_nm_metric.get('status') == 'success':
                # Store distance score (lower = more similar distributions)
                metrics['jsd_nannyml'] = jsd_nm_metric.get('distance_score', 1.0)

            if 'tv_complement' in metrics_data:
                tv_metric = metrics_data['tv_complement']
                if tv_metric.get('status') == 'success':
                    # Get aggregate score (None if no compatible columns)
                    tv_score = tv_metric.get('aggregate_score')
                    if tv_score is not None:
                        metrics['tv_complement'] = tv_score

            if 'ks_complement' in metrics_data:
                ks_metric = metrics_data['ks_complement']
                if ks_metric.get('status') == 'success':
                    # Get aggregate score (None if no compatible columns)
                    ks_score = ks_metric.get('aggregate_score')
                    if ks_score is not None:
                        metrics['ks_complement'] = ks_score

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
        library = parsed['library']
        model_type = parsed['model_type']
        ref_hash = parsed['ref_hash']        
        root_hash = parsed['root_hash']
        seed = parsed['seed']
    except ValueError as e:
        console.print(f"[red]Error parsing model_id: {e}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Tracing chain:[/cyan] {library}/{model_type}, ref_hash={ref_hash}, root_hash={root_hash}, seed={seed}")
    
    results = []
    
    # Search for each generation
    for gen in range(max_generations):
        # Glob broadly for generation + seed, then filter by root_hash
        # This is more reliable because training_hash changes each generation
        pattern = f"experiments/metrics/statistical_similarity_*_gen_{gen}_*_{seed}.json"
        metric_files = list(Path().glob(pattern))
        
        if not metric_files:
            # No more generations found
            break
        
        # Filter by all five invariants: library, model_type, ref_hash, root_hash, seed
        matching_model_id = None
        for metric_file in metric_files:
            filename = metric_file.stem
            candidate_model_id = filename.replace("statistical_similarity_", "")
            
            # Parse and check if all invariants match
            try:
                parsed_candidate = parse_model_id(candidate_model_id)

                # All five invariants must match for same chain
                if (parsed_candidate['library'] == library and
                    parsed_candidate['model_type'] == model_type and
                    parsed_candidate['ref_hash'] == ref_hash and
                    parsed_candidate['root_hash'] == root_hash and
                    parsed_candidate['seed'] == seed):
                    matching_model_id = candidate_model_id
                    break
            except ValueError:
                continue
        
        if not matching_model_id:
            break
        
        model_id_for_gen = matching_model_id

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
    table.add_column("TV Complement", justify="right")
    table.add_column("KS Complement", justify="right")
    table.add_column("Wasserstein Dist", justify="right")
    table.add_column("MMD", justify="right")
    table.add_column("JS Sim (Synthcity)", justify="right")
    table.add_column("JS Sim (SYNDAT)", justify="right")
    table.add_column("JS Sim (NannyML)", justify="right")
    table.add_column("Detection Avg", justify="right")
    table.add_column("Model Size", justify="right", style="dim")
    table.add_column("Status", justify="center")
    
    for r in results:
        gen = r['generation']
        metrics = r['metrics']
        
        # Format metrics with fallbacks
        alpha = f"{metrics.get('alpha_precision', 0):.3f}" if 'alpha_precision' in metrics else "—"
        prdc = f"{metrics.get('prdc_avg', 0):.3f}" if 'prdc_avg' in metrics else "—"
        tv_comp = f"{metrics.get('tv_complement', 0):.3f}" if 'tv_complement' in metrics else "—"
        ks_comp = f"{metrics.get('ks_complement', 0):.3f}" if 'ks_complement' in metrics else "—"
        wd = f"{metrics.get('wasserstein_dist', 0):.6f}" if 'wasserstein_dist' in metrics else "—"
        mmd_val = f"{metrics.get('mmd', 0):.6f}" if 'mmd' in metrics else "—"
        jsd_sc = f"{metrics.get('jsd_synthcity', 0):.3f}" if 'jsd_synthcity' in metrics else "—"
        jsd_sd = f"{metrics.get('jsd_syndat', 0):.3f}" if 'jsd_syndat' in metrics else "—"
        jsd_nm = f"{metrics.get('jsd_nannyml', 0):.3f}" if 'jsd_nannyml' in metrics else "—"
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
            tv_comp,
            ks_comp,
            wd,
            mmd_val,
            jsd_sc,
            jsd_sd,
            jsd_nm,
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
        return "generation,model_id,alpha_precision,prdc_avg,prdc_precision,prdc_recall,prdc_density,prdc_coverage,tv_complement,ks_complement,wasserstein_dist,mmd,jsd_synthcity,jsd_syndat,jsd_nannyml,detection_avg,model_size_mb\n"
    
    lines = ["generation,model_id,alpha_precision,prdc_avg,prdc_precision,prdc_recall,prdc_density,prdc_coverage,tv_complement,ks_complement,wasserstein_dist,mmd,jsd_synthcity,jsd_syndat,jsd_nannyml,detection_avg,model_size_mb"]
    
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
        tv = metrics.get('tv_complement', '')
        ks = metrics.get('ks_complement', '')
        wd = metrics.get('wasserstein_dist', '')
        mmd_val = metrics.get('mmd', '')
        jsd_sc = metrics.get('jsd_synthcity', '')
        jsd_sd = metrics.get('jsd_syndat', '')
        jsd_nm = metrics.get('jsd_nannyml', '')
        det = metrics.get('detection_avg', '')
        size = r['model_size_mb'] if r['model_exists'] else ''

        lines.append(f"{gen},{model_id},{alpha},{prdc_avg},{prdc_p},{prdc_r},{prdc_d},{prdc_c},{tv},{ks},{wd},{mmd_val},{jsd_sc},{jsd_sd},{jsd_nm},{det},{size}")

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


def plot_chain_static(results: List[Dict], output_file: Optional[str] = None):
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
    tv = [r['metrics'].get('tv_complement', None) for r in results]
    ks = [r['metrics'].get('ks_complement', None) for r in results]
    wd = [r['metrics'].get('wasserstein_dist', None) for r in results]
    mmd = [r['metrics'].get('mmd', None) for r in results]
    jsd_sc = [r['metrics'].get('jsd_synthcity', None) for r in results]
    jsd_sd = [r['metrics'].get('jsd_syndat', None) for r in results]
    jsd_nm = [r['metrics'].get('jsd_nannyml', None) for r in results]
    det = [r['metrics'].get('detection_avg', None) for r in results]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each metric if available
    if any(x is not None for x in alpha):
        ax.plot(generations, alpha, marker='o', label='Alpha Precision', linewidth=2)
    
    if any(x is not None for x in prdc):
        ax.plot(generations, prdc, marker='s', label='PRDC Avg', linewidth=2)

    if any(x is not None for x in tv):
        ax.plot(generations, tv, marker='D', label='TV Complement', linewidth=2, linestyle='--')

    if any(x is not None for x in ks):
        ax.plot(generations, ks, marker='v', label='KS Complement', linewidth=2, linestyle='--')

    if any(x is not None for x in det):
        ax.plot(generations, det, marker='^', label='Detection Avg', linewidth=2)

    # JSD metrics on primary axis (now similarity scores, higher is better)
    if any(x is not None for x in jsd_sc):
        ax.plot(generations, jsd_sc, marker='*', label='JSD (Synthcity)', linewidth=2)

    if any(x is not None for x in jsd_sd):
        ax.plot(generations, jsd_sd, marker='d', label='JSD (SYNDAT)', linewidth=2)

    # JSD NannyML on primary axis (similarity score, higher is better)
    if any(x is not None for x in jsd_nm):
        ax.plot(generations, jsd_nm, marker='p', label='JSD (NannyML)', linewidth=2)

    # Distance metrics on secondary y-axis (lower is better)
    if any(x is not None for x in wd):
        ax2 = ax.twinx()
        ax2.plot(generations, wd, marker='x', label='Wasserstein Dist', 
                 linewidth=2, linestyle=':', color='red', alpha=0.7)

        # Add MMD to same secondary axis if available
        if any(x is not None for x in mmd):
            ax2.plot(generations, mmd, marker='+', label='MMD', 
                     linewidth=2, linestyle=':', color='darkred', alpha=0.7)
       
        ax2.set_ylabel('Distance Metrics (lower is better)', fontsize=10, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')
    elif any(x is not None for x in mmd):
        # MMD only (no Wasserstein)
        ax2 = ax.twinx()
        ax2.plot(generations, mmd, marker='+', label='MMD', 
                 linewidth=2, linestyle=':', color='darkred', alpha=0.7)
        ax2.set_ylabel('MMD Distance (lower is better)', fontsize=10, color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.legend(loc='upper right')

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


"""
Enhanced trace_chain.py plotting with Plotly for interactive visualization.

Replace the plot_chain() function in trace_chain.py with this version.
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


"""
Enhanced trace_chain.py plotting with Plotly for interactive visualization.

Replace the plot_chain() function in trace_chain.py with this version.
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_chain_interactive(results: List[Dict], output_file: Optional[str] = None):
    """
    Plot metrics degradation across generations with interactive Plotly.
    
    Features:
    - Click legend items to toggle visibility
    - Hover for exact values
    - Zoom and pan
    - Export as standalone HTML
    
    Args:
        results: Chain results from trace_chain()
        output_file: Optional filename to save plot (e.g., 'chain.html')
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        console.print("[red]plotly not installed. Install with: pip install plotly[/red]")
        raise typer.Exit(1)
    
    if not results:
        console.print("[yellow]No data to plot[/yellow]")
        return
    
    generations = [r['generation'] for r in results]
    
    # Extract metrics
    alpha = [r['metrics'].get('alpha_precision', None) for r in results]
    prdc = [r['metrics'].get('prdc_avg', None) for r in results]
    tv = [r['metrics'].get('tv_complement', None) for r in results]
    ks = [r['metrics'].get('ks_complement', None) for r in results]
    wd = [r['metrics'].get('wasserstein_dist', None) for r in results]
    mmd = [r['metrics'].get('mmd', None) for r in results]
    jsd_sc = [r['metrics'].get('jsd_synthcity', None) for r in results]
    jsd_sd = [r['metrics'].get('jsd_syndat', None) for r in results]
    jsd_nm = [r['metrics'].get('jsd_nannyml', None) for r in results]
    det = [r['metrics'].get('detection_avg', None) for r in results]
    
    # Create figure with secondary y-axis for distance metrics
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=["Metric Degradation Across Generations"]
    )
    
    # Color scheme
    colors = {
        'alpha': '#1f77b4',
        'prdc': '#ff7f0e',
        'tv': '#2ca02c',
        'ks': '#d62728',
        'det': '#9467bd',
        'jsd_sc': '#8c564b',
        'jsd_sd': '#e377c2',
        'jsd_nm': '#7f7f7f',
        'wd': '#ff0000',
        'mmd': '#8b0000'
    }
    
    # Primary axis: Similarity scores (higher is better)
    if any(x is not None for x in alpha):
        fig.add_trace(
            go.Scatter(
                x=generations, y=alpha,
                mode='lines+markers',
                name='Alpha Precision',
                line=dict(color=colors['alpha'], width=2),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='Gen %{x}<br>Alpha: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    if any(x is not None for x in prdc):
        fig.add_trace(
            go.Scatter(
                x=generations, y=prdc,
                mode='lines+markers',
                name='PRDC Avg',
                line=dict(color=colors['prdc'], width=2),
                marker=dict(size=8, symbol='square'),
                hovertemplate='Gen %{x}<br>PRDC: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    if any(x is not None for x in tv):
        fig.add_trace(
            go.Scatter(
                x=generations, y=tv,
                mode='lines+markers',
                name='TV Complement',
                line=dict(color=colors['tv'], width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='Gen %{x}<br>TV: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    if any(x is not None for x in ks):
        fig.add_trace(
            go.Scatter(
                x=generations, y=ks,
                mode='lines+markers',
                name='KS Complement',
                line=dict(color=colors['ks'], width=2, dash='dash'),
                marker=dict(size=8, symbol='triangle-down'),
                hovertemplate='Gen %{x}<br>KS: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    if any(x is not None for x in det):
        fig.add_trace(
            go.Scatter(
                x=generations, y=det,
                mode='lines+markers',
                name='Detection Avg',
                line=dict(color=colors['det'], width=2),
                marker=dict(size=8, symbol='triangle-up'),
                hovertemplate='Gen %{x}<br>Detection: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    if any(x is not None for x in jsd_sc):
        fig.add_trace(
            go.Scatter(
                x=generations, y=jsd_sc,
                mode='lines+markers',
                name='JSD (Synthcity)',
                line=dict(color=colors['jsd_sc'], width=2),
                marker=dict(size=8, symbol='star'),
                hovertemplate='Gen %{x}<br>JSD SC: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    if any(x is not None for x in jsd_sd):
        fig.add_trace(
            go.Scatter(
                x=generations, y=jsd_sd,
                mode='lines+markers',
                name='JSD (SYNDAT)',
                line=dict(color=colors['jsd_sd'], width=2),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='Gen %{x}<br>JSD SD: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    if any(x is not None for x in jsd_nm):
        fig.add_trace(
            go.Scatter(
                x=generations, y=jsd_nm,
                mode='lines+markers',
                name='JSD (NannyML)',
                line=dict(color=colors['jsd_nm'], width=2),
                marker=dict(size=8, symbol='pentagon'),
                hovertemplate='Gen %{x}<br>JSD NM: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )
    
    # Secondary axis: Distance metrics (lower is better)
    if any(x is not None for x in wd):
        fig.add_trace(
            go.Scatter(
                x=generations, y=wd,
                mode='lines+markers',
                name='Wasserstein Dist',
                line=dict(color=colors['wd'], width=2, dash='dot'),
                marker=dict(size=8, symbol='x'),
                hovertemplate='Gen %{x}<br>Wasserstein: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )
    
    if any(x is not None for x in mmd):
        fig.add_trace(
            go.Scatter(
                x=generations, y=mmd,
                mode='lines+markers',
                name='MMD',
                line=dict(color=colors['mmd'], width=2, dash='dot'),
                marker=dict(size=8, symbol='cross'),
                hovertemplate='Gen %{x}<br>MMD: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Update axes
    fig.update_xaxes(
        title_text="Generation",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    fig.update_yaxes(
        title_text="Similarity Score (higher is better)",
        secondary_y=False,
        range=[0, 1.05],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    fig.update_yaxes(
        title_text="Distance (lower is better)",
        secondary_y=True,
        showgrid=False
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Metric Degradation Across Generations<br><sub>Click legend items to toggle visibility</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        height=700,
        width=1200,
        margin=dict(r=200)  # Extra space for legend
    )
    
    # Add annotation if alpha precision data available
    if alpha and alpha[0] is not None and alpha[-1] is not None:
        degradation = alpha[0] - alpha[-1]
        fig.add_annotation(
            x=generations[-1],
            y=alpha[-1],
            text=f"Δ Alpha = {degradation:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="rgba(0, 0, 0, 0.5)",
            ax=40,
            ay=-40,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        )
    
    # Save or show
    if output_file:
        # Default to .html extension
        if not output_file.endswith('.html'):
            output_file = output_file.rsplit('.', 1)[0] + '.html'
        
        fig.write_html(
            output_file,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
            }
        )
        console.print(f"[green]Interactive plot saved to: {output_file}[/green]")
        console.print(f"[dim]Open in browser to interact with the plot[/dim]")
    else:
        fig.show()


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
        plot_chain_interactive(results, plot_output)


if __name__ == "__main__":
    app()