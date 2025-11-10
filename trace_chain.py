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

    Format: library_modeltype_refhash_roothash_trnhash_gen_N_[variant]_cfghash_seed
    Example: sdv_gaussiancopula_0cf8e0f5_852ba944_852ba944_gen_0_9eadbd5d_51
    Example: synthcity_arf_504eda11_504eda11_504eda11_gen_0_variant2_d8f76fa1_2211

    Args:
        model_id: Full model identifier string

    Returns:
        Dict with parsed components: library, model_type, ref_hash, root_hash,
                                      training_hash, generation, config_hash, seed, experiment_name

    Raises:
        ValueError: If model_id format is invalid
    """
    parts = model_id.split("_")

    # Validate minimum length
    if len(parts) < 9:
        raise ValueError(
            f"Invalid model_id format (expected at least 9 components): {model_id}\n"
            f"Expected format: library_modeltype_refhash_roothash_trnhash_gen_N_cfghash_seed"
        )

    try:
        # Find "gen" marker position (supports variable-length experiment names)
        try:
            gen_idx = parts.index("gen")
        except ValueError:
            raise ValueError(f"'gen' marker not found in model_id: {model_id}")

        # Extract fixed position components before "gen"
        library = parts[0]
        model_type = parts[1]
        ref_hash = parts[2]
        root_hash = parts[3]
        training_hash = parts[4]

        # Extract generation number (right after "gen")
        if gen_idx + 1 >= len(parts):
            raise ValueError(f"Generation number missing after 'gen' marker: {model_id}")
        generation_num = int(parts[gen_idx + 1])

        # Use relative positions from end for config_hash and seed
        # This handles variable-length experiment names (e.g., with "variant2" suffix)
        seed = parts[-1]
        config_hash = parts[-2]

        # Experiment name is everything except the last two parts (config_hash and seed)
        # e.g., synthcity_arf_504eda11_504eda11_504eda11_gen_0_variant2
        experiment_name = "_".join(parts[:-2])

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
        Dict with metrics: α (Alpha Precision), PRDC (avg), Det (avg), Hallucination metrics
    """
    # Extract components: last 2 parts are always config_hash and seed
    # Everything before that is the experiment_name (includes generation number)
    parts = model_id.split("_")
    seed = parts[-1]
    config_hash = parts[-2]
    experiment_name = "_".join(parts[:-2])

    metrics = {}

    # Statistical similarity metrics
    stat_file = Path(
        f"experiments/metrics/statistical_similarity_{experiment_name}_{config_hash}_{seed}.json"
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

                    # Store all alpha precision components for detailed plotting
                    metrics['alpha_delta_precision_OC'] = scores.get('delta_precision_alpha_OC', 0)
                    metrics['alpha_delta_coverage_OC'] = scores.get('delta_coverage_beta_OC', 0)
                    metrics['alpha_authenticity_OC'] = scores.get('authenticity_OC', 0)

                    metrics['alpha_delta_precision_naive'] = scores.get('delta_precision_alpha_naive', 0)
                    metrics['alpha_delta_coverage_naive'] = scores.get('delta_coverage_beta_naive', 0)
                    metrics['alpha_authenticity_naive'] = scores.get('authenticity_naive', 0)
 
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

        # Jensen-Shannon (Synthcity) - lower is better
        if 'jensenshannon_synthcity' in metrics_data:
            jsd_sc_metric = metrics_data['jensenshannon_synthcity']
            if jsd_sc_metric.get('status') == 'success':
                # Store distance score (lower = more similar distributions)
                metrics['jsd_synthcity'] = jsd_sc_metric.get('distance_score', 1.0)
        
        # Jensen-Shannon (SYNDAT) - lower is better
        if 'jensenshannon_syndat' in metrics_data:
            jsd_sd_metric = metrics_data['jensenshannon_syndat']
            if jsd_sd_metric.get('status') == 'success':
                # Store distance score (lower = more similar distributions)
                metrics['jsd_syndat'] = jsd_sd_metric.get('distance_score', 1.0)

        # Jensen-Shannon (NannyML) - lower is better
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

            # SDV NewRowSynthesis - measures overall novelty (includes factual + hallucinated)
            if 'new_row_synthesis' in metrics_data:
                nrs_metric = metrics_data['new_row_synthesis']
                if nrs_metric.get('status') == 'success':
                    # Store as rate (0-1 range like other metrics)
                    nrs_score = nrs_metric.get('score')
                    if nrs_score is not None:
                        metrics['new_row_synthesis'] = nrs_score

    # Detection metrics
    det_file = Path(
        f"experiments/metrics/detection_evaluation_{experiment_name}_{config_hash}_{seed}.json"
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

    # Hallucination metrics (DDR + Plausibility)
    halluc_file = Path(
        f"experiments/metrics/hallucination_{experiment_name}_{config_hash}_{seed}.json"
    )

    if halluc_file.exists():
        with halluc_file.open() as f:
            data = json.load(f)

            # Factual (Total) - records in population
            quality_matrix = data.get('quality_matrix_2x2', {})
            if 'total_factual' in quality_matrix:
                metrics['factual_total'] = quality_matrix['total_factual'].get('rate_pct', 0) / 100

            # Novel Factual (DDR) - factual AND not memorized (THE IDEAL!)
            if 'novel_factual' in quality_matrix:
                metrics['ddr_novel_factual'] = quality_matrix['novel_factual'].get('rate_pct', 0) / 100

            # Plausible (Total) - passes validation rules
            if 'total_plausible' in quality_matrix:
                metrics['plausible_total'] = quality_matrix['total_plausible'].get('rate_pct', 0) / 100

            # Novel Plausible - plausible AND not memorized
            if 'novel_plausible' in quality_matrix:
                metrics['plausible_novel'] = quality_matrix['novel_plausible'].get('rate_pct', 0) / 100

            # Also get the breakdown for potential stacked chart
            ddr_metrics = data.get('ddr_metrics', {})
            train_copy_valid = data.get('training_copy_valid_metrics', {})
            train_copy_prop = data.get('training_copy_propagation_metrics', {})
            new_halluc = data.get('new_hallucination_metrics', {})

            if ddr_metrics.get('unique'):
                metrics['category_ddr'] = ddr_metrics['unique'].get('rate_pct', 0) / 100
            if train_copy_valid.get('unique'):
                metrics['category_train_copy_valid'] = train_copy_valid['unique'].get('rate_pct', 0) / 100
            if train_copy_prop.get('unique'):
                metrics['category_train_copy_prop'] = train_copy_prop['unique'].get('rate_pct', 0) / 100
            if new_halluc.get('unique'):
                metrics['category_new_halluc'] = new_halluc['unique'].get('rate_pct', 0) / 100

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
        # Glob broadly for generation + seed
        # Note: root_hash and training_hash change each generation in recursive training
        pattern = f"experiments/metrics/statistical_similarity_*_gen_{gen}_*_{seed}.json"
        metric_files = list(Path().glob(pattern))

        if not metric_files:
            # No more generations found
            break

        # Filter by the true chain invariants: library, model_type, ref_hash, seed
        # Note: root_hash changes each generation, so we don't filter by it
        matching_model_id = None
        for metric_file in metric_files:
            filename = metric_file.stem
            candidate_model_id = filename.replace("statistical_similarity_", "")

            # Parse and check if chain invariants match
            try:
                parsed_candidate = parse_model_id(candidate_model_id)

                # Chain invariants: library, model_type, ref_hash, seed
                # (root_hash and training_hash change each generation)
                if (parsed_candidate['library'] == library and
                    parsed_candidate['model_type'] == model_type and
                    parsed_candidate['ref_hash'] == ref_hash and
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
        return "generation,model_id,alpha_precision,prdc_avg,prdc_precision,prdc_recall,prdc_density,prdc_coverage,tv_complement,ks_complement,wasserstein_dist,mmd,jsd_synthcity,jsd_syndat,jsd_nannyml,detection_avg,new_row_synthesis,model_size_mb\n"

    lines = ["generation,model_id,alpha_precision,prdc_avg,prdc_precision,prdc_recall,prdc_density,prdc_coverage,tv_complement,ks_complement,wasserstein_dist,mmd,jsd_synthcity,jsd_syndat,jsd_nannyml,detection_avg,new_row_synthesis,model_size_mb"]
    
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
        nrs = metrics.get('new_row_synthesis', '')
        size = r['model_size_mb'] if r['model_exists'] else ''

        lines.append(f"{gen},{model_id},{alpha},{prdc_avg},{prdc_p},{prdc_r},{prdc_d},{prdc_c},{tv},{ks},{wd},{mmd_val},{jsd_sc},{jsd_sd},{jsd_nm},{det},{nrs},{size}")

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

    # Extract alpha precision components for second subplot
    alpha_components = {
        'OC': {k.replace('alpha_', ''): [r['metrics'].get(k, None) for r in results] 
               for k in ['alpha_delta_precision_OC', 'alpha_delta_coverage_OC', 'alpha_authenticity_OC']},
        'naive': {k.replace('alpha_', ''): [r['metrics'].get(k, None) for r in results]
                  for k in ['alpha_delta_precision_naive', 'alpha_delta_coverage_naive', 'alpha_authenticity_naive']}
    }

    # Calculate average of OC components for main plot
    alpha_oc_avg = []
    for r in results:
        oc_vals = [r['metrics'].get('alpha_delta_precision_OC'),
                   r['metrics'].get('alpha_delta_coverage_OC'),
                   r['metrics'].get('alpha_authenticity_OC')]
        alpha_oc_avg.append(sum(v for v in oc_vals if v is not None) / len([v for v in oc_vals if v is not None]) if any(v is not None for v in oc_vals) else None)

    # Calculate average of JS distances from different libraries for main plot
    js_avg = []
    for r in results:
        js_vals = [r['metrics'].get('jsd_synthcity'),
                   r['metrics'].get('jsd_syndat'),
                   r['metrics'].get('jsd_nannyml')]
        js_avg.append(sum(v for v in js_vals if v is not None) / len([v for v in js_vals if v is not None]) if any(v is not None for v in js_vals) else None)

    # Extract PRDC components for third subplot
    prdc_components = {
        'precision': [r['metrics'].get('prdc_precision', None) for r in results],
        'recall': [r['metrics'].get('prdc_recall', None) for r in results],
        'density': [r['metrics'].get('prdc_density', None) for r in results],
        'coverage': [r['metrics'].get('prdc_coverage', None) for r in results]
    }

    # Extract hallucination metrics for fifth subplot
    ddr = [r['metrics'].get('ddr_novel_factual', None) for r in results]
    factual = [r['metrics'].get('factual_total', None) for r in results]
    plausible = [r['metrics'].get('plausible_total', None) for r in results]
    plausible_novel = [r['metrics'].get('plausible_novel', None) for r in results]
    new_row_synthesis = [r['metrics'].get('new_row_synthesis', None) for r in results]

    # Create figure with 5 subplots (main, alpha, PRDC, JS, hallucination)
    fig, (ax, ax_alpha, ax_prdc, ax_js, ax_halluc) = plt.subplots(5, 1, figsize=(14, 20),
                                        gridspec_kw={'height_ratios': [2, 1, 1, 1, 1.2]})

    # Plot each metric if available
    # TOP SUBPLOT: Main metrics
    if any(x is not None for x in prdc):
        ax.plot(generations, prdc, marker='s', label='PRDC Avg', linewidth=2)

    if any(x is not None for x in alpha_oc_avg):
        ax.plot(generations, alpha_oc_avg, marker='*', label='Alpha OC Avg', linewidth=2, color='#9467bd')

    if any(x is not None for x in tv):
        ax.plot(generations, tv, marker='D', label='TV Complement', linewidth=2, linestyle='--')

    if any(x is not None for x in ks):
        ax.plot(generations, ks, marker='v', label='KS Complement', linewidth=2, linestyle='--')

    if any(x is not None for x in det):
        ax.plot(generations, det, marker='^', label='Detection Avg', linewidth=2)

    # Distance metrics on secondary y-axis (lower is better) - WD, MMD, and JSD
    # Check if we need to create secondary axis
    has_distance_metrics = (any(x is not None for x in wd) or 
                           any(x is not None for x in mmd) or
                           any(x is not None for x in jsd_sc) or 
                           any(x is not None for x in jsd_sd) or 
                           any(x is not None for x in jsd_nm))
    
    if has_distance_metrics:
        ax2 = ax.twinx()
        
        # Wasserstein Distance
        if any(x is not None for x in wd):
            ax2.plot(generations, wd, marker='x', label='Wasserstein Dist', 
                     linewidth=2, linestyle=':', color='red', alpha=0.7)

        # MMD
        if any(x is not None for x in mmd):
            ax2.plot(generations, mmd, marker='+', label='MMD', 
                     linewidth=2, linestyle=':', color='darkred', alpha=0.7)

        # Jensen-Shannon average
        if any(x is not None for x in js_avg):
            ax2.plot(generations, js_avg, marker='o', label='JS Avg', 
                     linewidth=2, linestyle='--', color='darkorange', alpha=0.7)
        
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
    if alpha_oc_avg and alpha_oc_avg[0] is not None and alpha_oc_avg[-1] is not None:
        degradation = alpha_oc_avg[0] - alpha_oc_avg[-1]
        ax.annotate(
            f'Δ = {degradation:.3f}',
            xy=(generations[-1], alpha_oc_avg[-1]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            alpha=0.7
        )

    # BOTTOM SUBPLOT: Alpha Precision Components
    ax_alpha.set_title('Alpha Precision Components', fontsize=12, fontweight='bold', pad=10)

    # Plot OC variant components
    if any(x is not None for x in alpha_components['OC']['delta_precision_OC']):
        ax_alpha.plot(generations, alpha_components['OC']['delta_precision_OC'], 
                      marker='o', label='δ Precision (OC)', linewidth=2, color='#1f77b4')

    if any(x is not None for x in alpha_components['OC']['delta_coverage_OC']):
        ax_alpha.plot(generations, alpha_components['OC']['delta_coverage_OC'], 
                      marker='s', label='δ Coverage (OC)', linewidth=2, color='#ff7f0e')

    if any(x is not None for x in alpha_components['OC']['authenticity_OC']):
        ax_alpha.plot(generations, alpha_components['OC']['authenticity_OC'], 
                      marker='^', label='Authenticity (OC)', linewidth=2, color='#2ca02c')

    # Plot naive variant components with dashed lines
    if any(x is not None for x in alpha_components['naive']['delta_precision_naive']):
        ax_alpha.plot(generations, alpha_components['naive']['delta_precision_naive'], 
                      marker='o', label='δ Precision (Naive)', linewidth=2, 
                      linestyle='--', color='#1f77b4', alpha=0.6)

    if any(x is not None for x in alpha_components['naive']['delta_coverage_naive']):
        ax_alpha.plot(generations, alpha_components['naive']['delta_coverage_naive'], 
                      marker='s', label='δ Coverage (Naive)', linewidth=2, 
                      linestyle='--', color='#ff7f0e', alpha=0.6)

    if any(x is not None for x in alpha_components['naive']['authenticity_naive']):
        ax_alpha.plot(generations, alpha_components['naive']['authenticity_naive'], 
                      marker='^', label='Authenticity (Naive)', linewidth=2, 
                      linestyle='--', color='#2ca02c', alpha=0.6)

    ax_alpha.set_xlabel('Generation', fontsize=12)
    ax_alpha.set_ylabel('Score', fontsize=12)
    ax_alpha.legend(loc='best', fontsize=9)
    ax_alpha.grid(True, alpha=0.3)
    ax_alpha.set_xlim(left=-0.5)
    ax_alpha.set_ylim(0, 1.05)

    # THIRD SUBPLOT: PRDC Components
    ax_prdc.set_title('PRDC Components', fontsize=12, fontweight='bold', pad=10)

    # Plot PRDC components
    if any(x is not None for x in prdc_components['precision']):
        ax_prdc.plot(generations, prdc_components['precision'], 
                     marker='o', label='Precision', linewidth=2, color='#e377c2')

    if any(x is not None for x in prdc_components['recall']):
        ax_prdc.plot(generations, prdc_components['recall'], 
                     marker='s', label='Recall', linewidth=2, color='#7f7f7f')

    if any(x is not None for x in prdc_components['density']):
        ax_prdc.plot(generations, prdc_components['density'], 
                     marker='D', label='Density', linewidth=2, color='#bcbd22')

    if any(x is not None for x in prdc_components['coverage']):
        ax_prdc.plot(generations, prdc_components['coverage'], 
                     marker='^', label='Coverage', linewidth=2, color='#17becf')

    ax_prdc.set_xlabel('Generation', fontsize=12)
    ax_prdc.set_ylabel('Score', fontsize=12)
    ax_prdc.legend(loc='best', fontsize=9)
    ax_prdc.grid(True, alpha=0.3)
    ax_prdc.set_xlim(left=-0.5)
    ax_prdc.set_ylim(0, 1.2)

    # FOURTH SUBPLOT: Jensen-Shannon Components
    ax_js.set_title('Jensen-Shannon Distance Components', fontsize=12, fontweight='bold', pad=10)

    # Plot individual JS distances from different libraries
    if any(x is not None for x in jsd_sc):
        ax_js.plot(generations, jsd_sc, 
                   marker='o', label='JSD (Synthcity)', linewidth=2, color='orange')

    if any(x is not None for x in jsd_sd):
        ax_js.plot(generations, jsd_sd, 
                   marker='s', label='JSD (SYNDAT)', linewidth=2, color='darkorange')

    if any(x is not None for x in jsd_nm):
        ax_js.plot(generations, jsd_nm, 
                   marker='^', label='JSD (NannyML)', linewidth=2, color='coral')

    # Add average as bold line
    if any(x is not None for x in js_avg):
        ax_js.plot(generations, js_avg, 
                   marker='D', label='JS Average', linewidth=3, color='red', alpha=0.8)

    ax_js.set_xlabel('Generation', fontsize=12)
    ax_js.set_ylabel('Distance (lower is better)', fontsize=12)
    ax_js.legend(loc='best', fontsize=9)
    ax_js.grid(True, alpha=0.3)
    ax_js.set_xlim(left=-0.5)

    # FIFTH SUBPLOT: Hallucination Metrics (DDR + Plausibility)
    ax_halluc.set_title('Hallucination Metrics (DDR + Plausibility)', fontsize=12, fontweight='bold', pad=10)

    # Plot hallucination metrics (all should be high - higher is better)
    if any(x is not None for x in ddr):
        ax_halluc.plot(generations, ddr,
                       marker='*', label='DDR (Novel Factual)', linewidth=3, color='#2ca02c', markersize=10)

    if any(x is not None for x in factual):
        ax_halluc.plot(generations, factual,
                       marker='o', label='Factual (Total)', linewidth=2, color='#1f77b4')

    if any(x is not None for x in plausible):
        ax_halluc.plot(generations, plausible,
                       marker='s', label='Plausible (Total)', linewidth=2, color='#ff7f0e')

    if any(x is not None for x in plausible_novel):
        ax_halluc.plot(generations, plausible_novel,
                       marker='^', label='Novel Plausible', linewidth=2, color='#9467bd')

    if any(x is not None for x in new_row_synthesis):
        ax_halluc.plot(generations, new_row_synthesis,
                       marker='D', label='NewRowSynthesis (SDV)', linewidth=2, color='#8c564b', linestyle='--')

    ax_halluc.set_xlabel('Generation', fontsize=12)
    ax_halluc.set_ylabel('Rate (higher is better)', fontsize=12)
    ax_halluc.legend(loc='best', fontsize=9)
    ax_halluc.grid(True, alpha=0.3)
    ax_halluc.set_xlim(left=-0.5)
    ax_halluc.set_ylim(0, 1.05)

    # Add interpretation annotations
    ax_halluc.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax_halluc.text(0.02, 0.95, 'Higher = Better Quality', transform=ax_halluc.transAxes,
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

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
    
    Creates two independent plots in a single HTML file:
    1. Main metrics degradation plot (with dual Y-axes)
    2. Alpha precision components breakdown
    
    Features:
    - Click legend items to toggle visibility independently
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

    # Extract alpha precision components for second subplot
    alpha_components = {
        'OC': {k.replace('alpha_', ''): [r['metrics'].get(k, None) for r in results] 
               for k in ['alpha_delta_precision_OC', 'alpha_delta_coverage_OC', 'alpha_authenticity_OC']},
        'naive': {k.replace('alpha_', ''): [r['metrics'].get(k, None) for r in results]
                  for k in ['alpha_delta_precision_naive', 'alpha_delta_coverage_naive', 'alpha_authenticity_naive']}
    }

    # Calculate average of OC components for main plot
    alpha_oc_avg = []
    for r in results:
        oc_vals = [r['metrics'].get('alpha_delta_precision_OC'),
                   r['metrics'].get('alpha_delta_coverage_OC'),
                   r['metrics'].get('alpha_authenticity_OC')]
        alpha_oc_avg.append(sum(v for v in oc_vals if v is not None) / len([v for v in oc_vals if v is not None]) if any(v is not None for v in oc_vals) else None)

    # Calculate average of JS distances from different libraries for main plot
    js_avg = []
    for r in results:
        js_vals = [r['metrics'].get('jsd_synthcity'),
                   r['metrics'].get('jsd_syndat'),
                   r['metrics'].get('jsd_nannyml')]
        js_avg.append(sum(v for v in js_vals if v is not None) / len([v for v in js_vals if v is not None]) if any(v is not None for v in js_vals) else None)

    # Extract PRDC components for third plot
    prdc_components = {
        'precision': [r['metrics'].get('prdc_precision', None) for r in results],
        'recall': [r['metrics'].get('prdc_recall', None) for r in results],
        'density': [r['metrics'].get('prdc_density', None) for r in results],
        'coverage': [r['metrics'].get('prdc_coverage', None) for r in results]
    }

    # Extract hallucination metrics for fifth plot
    ddr = [r['metrics'].get('ddr_novel_factual', None) for r in results]
    factual = [r['metrics'].get('factual_total', None) for r in results]
    plausible = [r['metrics'].get('plausible_total', None) for r in results]
    plausible_novel = [r['metrics'].get('plausible_novel', None) for r in results]
    new_row_synthesis = [r['metrics'].get('new_row_synthesis', None) for r in results]

    # Color scheme
    colors = {
        'alpha': '#1f77b4',
        'prdc': '#ff7f0e',
        'tv': '#2ca02c',
        'ks': '#d62728',
        'det': '#9467bd',
        'alpha_oc_avg': '#8c564b',
        'wd': '#ff0000',
        'mmd': '#8b0000'
    }
    
    # ========================================
    # PLOT 1: Main Metrics with Dual Y-Axes
    # ========================================
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Primary axis: Similarity scores (higher is better)
    if any(x is not None for x in prdc):
        fig1.add_trace(
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
    
    if any(x is not None for x in alpha_oc_avg):
        fig1.add_trace(
            go.Scatter(
                x=generations, y=alpha_oc_avg,
                mode='lines+markers',
                name='Alpha OC Avg',
                line=dict(color=colors['alpha_oc_avg'], width=2),
                marker=dict(size=8, symbol='star'),
                hovertemplate='Gen %{x}<br>Alpha OC Avg: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )

    if any(x is not None for x in tv):
        fig1.add_trace(
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
        fig1.add_trace(
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
        fig1.add_trace(
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
    
    # Secondary axis: Distance metrics (lower is better)
    if any(x is not None for x in wd):
        fig1.add_trace(
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
        fig1.add_trace(
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

    # JS Average on secondary axis
    if any(x is not None for x in js_avg):
        fig1.add_trace(
            go.Scatter(
                x=generations, y=js_avg,
                mode='lines+markers',
                name='JS Avg',
                line=dict(color='darkorange', width=2, dash='dash'),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='Gen %{x}<br>JS Avg: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )

    # Configure axes for Plot 1
    fig1.update_xaxes(
        title_text="Generation",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    fig1.update_yaxes(
        title_text="Similarity Score (higher is better)",
        secondary_y=False,
        range=[0, 1.05],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    fig1.update_yaxes(
        title_text="Distance (lower is better)",
        secondary_y=True,
        showgrid=False
    )

    # Layout for Plot 1
    fig1.update_layout(
        title={
            'text': 'Metric Degradation Across Generations',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        height=550,
        width=1400,
        margin=dict(r=250, l=80, t=80, b=60)
    )
    
    # Add annotation if alpha precision data available
    if alpha_oc_avg and alpha_oc_avg[0] is not None and alpha_oc_avg[-1] is not None:
        degradation = alpha_oc_avg[0] - alpha_oc_avg[-1]
        fig1.add_annotation(
            x=generations[-1],
            y=alpha_oc_avg[-1],
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
    
    # ========================================
    # PLOT 2: Alpha Precision Components
    # ========================================
    fig2 = go.Figure()
    
    # OC variant - solid lines
    if any(x is not None for x in alpha_components['OC']['delta_precision_OC']):
        fig2.add_trace(
            go.Scatter(
                x=generations, y=alpha_components['OC']['delta_precision_OC'],
                mode='lines+markers',
                name='δ Precision (OC)',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='Gen %{x}<br>δ Prec (OC): %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in alpha_components['OC']['delta_coverage_OC']):
        fig2.add_trace(
            go.Scatter(
                x=generations, y=alpha_components['OC']['delta_coverage_OC'],
                mode='lines+markers',
                name='δ Coverage (OC)',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=7, symbol='square'),
                hovertemplate='Gen %{x}<br>δ Cov (OC): %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in alpha_components['OC']['authenticity_OC']):
        fig2.add_trace(
            go.Scatter(
                x=generations, y=alpha_components['OC']['authenticity_OC'],
                mode='lines+markers',
                name='Authenticity (OC)',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=7, symbol='triangle-up'),
                hovertemplate='Gen %{x}<br>Auth (OC): %{y:.4f}<extra></extra>'
            )
        )

    # Naive variant - dashed lines
    if any(x is not None for x in alpha_components['naive']['delta_precision_naive']):
        fig2.add_trace(
            go.Scatter(
                x=generations, y=alpha_components['naive']['delta_precision_naive'],
                mode='lines+markers',
                name='δ Precision (Naive)',
                line=dict(color='#1f77b4', width=2, dash='dash'),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='Gen %{x}<br>δ Prec (Naive): %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in alpha_components['naive']['delta_coverage_naive']):
        fig2.add_trace(
            go.Scatter(
                x=generations, y=alpha_components['naive']['delta_coverage_naive'],
                mode='lines+markers',
                name='δ Coverage (Naive)',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=7, symbol='square'),
                hovertemplate='Gen %{x}<br>δ Cov (Naive): %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in alpha_components['naive']['authenticity_naive']):
        fig2.add_trace(
            go.Scatter(
                x=generations, y=alpha_components['naive']['authenticity_naive'],
                mode='lines+markers',
                name='Authenticity (Naive)',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                marker=dict(size=7, symbol='triangle-up'),
                hovertemplate='Gen %{x}<br>Auth (Naive): %{y:.4f}<extra></extra>'
            )
        )

    # Configure axes for Plot 2
    fig2.update_xaxes(
        title_text="Generation",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )

    fig2.update_yaxes(
        title_text="Score",
        range=[0, 1.05],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )

    # Layout for Plot 2
    fig2.update_layout(
        title={
            'text': 'Alpha Precision Components',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        height=450,
        width=1400,
        margin=dict(r=250, l=80, t=80, b=60)
    )

    # ========================================
    # PLOT 3: PRDC Components
    # ========================================
    fig3 = go.Figure()
    
    if any(x is not None for x in prdc_components['precision']):
        fig3.add_trace(
            go.Scatter(
                x=generations, y=prdc_components['precision'],
                mode='lines+markers',
                name='Precision',
                line=dict(color='#e377c2', width=2),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='Gen %{x}<br>Precision: %{y:.4f}<extra></extra>'
            )
        )
    
    if any(x is not None for x in prdc_components['recall']):
        fig3.add_trace(
            go.Scatter(
                x=generations, y=prdc_components['recall'],
                mode='lines+markers',
                name='Recall',
                line=dict(color='#7f7f7f', width=2),
                marker=dict(size=7, symbol='square'),
                hovertemplate='Gen %{x}<br>Recall: %{y:.4f}<extra></extra>'
            )
        )
    
    if any(x is not None for x in prdc_components['density']):
        fig3.add_trace(
            go.Scatter(
                x=generations, y=prdc_components['density'],
                mode='lines+markers',
                name='Density',
                line=dict(color='#bcbd22', width=2),
                marker=dict(size=7, symbol='diamond'),
                hovertemplate='Gen %{x}<br>Density: %{y:.4f}<extra></extra>'
            )
        )
    
    if any(x is not None for x in prdc_components['coverage']):
        fig3.add_trace(
            go.Scatter(
                x=generations, y=prdc_components['coverage'],
                mode='lines+markers',
                name='Coverage',
                line=dict(color='#17becf', width=2),
                marker=dict(size=7, symbol='triangle-up'),
                hovertemplate='Gen %{x}<br>Coverage: %{y:.4f}<extra></extra>'
            )
        )
    
    # Configure axes for Plot 3
    fig3.update_xaxes(
        title_text="Generation",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    fig3.update_yaxes(
        title_text="Score",
        range=[0, 1.20],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )

    # Layout for Plot 3
    fig3.update_layout(
        title={
            'text': 'PRDC Components',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        height=450,
        width=1400,
        margin=dict(r=250, l=80, t=80, b=60)
    )

    # ========================================
    # PLOT 4: Jensen-Shannon Distance Components
    # ========================================
    fig4 = go.Figure()

    if any(x is not None for x in jsd_sc):
        fig4.add_trace(
            go.Scatter(
                x=generations, y=jsd_sc,
                mode='lines+markers',
                name='JSD (Synthcity)',
                line=dict(color='orange', width=2),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='Gen %{x}<br>JSD Synthcity: %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in jsd_sd):
        fig4.add_trace(
            go.Scatter(
                x=generations, y=jsd_sd,
                mode='lines+markers',
                name='JSD (SYNDAT)',
                line=dict(color='darkorange', width=2),
                marker=dict(size=7, symbol='square'),
                hovertemplate='Gen %{x}<br>JSD SYNDAT: %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in jsd_nm):
        fig4.add_trace(
            go.Scatter(
                x=generations, y=jsd_nm,
                mode='lines+markers',
                name='JSD (NannyML)',
                line=dict(color='coral', width=2),
                marker=dict(size=7, symbol='triangle-up'),
                hovertemplate='Gen %{x}<br>JSD NannyML: %{y:.4f}<extra></extra>'
            )
        )

    # Add average as bold line
    if any(x is not None for x in js_avg):
        fig4.add_trace(
            go.Scatter(
                x=generations, y=js_avg,
                mode='lines+markers',
                name='JS Average',
                line=dict(color='red', width=3),
                marker=dict(size=9, symbol='diamond'),
                hovertemplate='Gen %{x}<br>JS Avg: %{y:.4f}<extra></extra>'
            )
        )

    # Configure axes for Plot 4
    fig4.update_xaxes(
        title_text="Generation",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )

    fig4.update_yaxes(
        title_text="Distance (lower is better)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )

    # Layout for Plot 4
    fig4.update_layout(
        title={
            'text': 'Jensen-Shannon Distance Components',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        height=450,
        width=1400,
        margin=dict(r=250, l=80, t=80, b=60)
    )

    # ========================================
    # PLOT 5: Hallucination Metrics (DDR + Plausibility)
    # ========================================
    fig5 = go.Figure()

    if any(x is not None for x in ddr):
        fig5.add_trace(
            go.Scatter(
                x=generations, y=ddr,
                mode='lines+markers',
                name='DDR (Novel Factual)',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=10, symbol='star'),
                hovertemplate='Gen %{x}<br>DDR: %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in factual):
        fig5.add_trace(
            go.Scatter(
                x=generations, y=factual,
                mode='lines+markers',
                name='Factual (Total)',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='Gen %{x}<br>Factual: %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in plausible):
        fig5.add_trace(
            go.Scatter(
                x=generations, y=plausible,
                mode='lines+markers',
                name='Plausible (Total)',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8, symbol='square'),
                hovertemplate='Gen %{x}<br>Plausible: %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in plausible_novel):
        fig5.add_trace(
            go.Scatter(
                x=generations, y=plausible_novel,
                mode='lines+markers',
                name='Novel Plausible',
                line=dict(color='#9467bd', width=2),
                marker=dict(size=8, symbol='triangle-up'),
                hovertemplate='Gen %{x}<br>Novel Plausible: %{y:.4f}<extra></extra>'
            )
        )

    if any(x is not None for x in new_row_synthesis):
        fig5.add_trace(
            go.Scatter(
                x=generations, y=new_row_synthesis,
                mode='lines+markers',
                name='NewRowSynthesis (SDV)',
                line=dict(color='#8c564b', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='Gen %{x}<br>NewRowSynthesis: %{y:.4f}<extra></extra>'
            )
        )

    fig5.update_xaxes(
        title_text="Generation",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )

    fig5.update_yaxes(
        title_text="Rate (higher is better)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        range=[0, 1.05]
    )

    # Add reference line at 0.5
    fig5.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3)

    # Layout for Plot 5
    fig5.update_layout(
        title={
            'text': 'Hallucination Metrics (DDR + Plausibility)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        height=450,
        width=1400,
        margin=dict(r=250, l=80, t=80, b=60),
        annotations=[
            dict(
                text="Higher values = Better quality<br>DDR = Factual AND Novel (ideal metric)",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255, 248, 220, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                font=dict(size=10),
                xanchor='left',
                yanchor='top'
            )
        ]
    )

    # ========================================
    # Save or show all five plots
    # ========================================
    if output_file:
        # Default to .html extension
        if not output_file.endswith('.html'):
            output_file = output_file.rsplit('.', 1)[0] + '.html'
        
        # Combine all three figures into a single HTML file
        with open(output_file, 'w') as f:
            f.write('<html><head><meta charset="utf-8" /></head><body>\n')
            f.write('<h1 style="text-align: center; font-family: Arial, sans-serif;">Chain Metrics Analysis</h1>\n')
            f.write('<p style="text-align: center; color: #666; font-family: Arial, sans-serif;">Click legend items to toggle visibility. Hover over lines for exact values.</p>\n')
            
            # Write first plot
            f.write(fig1.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                }
            ))
            
            f.write('<br><hr style="margin: 40px auto; width: 80%; border: 1px solid #ddd;"><br>\n')
            
            # Write second plot
            f.write(fig2.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                }
            ))

            f.write('<br><hr style="margin: 40px auto; width: 80%; border: 1px solid #ddd;"><br>\n')

            # Write third plot
            f.write(fig3.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                }
            ))

            f.write('<br><hr style="margin: 40px auto; width: 80%; border: 1px solid #ddd;"><br>\n')

            # Write fourth plot
            f.write(fig4.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                }
            ))

            f.write('<br><hr style="margin: 40px auto; width: 80%; border: 1px solid #ddd;"><br>\n')

            # Write fifth plot
            f.write(fig5.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                }
            ))

            f.write('</body></html>')

        console.print(f"[green]Interactive plots saved to: {output_file}[/green]")
        console.print(f"[dim]Open in browser to interact with the plots[/dim]")
    else:
        # Show all plots sequentially
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()

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