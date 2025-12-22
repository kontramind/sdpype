"""
Statistical metrics evaluation script for SDPype - Alpha Precision and PRDC
"""

import json
from pathlib import Path

import duckdb
import hydra
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from sdv.metadata import SingleTableMetadata
from sdpype.evaluation.statistical import evaluate_statistical_metrics, generate_statistical_report
from sdpype.encoding import load_encoding_config
from sdpype.metadata import load_csv_with_metadata

console = Console()


def _get_config_hash() -> str:
    """Get config hash from temporary file created during pipeline execution"""
    try:
        if Path('.sdpype_config_hash').exists():
            with open('.sdpype_config_hash', 'r', encoding='utf-8') as f:
                return f.read().strip()
        return "nohash"
    except Exception:
        return "nohash"


def _validate_reference_subset_of_population(
    population_file: Path,
    reference_file: Path,
    metadata_file: Path
):
    """
    Validate that all reference records exist in the population dataset.

    In recursive training (RD → SD gen0 → SD gen1 ...), the reference dataset
    is always the original real data, fixed across all generations. It must be
    a subset of the population for meaningful evaluation.

    Args:
        population_file: Path to population CSV (full real dataset)
        reference_file: Path to reference CSV (evaluation real dataset)
        metadata_file: Path to metadata JSON for type consistency

    Raises:
        ValueError: If reference contains records not found in population
    """

    if not population_file.exists():
        print(f"⚠️  Population file not found, skipping reference validation: {population_file}")
        return

    if not reference_file.exists():
        print(f"⚠️  Reference file not found, skipping reference validation: {reference_file}")
        return

    print("🔍 Validating reference ⊆ population assumption...")

    # Load both datasets with metadata for type consistency
    population = load_csv_with_metadata(population_file, metadata_file, low_memory=False)
    reference = load_csv_with_metadata(reference_file, metadata_file, low_memory=False)

    # Use DuckDB to efficiently compute hashes and check set membership
    con = duckdb.connect()
    con.register('population', population)
    con.register('reference', reference)

    # Build hash expression with actual column names
    columns = population.columns.tolist()
    hash_cols = ', '.join([f'"{col}"' for col in columns])

    # Query to find reference records not in population
    validation_query = f"""
    WITH
    population_hashes AS (
        SELECT DISTINCT hash({hash_cols}) as row_hash FROM population
    ),
    reference_hashes AS (
        SELECT hash({hash_cols}) as row_hash FROM reference
    ),
    reference_not_in_pop AS (
        SELECT COUNT(*) as count
        FROM reference_hashes r
        WHERE r.row_hash NOT IN (SELECT row_hash FROM population_hashes)
    )
    SELECT count FROM reference_not_in_pop;
    """

    result = con.execute(validation_query).fetchone()
    con.close()

    invalid_count = result[0]

    if invalid_count > 0:
        raise ValueError(
            f"❌ Reference dataset validation failed!\n"
            f"\n"
            f"Found {invalid_count} reference records that DO NOT exist in population.\n"
            f"\n"
            f"CRITICAL ASSUMPTION VIOLATED: reference ⊆ population\n"
            f"\n"
            f"In recursive training (RD → SD gen0 → SD gen1 ...), the reference dataset\n"
            f"represents the original real data and must be a subset of the population.\n"
            f"This ensures meaningful evaluation across generations:\n"
            f"  - Statistical metrics track drift from real data distribution\n"
            f"  - Detection metrics measure synthetic vs. real distinguishability\n"
            f"  - Results are comparable across generations\n"
            f"\n"
            f"Please check your data configuration:\n"
            f"  - data.population_file: Should contain the full real dataset\n"
            f"  - data.reference_file: Should be a subset of population (original real data)\n"
            f"\n"
            f"Population: {population.shape[0]:,} rows\n"
            f"Reference: {reference.shape[0]:,} rows\n"
            f"Invalid: {invalid_count:,} reference rows not in population"
        )

    print(f"✅ Validation passed: All {reference.shape[0]:,} reference records exist in population")


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Run statistical metrics evaluation between reference and synthetic data"""

    # Check if statistical metrics are configured
    metrics_config = cfg.get("evaluation", {}).get("statistical_similarity", {}).get("metrics", [])

    if not metrics_config:
        print("📊 No statistical metrics configured, skipping evaluation")
        return

    print(f"📊 Starting statistical evaluation with {len(metrics_config)} metrics...")
    print(f"Experiment seed: {cfg.experiment.seed}")

    config_hash = _get_config_hash()

    # Define which metrics need encoded vs decoded data
    # Metrics that need all-numeric (encoded) data:
    # - Synthcity distance metrics require numeric data
    # - KSComplement works on distributions and should evaluate ALL encoded columns
    ENCODED_METRICS = {
        'alpha_precision',
        'prdc_score',
        'jensenshannon_synthcity',
        'jensenshannon_syndat',
        'jensenshannon_nannyml',
        'wasserstein_distance',
        'maximum_mean_discrepancy',
        'ks_complement'  # Moved to encoded to evaluate all numeric columns including one-hot
    }

    # Metrics from SDV that understand sdtypes (decoded/native data)
    # These need semantic understanding of original data types
    DECODED_METRICS = {
        'tv_complement',        # Needs original categorical columns
        'table_structure',      # Needs original table structure
        'semantic_structure',   # Needs original sdtypes for comparison
        'boundary_adherence',   # Needs original numeric ranges
        'category_adherence',   # Needs original categories
        'new_row_synthesis'     # Needs to compare original rows
    }

    # Determine which data formats we need based on configured metrics
    needs_encoded = False
    needs_decoded = False

    for metric_config in metrics_config:
        metric_name = metric_config.get('name', '')
        if metric_name in ENCODED_METRICS:
            needs_encoded = True
        elif metric_name in DECODED_METRICS:
            needs_decoded = True

    # Load metadata
    metadata_path = cfg.data.metadata_file
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    metadata = SingleTableMetadata.load_from_json(metadata_path)

    # Validate that reference ⊆ population (important for recursive training)
    # Reference is always real data, even in later generations, so it should be in population
    population_file = cfg.data.get("population_file")
    reference_file = cfg.data.get("reference_file")
    if population_file and reference_file:
        _validate_reference_subset_of_population(
            Path(population_file),
            Path(reference_file),
            Path(metadata_path)
        )

    # Load encoding config (for determining numeric columns in encoded data)
    encoding_config = None
    if needs_encoded and hasattr(cfg, 'encoding') and cfg.encoding.get('config_file'):
        encoding_config_path = Path(cfg.encoding.config_file)
        if encoding_config_path.exists():
            print(f"📋 Loading encoding config: {encoding_config_path}")
            encoding_config = load_encoding_config(encoding_config_path)
        else:
            print(f"⚠️  Warning: Encoding config not found at {encoding_config_path}")
            print(f"   Metrics will use fallback column detection")

    # Load reference data (both formats if needed)
    base_name = f"{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}"

    reference_data_encoded = None
    reference_data_decoded = None
    synthetic_data_encoded = None
    synthetic_data_decoded = None

    if needs_encoded:
        reference_encoded_path = f"experiments/data/encoded/reference_{base_name}.csv"
        synthetic_encoded_path = f"experiments/data/encoded/synthetic_{base_name}.csv"

        if not Path(reference_encoded_path).exists():
            raise FileNotFoundError(f"Encoded reference data not found: {reference_encoded_path}")
        if not Path(synthetic_encoded_path).exists():
            raise FileNotFoundError(f"Encoded synthetic data not found: {synthetic_encoded_path}")

        print(f"📊 Loading encoded reference data: {reference_encoded_path}")
        print(f"📊 Loading encoded synthetic data: {synthetic_encoded_path}")

        reference_data_encoded = pd.read_csv(reference_encoded_path)
        synthetic_data_encoded = pd.read_csv(synthetic_encoded_path)

    if needs_decoded:
        reference_decoded_path = f"experiments/data/decoded/reference_{base_name}.csv"
        synthetic_decoded_path = f"experiments/data/synthetic/synthetic_data_{base_name}_decoded.csv"

        if not Path(reference_decoded_path).exists():
            raise FileNotFoundError(f"Decoded reference data not found: {reference_decoded_path}")
        if not Path(synthetic_decoded_path).exists():
            raise FileNotFoundError(f"Decoded synthetic data not found: {synthetic_decoded_path}")

        print(f"📊 Loading decoded reference data: {reference_decoded_path}")
        print(f"📊 Loading decoded synthetic data: {synthetic_decoded_path}")

        # Use metadata-based loading for type consistency
        reference_data_decoded = load_csv_with_metadata(Path(reference_decoded_path), Path(metadata_path))
        synthetic_data_decoded = load_csv_with_metadata(Path(synthetic_decoded_path), Path(metadata_path))

    # Set reference_data for display purposes (prefer decoded for original column names)
    reference_data = reference_data_decoded if reference_data_decoded is not None else reference_data_encoded
    synthetic_data = synthetic_data_decoded if synthetic_data_decoded is not None else synthetic_data_encoded

    # Run statistical metrics evaluation with routing
    print("🔄 Running statistical metrics analysis...")
    statistical_results = evaluate_statistical_metrics(
        reference_data_encoded if needs_encoded else reference_data_decoded,
        synthetic_data_encoded if needs_encoded else synthetic_data_decoded,
        metrics_config,
        experiment_name=f"{cfg.experiment.name}_seed_{cfg.experiment.seed}",
        metadata=metadata,
        # Pass both data formats for metrics that might need routing
        reference_data_decoded=reference_data_decoded,
        synthetic_data_decoded=synthetic_data_decoded,
        reference_data_encoded=reference_data_encoded,
        synthetic_data_encoded=synthetic_data_encoded,
        encoded_metrics=ENCODED_METRICS,
        decoded_metrics=DECODED_METRICS,
        encoding_config=encoding_config
    )

    # Save statistical similarity results
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    with open(f"experiments/metrics/statistical_similarity_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json", "w", encoding='utf-8') as f:
        json.dump(statistical_results, f, indent=2, ensure_ascii=False)

    print(f"📊 Statistical metrics results saved: experiments/metrics/statistical_similarity_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json")

    # Generate and save human-readable report
    report = generate_statistical_report(statistical_results)
    with open(f"experiments/metrics/statistical_report_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.txt", "w", encoding='utf-8') as f:
        f.write(report)

    print(f"📋 Statistical metrics report saved: experiments/metrics/statistical_report_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.txt")

    # Print individual metrics summary
    console.print("\n📊 Statistical Metrics Summary:", style="bold cyan")

    metrics = statistical_results.get("metrics", {})

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

        # Show all columns from the reference dataset
        all_columns = list(reference_data.columns)
        column_scores = ba_result['column_scores']
        compatible_columns = ba_result['compatible_columns']

        for col in sorted(all_columns):
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

            # Show all columns from the reference dataset
            all_columns = list(reference_data.columns)
            column_scores = ca_result['column_scores']
            compatible_columns = ca_result['compatible_columns']

            for col in sorted(all_columns):
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

    # KSComplement results table
    if "ks_complement" in metrics and metrics["ks_complement"]["status"] == "success":
        ks_result = metrics["ks_complement"]

        # Get parameters info for display
        params_info = ks_result["parameters"]
        target_cols = params_info.get("target_columns", "all numerical/datetime")
        params_display = f"target_columns={target_cols}"

        # Create KSComplement results table
        ks_table = Table(title=f"✅ KSComplement Results ({params_display})", show_header=True, header_style="bold blue")
        ks_table.add_column("Column", style="cyan", no_wrap=True)
        ks_table.add_column("KS Score", style="bright_green", justify="right")
        ks_table.add_column("Status", style="yellow", justify="center")

        # Add aggregate score as first row
        if ks_result['aggregate_score'] is not None:
            ks_table.add_row("AGGREGATE", f"{ks_result['aggregate_score']:.3f}", "✓")
        else:
            ks_table.add_row("AGGREGATE", "n/a", "ℹ️")

        ks_table.add_section()

        # Show all columns from the reference dataset
        all_columns = list(reference_data.columns)
        column_scores = ks_result['column_scores']
        compatible_columns = ks_result['compatible_columns']

        for col in sorted(all_columns):
            if col in column_scores:
                ks_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
            elif col in compatible_columns:
                ks_table.add_row(col, "error", "⚠️")
            else:
                ks_table.add_row(col, "n/a", "—")

        # Add message at the bottom if no compatible columns found
        if ks_result.get("message"):
            ks_table.add_section()
            ks_table.add_row("INFO", ks_result["message"], "ℹ️")

        console.print(ks_table)
    else:
        console.print("❌ KSComplement failed", style="bold red")

    # TVComplement results table
    if "tv_complement" in metrics and metrics["tv_complement"]["status"] == "success":
        tv_result = metrics["tv_complement"]

        # Get parameters info for display
        params_info = tv_result["parameters"]
        target_cols = params_info.get("target_columns", "all categorical/boolean")
        params_display = f"target_columns={target_cols}"

        # Create TVComplement results table
        tv_table = Table(title=f"✅ TVComplement Results ({params_display})", show_header=True, header_style="bold blue")
        tv_table.add_column("Column", style="cyan", no_wrap=True)
        tv_table.add_column("TV Score", style="bright_green", justify="right")
        tv_table.add_column("Status", style="yellow", justify="center")

        # Add aggregate score as first row
        if tv_result['aggregate_score'] is not None:
            tv_table.add_row("AGGREGATE", f"{tv_result['aggregate_score']:.3f}", "✓")
        else:
            tv_table.add_row("AGGREGATE", "n/a", "ℹ️")

        tv_table.add_section()

        # Show all columns from the reference dataset
        all_columns = list(reference_data.columns)
        column_scores = tv_result['column_scores']
        compatible_columns = tv_result['compatible_columns']

        for col in sorted(all_columns):
            if col in column_scores:
                tv_table.add_row(col, f"{column_scores[col]:.3f}", "✓")
            elif col in compatible_columns:
                tv_table.add_row(col, "error", "⚠️")
            else:
                tv_table.add_row(col, "n/a", "—")

        # Add message at the bottom if no compatible columns found
        if tv_result.get("message"):
            tv_table.add_section()
            tv_table.add_row("INFO", tv_result["message"], "ℹ️")

        console.print(tv_table)
    else:
        console.print("❌ TVComplement failed", style="bold red")


    if "alpha_precision" in metrics and metrics["alpha_precision"]["status"] == "success":
        scores = metrics["alpha_precision"]["scores"]
        # Get parameters info for display
        params_info = metrics["alpha_precision"]["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        # Create Alpha Precision results table
        table = Table(title=f"✅ Alpha Precision Results ({params_display})", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("OC Variant", style="green", justify="right")
        table.add_column("Naive Variant", style="yellow", justify="right")

        table.add_row(
            "Delta Precision Alpha",
            f"{scores['delta_precision_alpha_OC']:.3f}",
            f"{scores['delta_precision_alpha_naive']:.3f}"
        )
        table.add_row(
            "Delta Coverage Beta",
            f"{scores['delta_coverage_beta_OC']:.3f}",
            f"{scores['delta_coverage_beta_naive']:.3f}"
        )
        table.add_row(
            "Authenticity",
            f"{scores['authenticity_OC']:.3f}",
            f"{scores['authenticity_naive']:.3f}"
        )

        console.print(table)
    else:
        console.print("❌ Alpha Precision failed", style="bold red")

    # PRDC Score results table
    if "prdc_score" in metrics and metrics["prdc_score"]["status"] == "success":
        prdc_result = metrics["prdc_score"]

        # Get parameters info for display
        params_info = prdc_result["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        # Create PRDC results table
        prdc_table = Table(title=f"✅ PRDC Score Results ({params_display})", show_header=True, header_style="bold blue")
        prdc_table.add_column("Metric", style="cyan", no_wrap=True)
        prdc_table.add_column("Score", style="bright_green", justify="right")

        prdc_table.add_row("Precision", f"{prdc_result['precision']:.3f}")
        prdc_table.add_row("Recall", f"{prdc_result['recall']:.3f}")
        prdc_table.add_row("Density", f"{prdc_result['density']:.3f}")
        prdc_table.add_row("Coverage", f"{prdc_result['coverage']:.3f}")

        console.print(prdc_table)
    else:
        console.print("❌ PRDC Score failed", style="bold red")

    # Wasserstein Distance results table
    if "wasserstein_distance" in metrics and metrics["wasserstein_distance"]["status"] == "success":
        wd_result = metrics["wasserstein_distance"]

        # Get parameters info for display
        params_info = wd_result["parameters"]
        params_display = str(params_info) if params_info else "default settings"

        # Create Wasserstein Distance results table
        wd_table = Table(title=f"✅ Wasserstein Distance Results ({params_display})", show_header=True, header_style="bold blue")
        wd_table.add_column("Metric", style="cyan", no_wrap=True)
        wd_table.add_column("Score", style="bright_green", justify="right")
        wd_table.add_column("Interpretation", style="yellow")

        distance = wd_result['joint_distance']
        interpretation = "Identical" if distance < 0.01 else "Very Similar" if distance < 0.05 else "Similar" if distance < 0.1 else "Different"
        
        wd_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
        wd_table.add_row("", "", "Lower is better")

        console.print(wd_table)
    else:
        console.print("❌ Wasserstein Distance failed", style="bold red")

    # Maximum Mean Discrepancy results table
    if "maximum_mean_discrepancy" in metrics and metrics["maximum_mean_discrepancy"]["status"] == "success":
        mmd_result = metrics["maximum_mean_discrepancy"]

        # Get parameters info for display
        params_info = mmd_result["parameters"]
        kernel = mmd_result.get("kernel", "rbf")
        params_display = f"kernel={kernel}"

        # Create MMD results table
        mmd_table = Table(title=f"✅ Maximum Mean Discrepancy Results ({params_display})", show_header=True, header_style="bold blue")
        mmd_table.add_column("Metric", style="cyan", no_wrap=True)
        mmd_table.add_column("Score", style="bright_green", justify="right")
        mmd_table.add_column("Interpretation", style="yellow")

        distance = mmd_result['joint_distance']
        interpretation = "Identical" if distance < 0.001 else "Very Similar" if distance < 0.01 else "Similar" if distance < 0.1 else "Different"
        
        mmd_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
        mmd_table.add_row("", "", "Lower is better")

        console.print(mmd_table)
    else:
        console.print("❌ Maximum Mean Discrepancy failed", style="bold red")

    # Jensen-Shannon Distance (Synthcity) results table
    if "jensenshannon_synthcity" in metrics and metrics["jensenshannon_synthcity"]["status"] == "success":
        jsd_sc_result = metrics["jensenshannon_synthcity"]

        # Get parameters info for display
        params_info = jsd_sc_result["parameters"]
        normalize = jsd_sc_result.get("normalize", True)
        n_histogram_bins = jsd_sc_result.get("n_histogram_bins", 10)
        params_display = f"normalize={normalize}, n_histogram_bins={n_histogram_bins}"

        # Create JSD Synthcity results table
        jsd_sc_table = Table(title=f"✅ Jensen-Shannon Distance (Synthcity) Results ({params_display})", show_header=True, header_style="bold blue")
        jsd_sc_table.add_column("Metric", style="cyan", no_wrap=True)
        jsd_sc_table.add_column("Score", style="bright_green", justify="right")
        jsd_sc_table.add_column("Interpretation", style="yellow")

        distance = jsd_sc_result['distance_score']
        interpretation = "Identical" if distance < 0.01 else "Very Similar" if distance < 0.05 else "Similar" if distance < 0.1 else "Different"
        
        jsd_sc_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
        jsd_sc_table.add_row("", "", "Lower is better")

        console.print(jsd_sc_table)
    else:
        console.print("❌ Jensen-Shannon Distance (Synthcity) failed", style="bold red")

    # Jensen-Shannon Distance (SYNDAT) results table
    if "jensenshannon_syndat" in metrics and metrics["jensenshannon_syndat"]["status"] == "success":
        jsd_sd_result = metrics["jensenshannon_syndat"]

        # Get parameters info for display
        params_info = jsd_sd_result["parameters"]
        n_unique_threshold = jsd_sd_result.get("n_unique_threshold", 10)
        params_display = f"n_unique_threshold={n_unique_threshold}"

        # Create JSD SYNDAT results table
        jsd_sd_table = Table(title=f"✅ Jensen-Shannon Distance (SYNDAT) Results ({params_display})", show_header=True, header_style="bold blue")
        jsd_sd_table.add_column("Metric", style="cyan", no_wrap=True)
        jsd_sd_table.add_column("Score", style="bright_green", justify="right")
        jsd_sd_table.add_column("Interpretation", style="yellow")

        distance = jsd_sd_result['distance_score']
        interpretation = "Identical" if distance < 0.01 else "Very Similar" if distance < 0.05 else "Similar" if distance < 0.1 else "Different"
        
        jsd_sd_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
        jsd_sd_table.add_row("", "", "Lower is better")

        console.print(jsd_sd_table)
    else:
        console.print("❌ Jensen-Shannon Distance (SYNDAT) failed", style="bold red")

    # Jensen-Shannon Distance (NannyML) results table
    if "jensenshannon_nannyml" in metrics and metrics["jensenshannon_nannyml"]["status"] == "success":
        jsd_nm_result = metrics["jensenshannon_nannyml"]

        # Get parameters info for display
        params_info = jsd_nm_result["parameters"]
        params_display = str(params_info) if params_info else "adaptive binning (Doane's formula)"

        # Create JSD NannyML results table
        jsd_nm_table = Table(title=f"✅ Jensen-Shannon Distance (NannyML) Results ({params_display})", show_header=True, header_style="bold blue")
        jsd_nm_table.add_column("Metric", style="cyan", no_wrap=True)
        jsd_nm_table.add_column("Score", style="bright_green", justify="right")
        jsd_nm_table.add_column("Interpretation", style="yellow")

        distance = jsd_nm_result['distance_score']
        interpretation = "Identical" if distance < 0.01 else "Very Similar" if distance < 0.05 else "Similar" if distance < 0.1 else "Different"

        jsd_nm_table.add_row("Joint Distance", f"{distance:.6f}", interpretation)
        jsd_nm_table.add_row("", "", "Lower is better")

        console.print(jsd_nm_table)
    else:
        console.print("❌ Jensen-Shannon Distance (NannyML) failed", style="bold red")

    # SDMetrics Quality Report - Comprehensive Display
    if "sdmetrics_quality" in metrics and metrics["sdmetrics_quality"]["status"] == "success":
        sq_result = metrics["sdmetrics_quality"]

        # Helper functions for comprehensive display
        def get_score_color(score: float) -> str:
            """Return color based on score threshold."""
            if pd.isna(score):
                return "dim"
            elif score >= 0.8:
                return "green"
            elif score >= 0.6:
                return "yellow"
            else:
                return "red"

        def get_confidence_flag(n_samples: int) -> str:
            """Return confidence flag based on sample size."""
            if n_samples >= 1000:
                return '✓'
            elif n_samples >= 500:
                return '⚠'
            elif n_samples >= 50:
                return '✗'
            else:
                return '-'

        def get_diagnostic_flag(real_corr: float, synth_corr: float, quality_score: float) -> str:
            """Generate diagnostic flag based on three metrics."""
            if pd.isna(real_corr) or pd.isna(synth_corr) or pd.isna(quality_score):
                return '-'
            if real_corr >= 0.70 and quality_score >= 0.80:
                return '✓'
            if real_corr < 0.30 and synth_corr < 0.30:
                return '✓'
            if real_corr >= 0.50 and quality_score >= 0.65:
                return '✓'
            if real_corr >= 0.70 and quality_score < 0.60:
                return '✗'
            return '⚠'

        # Get parameters
        params_info = sq_result["parameters"]
        max_display = params_info.get("max_display_cols", 10)

        # 1. Column Shapes Details
        column_shapes_details = sq_result.get("column_shapes_details", [])
        if column_shapes_details:
            console.print(f"\n[bold cyan]{'═' * 100}[/bold cyan]")
            console.print("[bold cyan]Column Shapes Details (Distribution Similarity)[/bold cyan]")
            console.print(f"[bold cyan]{'═' * 100}[/bold cyan]\n")

            shapes_table = Table(show_header=True, header_style="bold cyan")
            shapes_table.add_column("Column", style="dim white", width=25)
            shapes_table.add_column("Metric", style="magenta", width=15)
            shapes_table.add_column("Score", justify="center", width=12)
            shapes_table.add_column("Interpretation", style="dim", width=50)

            for detail in column_shapes_details:
                col_name = detail.get("Column", "")
                metric = detail.get("Metric", "")
                score = detail.get("Score")

                if score is None or pd.isna(score):
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

                shapes_table.add_row(
                    col_name,
                    metric,
                    f"[{color}]{score:.3f}[/{color}]" if score is not None else "[dim]N/A[/dim]",
                    interpretation,
                )

            console.print(shapes_table)

        # 2. Quality Report Scores
        console.print(f"\n[bold cyan]{'═' * 70}[/bold cyan]")
        console.print("[bold cyan]Quality Report Scores (Real vs Synthetic)[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 70}[/bold cyan]\n")

        scores_table = Table(show_header=True, header_style="bold cyan")
        scores_table.add_column("Property", style="dim white")
        scores_table.add_column("Score", justify="center")

        property_scores = sq_result.get("property_scores", {})
        overall_score = sq_result.get("score", 0.0)

        for prop_name, prop_key in [("Column Shapes", "column_shapes"), ("Column Pair Trends", "column_pair_trends")]:
            if prop_key in property_scores:
                score = property_scores[prop_key]
                color = get_score_color(score)
                scores_table.add_row(prop_name, f"[{color}]{score:.2%}[/{color}]")

        overall_color = get_score_color(overall_score)
        scores_table.add_row(
            "[bold]Overall Quality Score[/bold]",
            f"[bold {overall_color}]{overall_score:.2%}[/bold {overall_color}]",
        )

        console.print(scores_table)

        # Display null value info
        null_values = sq_result.get("null_values", {})
        if null_values:
            real_nulls = null_values.get("real_data", {})
            synth_nulls = null_values.get("synthetic_data", {})
            total_real = sum(real_nulls.values()) if real_nulls else 0
            total_synth = sum(synth_nulls.values()) if synth_nulls else 0
            if total_real > 0 or total_synth > 0:
                console.print(f"\n[dim]Null values: Real={total_real:,}, Synthetic={total_synth:,}[/dim]")

        # 3. Display Correlation Matrices
        matrices = sq_result.get("matrices", {})
        if matrices:
            from rich.text import Text

            # Reconstruct matrices from dicts
            real_corr = pd.DataFrame(matrices.get("real_correlations", {}))
            real_samples = pd.DataFrame(matrices.get("real_sample_sizes", {}))
            synth_corr = pd.DataFrame(matrices.get("synthetic_correlations", {}))
            quality_matrix = pd.DataFrame(matrices.get("quality_scores", {}))

            # Confidence levels legend
            console.print(f"\n[bold cyan]Correlation Confidence Levels:[/bold cyan]")
            console.print("  ✓ = ≥1000 samples (high confidence)")
            console.print("  ⚠ = 500-1000 samples (moderate confidence)")
            console.print("  ✗ = 50-499 samples (low confidence)")
            console.print("  - = <50 samples (insufficient data)")

            # Display each matrix
            for matrix_data, title in [
                (real_corr, "Matrix 1: Real Data Correlations (with pairwise deletion)"),
                (synth_corr, "Matrix 2: Synthetic Data Correlations"),
                (quality_matrix, "Matrix 3: Quality Scores (Correlation Preservation)")
            ]:
                if not matrix_data.empty:
                    console.print(f"\n[bold cyan]{'═' * 100}[/bold cyan]")
                    console.print(f"[bold cyan]{title}[/bold cyan]")
                    console.print(f"[bold cyan]{'═' * 100}[/bold cyan]\n")

                    cols = sorted(matrix_data.columns)
                    display_cols = cols[:max_display]

                    matrix_table = Table(show_header=True, header_style="dim cyan", width=160)
                    matrix_table.add_column("", style="bold dim white", width=18)

                    for col in display_cols:
                        matrix_table.add_column(col[:11], width=13, justify="center")

                    for row_col in display_cols:
                        row_data = [Text(row_col[:11], style="bold dim white")]

                        for col_col in display_cols:
                            if row_col in matrix_data.index and col_col in matrix_data.columns:
                                corr = matrix_data.loc[row_col, col_col]

                                if pd.isna(corr):
                                    row_data.append(Text("    -     ", style="dim"))
                                else:
                                    color = get_score_color(corr)
                                    # For real correlation matrix, show confidence flags
                                    if "Real Data" in title and row_col in real_samples.index and col_col in real_samples.columns:
                                        n_samples = real_samples.loc[row_col, col_col]
                                        confidence = get_confidence_flag(int(n_samples)) if not pd.isna(n_samples) else '-'
                                        row_data.append(Text(f"{corr:.3f}({confidence})", style=color, justify="center"))
                                    else:
                                        row_data.append(Text(f"{corr:.3f}", style=color, justify="center"))
                            else:
                                row_data.append(Text("    -     ", style="dim"))

                        matrix_table.add_row(*row_data)

                    console.print(matrix_table)

                    if len(cols) > max_display:
                        console.print(f"[dim]... and {len(cols) - max_display} more columns[/dim]")

        # 4. Diagnostic Analysis
        diagnostics = sq_result.get("diagnostics", {})
        if diagnostics:
            console.print(f"\n[bold cyan]{'═' * 140}[/bold cyan]")
            console.print("[bold cyan]Diagnostic Analysis: Correlation Preservation (Bottom 10 Pairs)[/bold cyan]")
            console.print(f"[bold cyan]{'═' * 140}[/bold cyan]\n")

            diag_table = Table(show_header=True, header_style="bold cyan", width=160)
            diag_table.add_column("Column Pair", style="dim white", width=28)
            diag_table.add_column("Real Corr", justify="center", width=12)
            diag_table.add_column("n_samples", justify="center", width=10)
            diag_table.add_column("Synth Corr", justify="center", width=12)
            diag_table.add_column("Quality", justify="center", width=10)
            diag_table.add_column("Status", justify="center", width=8)
            diag_table.add_column("Diagnosis", style="dim", width=60)

            # Sort by quality score
            pairs_list = [(k, v) for k, v in diagnostics.items()]
            pairs_sorted = sorted(pairs_list, key=lambda x: x[1].get("quality_score", 1.0))

            # Show bottom 10
            for pair_key, pair_data in pairs_sorted[:10]:
                real_corr = pair_data.get("real_correlation")
                synth_corr = pair_data.get("synthetic_correlation")
                quality = pair_data.get("quality_score")
                n_samples = pair_data.get("real_n_samples", 0)

                real_str = f"{real_corr:.3f}" if real_corr is not None else "-"
                synth_str = f"{synth_corr:.3f}" if synth_corr is not None else "-"
                quality_str = f"{quality:.3f}" if quality is not None else "-"

                confidence = get_confidence_flag(n_samples) if n_samples else '-'
                flag = get_diagnostic_flag(real_corr or 0, synth_corr or 0, quality or 0)
                flag_color = "green" if flag == "✓" else "yellow" if flag == "⚠" else "red" if flag == "✗" else "dim"

                # Generate diagnosis
                if real_corr is None or synth_corr is None or quality is None:
                    diagnosis = "Insufficient data"
                elif real_corr >= 0.75 and quality >= 0.75:
                    diagnosis = "Strong relationship preserved"
                elif real_corr < 0.25 and synth_corr < 0.25:
                    diagnosis = "Weak relationship reproduced"
                elif real_corr >= 0.70 and quality < 0.50:
                    diagnosis = "Strong trend lost in synthesis"
                elif confidence in ['✗', '-']:
                    diagnosis = f"Low confidence ({n_samples} samples)"
                else:
                    diagnosis = "Moderate preservation"

                diag_table.add_row(
                    pair_key[:26],
                    real_str,
                    f"{n_samples}" if n_samples > 0 else "-",
                    synth_str,
                    quality_str,
                    Text(flag, style=f"bold {flag_color}"),
                    diagnosis[:55] + ("..." if len(diagnosis) > 55 else ""),
                )

            console.print(diag_table)
    else:
        if "sdmetrics_quality" in metrics:
            console.print("❌ SDMetrics Quality Report failed", style="bold red")

    console.print("\n✅ Statistical metrics evaluation completed", style="bold green")


if __name__ == "__main__":
    main()
