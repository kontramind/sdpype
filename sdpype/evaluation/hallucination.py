"""
Hallucination evaluation metrics for synthetic data quality assessment.

Computes comprehensive quality metrics using DuckDB:
1. DDR (Desirable Diverse Records): Records in population but not in training
2. Plausibility: Records passing all validation rules
3. Hallucination: Records not in population
4. Training copies: Records matching training data

All metrics computed with dual perspectives (unique + total counts).
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

from sdv.metadata import SingleTableMetadata


# ============================================================================
# Query Parsing
# ============================================================================

def parse_query_file(query_file: Path) -> Dict[str, str]:
    """Parse SQL file with @query: markers into dictionary of named queries."""

    with open(query_file, 'r') as f:
        content = f.read()

    queries = {}
    current_query_name = None
    current_query_lines = []

    for line in content.split('\n'):
        # Check for query marker
        if line.strip().startswith('-- @query:'):
            # Save previous query if exists
            if current_query_name:
                queries[current_query_name] = '\n'.join(current_query_lines)

            # Start new query
            current_query_name = line.strip().replace('-- @query:', '').strip()
            current_query_lines = []
        else:
            # Accumulate query lines (skip if no query started yet)
            if current_query_name:
                current_query_lines.append(line)

    # Save last query
    if current_query_name:
        queries[current_query_name] = '\n'.join(current_query_lines)

    return queries


# ============================================================================
# DuckDB Query Execution
# ============================================================================

def execute_validation_queries(
    population: pd.DataFrame,
    training: pd.DataFrame,
    synthetic: pd.DataFrame,
    query_file: Path,
    output_dir: Path = None,
    experiment_name: str = None
) -> Dict[str, Any]:
    """
    Execute unified validation queries using DuckDB.

    Args:
        population: Population dataset
        training: Training dataset
        synthetic: Synthetic dataset
        query_file: Path to SQL query file
        output_dir: Optional directory to save binned CSVs
        experiment_name: Experiment name for CSV filenames

    Returns dictionary with all metrics:
    - synthetic_total_count, synthetic_unique_count
    - ddr_unique_count, ddr_unique_rate_pct, ddr_total_count, ddr_total_rate_pct
    - hallucinated_unique_count, hallucinated_unique_rate_pct, etc.
    - training_copy_unique_count, training_copy_unique_rate_pct, etc.
    - plausible_unique_count, plausible_unique_rate_pct, etc.
    - implausible_unique_count, implausible_unique_rate_pct, etc.
    """

    # Create DuckDB connection and register DataFrames
    con = duckdb.connect()
    con.register('population', population)
    con.register('training', training)
    con.register('synthetic', synthetic)

    # Parse query file
    queries = parse_query_file(query_file)

    if 'summary' not in queries:
        raise ValueError("Query file must contain a 'summary' query marked with '-- @query: summary'")

    # Build hash expression with actual column names
    columns = synthetic.columns.tolist()
    hash_cols = ', '.join([f'"{col}"' for col in columns])

    # Replace template placeholder with actual columns
    summary_query = queries['summary'].replace('{{HASH_COLS}}', hash_cols)

    # Execute query
    print("⚙️  Executing validation queries in DuckDB...")
    result_df = con.execute(summary_query).fetchdf()

    # Convert to dictionary
    result_dict = result_df.to_dict('records')[0]

    # Export binned datasets if output directory is provided
    if output_dir and experiment_name:
        print("📊 Exporting binned datasets for hallucination calculations...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract CTEs from the summary query to access binned tables
        # Split on the final summary SELECT (marked by comment)
        if '-- Final Summary' in summary_query:
            # Split at the final summary comment
            binning_query_base = summary_query.split('-- Final Summary')[0]
        else:
            # Fallback: find the last standalone SELECT statement
            # Look for the pattern of ") \n\n SELECT" which indicates the end of CTEs
            parts = summary_query.split('\n\nSELECT')
            if len(parts) > 1:
                binning_query_base = parts[0]
            else:
                # Fallback: just include everything up to the last closing paren before SELECT
                binning_query_base = summary_query.rsplit('\n\nSELECT', 1)[0]

        # Export population_binned
        pop_query = binning_query_base + "\n\nSELECT * FROM population_binned"
        pop_binned = con.execute(pop_query).fetchdf()
        pop_output = output_dir / f"population_data_for_hallucinations.csv"
        pop_binned.to_csv(pop_output, index=False)
        print(f"   ✓ Saved: {pop_output}")

        # Export training_binned
        train_query = binning_query_base + "\n\nSELECT * FROM training_binned"
        train_binned = con.execute(train_query).fetchdf()
        train_output = output_dir / f"training_data_for_hallucinations.csv"
        train_binned.to_csv(train_output, index=False)
        print(f"   ✓ Saved: {train_output}")

        # Export synthetic_binned (without original columns)
        synth_query = binning_query_base + "\n\nSELECT GENDER_CAT, ETHNICITY_CAT, ADMISSION_CAT, READMISSION_CAT, HR_BIN, SYSBP_BIN, DIASBP_BIN FROM synthetic_binned"
        synth_binned = con.execute(synth_query).fetchdf()
        synth_output = output_dir / f"synthetic_data_{experiment_name}_for_hallucinations.csv"
        synth_binned.to_csv(synth_output, index=False)
        print(f"   ✓ Saved: {synth_output}")

    # Close connection
    con.close()

    return result_dict


# ============================================================================
# Dataset Complexity Computation
# ============================================================================

def compute_dataset_complexity(
    df: pd.DataFrame,
    metadata: SingleTableMetadata,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Compute combinatorial complexity of a dataset.

    Formula: c = sum(log(cardinality_i)) for all columns
    This captures the combinatorial search space - total possible unique records.

    Args:
        df: DataFrame to analyze
        metadata: SDV metadata for column type information
        dataset_name: Name for logging/identification

    Returns:
        Dictionary with total complexity and per-column breakdown sorted by contribution
    """

    column_contributions = []

    for col in df.columns:
        cardinality = df[col].nunique()

        # Handle edge cases
        if cardinality <= 0:
            log_card = 0.0
        else:
            log_card = np.log(cardinality)

        column_contributions.append({
            'column': col,
            'cardinality': int(cardinality),
            'log_cardinality': float(log_card)
        })

    # Sort by contribution (descending)
    column_contributions.sort(key=lambda x: x['log_cardinality'], reverse=True)

    # Compute total complexity
    total_complexity = sum(item['log_cardinality'] for item in column_contributions)

    return {
        'dataset_name': dataset_name,
        'total_complexity': float(total_complexity),
        'num_columns': len(df.columns),
        'num_records': len(df),
        'column_contributions': column_contributions
    }


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_hallucination_metrics(
    population: pd.DataFrame,
    training: pd.DataFrame,
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: SingleTableMetadata,
    query_file: Path,
    experiment_name: str
) -> Dict[str, Any]:
    """
    Evaluate hallucination metrics using DuckDB-based validation.

    Args:
        population: Population dataset (decoded)
        training: Training dataset (decoded)
        reference: Reference dataset (decoded)
        synthetic: Synthetic dataset (decoded)
        metadata: SDV metadata
        query_file: Path to SQL validation queries
        experiment_name: Experiment identifier

    Returns:
        Complete hallucination metrics results
    """

    print(f"Evaluating hallucination metrics for experiment: {experiment_name}")
    print(f"Population shape: {population.shape}")
    print(f"Training shape: {training.shape}")
    print(f"Reference shape: {reference.shape}")
    print(f"Synthetic shape: {synthetic.shape}")

    # Validate query file exists
    if not query_file.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")

    # Execute validation queries
    try:
        raw_results = execute_validation_queries(
            population=population,
            training=training,
            synthetic=synthetic,
            query_file=query_file,
            output_dir=Path("experiments/data/binned"),
            experiment_name=experiment_name
        )
    except Exception as e:
        raise RuntimeError(f"Failed to execute validation queries: {str(e)}")

    # Structure results with metadata
    results = {
        "metadata": {
            "experiment_name": experiment_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "population_shape": list(population.shape),
            "training_shape": list(training.shape),
            "synthetic_shape": list(synthetic.shape),
            "evaluation_type": "hallucination_metrics",
            "query_file": str(query_file)
        },
        "population_statistics": {
            "total_count": int(raw_results['population_total_count']),
            "unique_count": int(raw_results['population_unique_count'])
        },
        "training_statistics": {
            "total_count": int(raw_results['training_total_count']),
            "unique_count": int(raw_results['training_unique_count'])
        },
        "synthetic_statistics": {
            "total_count": int(raw_results['synthetic_total_count']),
            "unique_count": int(raw_results['synthetic_unique_count'])
        },
        "ddr_metrics": {
            "unique": {
                "count": int(raw_results['ddr_unique_count']),
                "rate_pct": float(raw_results['ddr_unique_rate_pct'])
            },
            "total": {
                "count": int(raw_results['ddr_total_count']),
                "rate_pct": float(raw_results['ddr_total_rate_pct'])
            }
        },
        "training_copy_valid_metrics": {
            "unique": {
                "count": int(raw_results['training_copy_valid_unique_count']),
                "rate_pct": float(raw_results['training_copy_valid_unique_rate_pct'])
            },
            "total": {
                "count": int(raw_results['training_copy_valid_total_count']),
                "rate_pct": float(raw_results['training_copy_valid_total_rate_pct'])
            }
        },
        "training_copy_propagation_metrics": {
            "unique": {
                "count": int(raw_results['training_copy_propagation_unique_count']),
                "rate_pct": float(raw_results['training_copy_propagation_unique_rate_pct'])
            },
            "total": {
                "count": int(raw_results['training_copy_propagation_total_count']),
                "rate_pct": float(raw_results['training_copy_propagation_total_rate_pct'])
            }
        },
        "new_hallucination_metrics": {
            "unique": {
                "count": int(raw_results['new_hallucination_unique_count']),
                "rate_pct": float(raw_results['new_hallucination_unique_rate_pct'])
            },
            "total": {
                "count": int(raw_results['new_hallucination_total_count']),
                "rate_pct": float(raw_results['new_hallucination_total_rate_pct'])
            }
        },
        "plausibility_metrics": {
            "unique": {
                "count": int(raw_results['plausible_unique_count']),
                "rate_pct": float(raw_results['plausible_unique_rate_pct'])
            },
            "total": {
                "count": int(raw_results['plausible_total_count']),
                "rate_pct": float(raw_results['plausible_total_rate_pct'])
            }
        },
        "implausibility_metrics": {
            "unique": {
                "count": int(raw_results['implausible_unique_count']),
                "rate_pct": float(raw_results['implausible_unique_rate_pct'])
            },
            "total": {
                "count": int(raw_results['implausible_total_count']),
                "rate_pct": float(raw_results['implausible_total_rate_pct'])
            }
        },
        "quality_matrix_2x2": {
            "total_factual": {
                "count": int(raw_results['total_factual_count']),
                "rate_pct": float(raw_results['total_factual_rate_pct'])
            },
            "novel_factual": {
                "count": int(raw_results['novel_factual_count']),
                "rate_pct": float(raw_results['novel_factual_rate_pct'])
            },
            "total_plausible": {
                "count": int(raw_results['total_plausible_count']),
                "rate_pct": float(raw_results['total_plausible_rate_pct'])
            },
            "novel_plausible": {
                "count": int(raw_results['novel_plausible_count']),
                "rate_pct": float(raw_results['novel_plausible_rate_pct'])
            }
        },
        "cross_tabulation": {
            "ddr": {
                "plausible": {
                    "count": int(raw_results['ddr_plausible_total_count']),
                    "rate_pct": float(raw_results['ddr_plausible_total_rate_pct'])
                },
                "implausible": {
                    "count": int(raw_results['ddr_implausible_total_count']),
                    "rate_pct": float(raw_results['ddr_implausible_total_rate_pct'])
                }
            },
            "training_copy_valid": {
                "plausible": {
                    "count": int(raw_results['training_copy_valid_plausible_total_count']),
                    "rate_pct": float(raw_results['training_copy_valid_plausible_total_rate_pct'])
                },
                "implausible": {
                    "count": int(raw_results['training_copy_valid_implausible_total_count']),
                    "rate_pct": float(raw_results['training_copy_valid_implausible_total_rate_pct'])
                }
            },
            "training_copy_propagation": {
                "plausible": {
                    "count": int(raw_results['training_copy_propagation_plausible_total_count']),
                    "rate_pct": float(raw_results['training_copy_propagation_plausible_total_rate_pct'])
                },
                "implausible": {
                    "count": int(raw_results['training_copy_propagation_implausible_total_count']),
                    "rate_pct": float(raw_results['training_copy_propagation_implausible_total_rate_pct'])
                }
            },
            "new_hallucination": {
                "plausible": {
                    "count": int(raw_results['new_hallucination_plausible_total_count']),
                    "rate_pct": float(raw_results['new_hallucination_plausible_total_rate_pct'])
                },
                "implausible": {
                    "count": int(raw_results['new_hallucination_implausible_total_count']),
                    "rate_pct": float(raw_results['new_hallucination_implausible_total_rate_pct'])
                }
            }
        }
    }

    # Compute complexity for all datasets
    print("Computing dataset complexity metrics...")
    complexity_population = compute_dataset_complexity(population, metadata, "Population")
    complexity_training = compute_dataset_complexity(training, metadata, "Training")
    complexity_reference = compute_dataset_complexity(reference, metadata, "Reference")
    complexity_synthetic = compute_dataset_complexity(synthetic, metadata, "Synthetic")

    # Add complexity results
    results["complexity_metrics"] = {
        "population": complexity_population,
        "training": complexity_training,
        "reference": complexity_reference,
        "synthetic": complexity_synthetic,
        "comparisons": {
            "synthetic_vs_population_ratio": complexity_synthetic["total_complexity"] / complexity_population["total_complexity"] if complexity_population["total_complexity"] > 0 else 0.0,
            "synthetic_vs_training_ratio": complexity_synthetic["total_complexity"] / complexity_training["total_complexity"] if complexity_training["total_complexity"] > 0 else 0.0,
            "synthetic_vs_reference_ratio": complexity_synthetic["total_complexity"] / complexity_reference["total_complexity"] if complexity_reference["total_complexity"] > 0 else 0.0,
            "training_vs_population_ratio": complexity_training["total_complexity"] / complexity_population["total_complexity"] if complexity_population["total_complexity"] > 0 else 0.0,
        }
    }

    return results


# ============================================================================
# Report Generation
# ============================================================================

def generate_hallucination_report(results: Dict[str, Any]) -> str:
    """Generate human-readable text report from hallucination evaluation results."""

    metadata = results.get("metadata", {})
    pop_stats = results.get("population_statistics", {})
    train_stats = results.get("training_statistics", {})
    synth_stats = results.get("synthetic_statistics", {})
    ddr = results.get("ddr_metrics", {})
    train_copy_valid = results.get("training_copy_valid_metrics", {})
    train_copy_prop = results.get("training_copy_propagation_metrics", {})
    new_halluc = results.get("new_hallucination_metrics", {})
    plaus = results.get("plausibility_metrics", {})
    implaus = results.get("implausibility_metrics", {})
    cross_tab = results.get("cross_tabulation", {})

    report_lines = [
        "=" * 80,
        "HALLUCINATION EVALUATION REPORT",
        "DDR + Plausibility Validation",
        "=" * 80,
        "",
        f"Experiment: {metadata.get('experiment_name', 'N/A')}",
        f"Timestamp: {metadata.get('evaluation_timestamp', 'N/A')}",
        f"Query File: {metadata.get('query_file', 'N/A')}",
        "",
        "-" * 80,
        "DATASET STATISTICS",
        "-" * 80,
        "",
        f"Population Dataset: {pop_stats.get('total_count', 0):,} rows ({pop_stats.get('unique_count', 0):,} unique)",
        f"Training Dataset:   {train_stats.get('total_count', 0):,} rows ({train_stats.get('unique_count', 0):,} unique)",
        f"Synthetic Dataset:  {synth_stats.get('total_count', 0):,} rows ({synth_stats.get('unique_count', 0):,} unique)",
        "",
        "-" * 80,
        "DDR METRICS (Desirable Diverse Records)",
        "Records in population but NOT in training - IDEAL for synthesis",
        "-" * 80,
        "",
        f"Unique Count: {ddr.get('unique', {}).get('count', 0):,} ({ddr.get('unique', {}).get('rate_pct', 0):.2f}%)",
        f"Total Count:  {ddr.get('total', {}).get('count', 0):,} ({ddr.get('total', {}).get('rate_pct', 0):.2f}%)",
        f"Interpretation: Factual AND Novel (IDEAL)",
        "",
        "-" * 80,
        "TRAINING COPY (VALID) METRICS",
        "Records copied from training AND exist in population - Real data memorization",
        "-" * 80,
        "",
        f"Unique Count: {train_copy_valid.get('unique', {}).get('count', 0):,} ({train_copy_valid.get('unique', {}).get('rate_pct', 0):.2f}%)",
        f"Total Count:  {train_copy_valid.get('total', {}).get('count', 0):,} ({train_copy_valid.get('total', {}).get('rate_pct', 0):.2f}%)",
        f"Interpretation: Gen 0 = Privacy risk | Gen 1+ = Overfitting to real records",
        "",
        "-" * 80,
        "TRAINING COPY (PROPAGATION) METRICS",
        "Records copied from training but NOT in population - Hallucination propagation!",
        "-" * 80,
        "",
        f"Unique Count: {train_copy_prop.get('unique', {}).get('count', 0):,} ({train_copy_prop.get('unique', {}).get('rate_pct', 0):.2f}%)",
        f"Total Count:  {train_copy_prop.get('total', {}).get('count', 0):,} ({train_copy_prop.get('total', {}).get('rate_pct', 0):.2f}%)",
        f"Interpretation: Mode collapse + hallucination persistence across generations",
        "",
        "-" * 80,
        "NEW HALLUCINATION METRICS",
        "Records NOT in population AND NOT in training - Freshly fabricated",
        "-" * 80,
        "",
        f"Unique Count: {new_halluc.get('unique', {}).get('count', 0):,} ({new_halluc.get('unique', {}).get('rate_pct', 0):.2f}%)",
        f"Total Count:  {new_halluc.get('total', {}).get('count', 0):,} ({new_halluc.get('total', {}).get('rate_pct', 0):.2f}%)",
        f"Interpretation: Model generating invalid data (not from training)",
        "",
        "-" * 80,
        "PLAUSIBILITY METRICS",
        "Records passing all validation rules",
        "-" * 80,
        "",
        f"Unique Count: {plaus.get('unique', {}).get('count', 0):,} ({plaus.get('unique', {}).get('rate_pct', 0):.2f}%)",
        f"Total Count:  {plaus.get('total', {}).get('count', 0):,} ({plaus.get('total', {}).get('rate_pct', 0):.2f}%)",
        f"Interpretation: Passes all validation rules",
        "",
        "-" * 80,
        "IMPLAUSIBILITY METRICS",
        "Records failing validation rules",
        "-" * 80,
        "",
        f"Unique Count: {implaus.get('unique', {}).get('count', 0):,} ({implaus.get('unique', {}).get('rate_pct', 0):.2f}%)",
        f"Total Count:  {implaus.get('total', {}).get('count', 0):,} ({implaus.get('total', {}).get('rate_pct', 0):.2f}%)",
        f"Interpretation: Fails validation rules",
        "",
        "-" * 80,
        "CROSS-TABULATION: Category × Plausibility (Total perspective)",
        "-" * 80,
        "",
        "DDR (Desirable Diverse):",
        f"  Plausible:   {cross_tab.get('ddr', {}).get('plausible', {}).get('count', 0):,} ({cross_tab.get('ddr', {}).get('plausible', {}).get('rate_pct', 0):.2f}%)",
        f"  Implausible: {cross_tab.get('ddr', {}).get('implausible', {}).get('count', 0):,} ({cross_tab.get('ddr', {}).get('implausible', {}).get('rate_pct', 0):.2f}%)",
        "",
        "Training Copy (Valid):",
        f"  Plausible:   {cross_tab.get('training_copy_valid', {}).get('plausible', {}).get('count', 0):,} ({cross_tab.get('training_copy_valid', {}).get('plausible', {}).get('rate_pct', 0):.2f}%)",
        f"  Implausible: {cross_tab.get('training_copy_valid', {}).get('implausible', {}).get('count', 0):,} ({cross_tab.get('training_copy_valid', {}).get('implausible', {}).get('rate_pct', 0):.2f}%)",
        "",
        "Training Copy (Propagation):",
        f"  Plausible:   {cross_tab.get('training_copy_propagation', {}).get('plausible', {}).get('count', 0):,} ({cross_tab.get('training_copy_propagation', {}).get('plausible', {}).get('rate_pct', 0):.2f}%)",
        f"  Implausible: {cross_tab.get('training_copy_propagation', {}).get('implausible', {}).get('count', 0):,} ({cross_tab.get('training_copy_propagation', {}).get('implausible', {}).get('rate_pct', 0):.2f}%)",
        "",
        "New Hallucination:",
        f"  Plausible:   {cross_tab.get('new_hallucination', {}).get('plausible', {}).get('count', 0):,} ({cross_tab.get('new_hallucination', {}).get('plausible', {}).get('rate_pct', 0):.2f}%)",
        f"  Implausible: {cross_tab.get('new_hallucination', {}).get('implausible', {}).get('count', 0):,} ({cross_tab.get('new_hallucination', {}).get('implausible', {}).get('rate_pct', 0):.2f}%)",
        "",
        "=" * 80,
        "QUALITY SUMMARY (Categories are mutually exclusive - sum = 100%)",
        "=" * 80,
        "",
        f"✓ DDR (Ideal):                   {ddr.get('unique', {}).get('rate_pct', 0):.2f}% (unique) | {ddr.get('total', {}).get('rate_pct', 0):.2f}% (total)",
        f"⚠ Training Copies (Valid):       {train_copy_valid.get('unique', {}).get('rate_pct', 0):.2f}% (unique) | {train_copy_valid.get('total', {}).get('rate_pct', 0):.2f}% (total)",
        f"⚠ Training Copies (Propagation): {train_copy_prop.get('unique', {}).get('rate_pct', 0):.2f}% (unique) | {train_copy_prop.get('total', {}).get('rate_pct', 0):.2f}% (total)",
        f"✗ New Hallucinations:            {new_halluc.get('unique', {}).get('rate_pct', 0):.2f}% (unique) | {new_halluc.get('total', {}).get('rate_pct', 0):.2f}% (total)",
        "",
        "PLAUSIBILITY (Orthogonal measure - not part of sum above):",
        f"✓ Plausible:                     {plaus.get('unique', {}).get('rate_pct', 0):.2f}% (unique) | {plaus.get('total', {}).get('rate_pct', 0):.2f}% (total)",
        f"✗ Implausible:                   {implaus.get('unique', {}).get('rate_pct', 0):.2f}% (unique) | {implaus.get('total', {}).get('rate_pct', 0):.2f}% (total)",
        "",
        "=" * 80,
    ]

    return "\n".join(report_lines)
