#!/usr/bin/env python3
"""
Test MIMIC-III validation SQL with generated test datasets.

Loads test CSVs into DuckDB and executes validation queries.
"""

import duckdb
import pandas as pd
from pathlib import Path

# Paths
TEST_DATA_DIR = Path(__file__).parent / "data" / "mimic_iii_validation"
VALIDATION_SQL = Path(__file__).parent.parent / "queries" / "mimic_iii_validation.sql"


def test_validation_sql(case_name="trivial"):
    """
    Load test datasets and run validation SQL.

    Args:
        case_name: Name of test case (e.g., 'trivial')
    """
    print(f"Testing MIMIC-III validation SQL with '{case_name}' case...")
    print()

    # File paths
    population_csv = TEST_DATA_DIR / f"population_{case_name}.csv"
    training_csv = TEST_DATA_DIR / f"training_{case_name}.csv"
    synthetic_csv = TEST_DATA_DIR / f"synthetic_{case_name}.csv"

    # Check files exist
    for csv_path in [population_csv, training_csv, synthetic_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Test data not found: {csv_path}")

    if not VALIDATION_SQL.exists():
        raise FileNotFoundError(f"Validation SQL not found: {VALIDATION_SQL}")

    # Load validation SQL
    with open(VALIDATION_SQL, 'r') as f:
        sql_content = f.read()

    # Extract summary query (everything after "-- @query: summary")
    if "-- @query: summary" in sql_content:
        summary_query = sql_content.split("-- @query: summary")[1].strip()
    else:
        summary_query = sql_content

    # Create DuckDB connection
    con = duckdb.connect(":memory:")

    # Load CSVs as tables
    print("Loading test datasets into DuckDB...")
    population_df = pd.read_csv(population_csv)
    training_df = pd.read_csv(training_csv)
    synthetic_df = pd.read_csv(synthetic_csv)

    con.register("population", population_df)
    con.register("training", training_df)
    con.register("synthetic", synthetic_df)

    print(f"  Population: {len(population_df)} rows")
    print(f"  Training: {len(training_df)} rows")
    print(f"  Synthetic: {len(synthetic_df)} rows")
    print()

    # Execute validation query
    print("Executing validation SQL...")
    try:
        result_df = con.execute(summary_query).fetchdf()
        print("✓ Query executed successfully!")
        print()

        # Print key metrics
        print("=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print()

        # Dataset counts
        print("Dataset Statistics:")
        print(f"  Population Total:   {result_df['population_total_count'].iloc[0]:>6}")
        print(f"  Training Total:     {result_df['training_total_count'].iloc[0]:>6}")
        print(f"  Synthetic Total:    {result_df['synthetic_total_count'].iloc[0]:>6}")
        print()

        # Factuality metrics
        print("Factuality Metrics (Total Perspective):")
        print(f"  DDR:                {result_df['ddr_total_count'].iloc[0]:>6}  ({result_df['ddr_total_rate_pct'].iloc[0]:>6.2f}%)")
        print(f"  Training Copy:      {result_df['training_copy_valid_total_count'].iloc[0]:>6}  ({result_df['training_copy_valid_total_rate_pct'].iloc[0]:>6.2f}%)")
        print(f"  New Hallucination:  {result_df['new_hallucination_total_count'].iloc[0]:>6}  ({result_df['new_hallucination_total_rate_pct'].iloc[0]:>6.2f}%)")
        print()

        # Plausibility metrics
        print("Plausibility Metrics:")
        print(f"  Plausible:          {result_df['plausible_total_count'].iloc[0]:>6}  ({result_df['plausible_total_rate_pct'].iloc[0]:>6.2f}%)")
        print(f"  Implausible:        {result_df['implausible_total_count'].iloc[0]:>6}  ({result_df['implausible_total_rate_pct'].iloc[0]:>6.2f}%)")
        print()

        # 2x2 Matrix
        print("2x2 Quality Matrix:")
        print(f"  Total Factual:      {result_df['total_factual_count'].iloc[0]:>6}  ({result_df['total_factual_rate_pct'].iloc[0]:>6.2f}%)")
        print(f"  Novel Factual:      {result_df['novel_factual_count'].iloc[0]:>6}  ({result_df['novel_factual_rate_pct'].iloc[0]:>6.2f}%)")
        print(f"  Total Plausible:    {result_df['total_plausible_count'].iloc[0]:>6}  ({result_df['total_plausible_rate_pct'].iloc[0]:>6.2f}%)")
        print(f"  Novel Plausible:    {result_df['novel_plausible_count'].iloc[0]:>6}  ({result_df['novel_plausible_rate_pct'].iloc[0]:>6.2f}%)")
        print()

        print("=" * 80)
        print()

        # Optionally save full results
        output_path = TEST_DATA_DIR / f"results_{case_name}.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Full results saved to: {output_path}")

        return result_df

    except Exception as e:
        print(f"✗ Query execution failed: {e}")
        raise

    finally:
        con.close()


if __name__ == "__main__":
    test_validation_sql(case_name="trivial")
