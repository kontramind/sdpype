#!/usr/bin/env python3
"""
Generate test datasets for MIMIC-III validation SQL testing.

Creates population, training, and synthetic datasets with known characteristics
to verify validation SQL logic.
"""

import pandas as pd
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "data" / "mimic_iii_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_trivial_case():
    """
    Generate trivial test case: 100% DDR (all synthetic in population, none in training).

    Expected results:
    - DDR rate: 100%
    - Training copy: 0%
    - Hallucination: 0%
    - Plausibility: 100%
    """
    print("Generating trivial test case (100% DDR)...")

    # Population: 100 rows
    population_data = []
    for i in range(100):
        population_data.append({
            'IS_READMISSION_30D': str(i % 2),  # '0' or '1'
            'AGE': 20 + i,  # 20-119
            'GENDER': ['M', 'F'][i % 2],
            'ETHNICITY_GROUPED': ['white', 'black', 'asian', 'other'][i % 4],
            'ADMISSION_TYPE': ['EMERGENCY', 'ELECTIVE', 'URGENT'][i % 3],
            'HR_FIRST': 60 + (i % 40),  # 60-99
            'SYSBP_FIRST': 100 + (i % 50),  # 100-149
            'DIASBP_FIRST': 60 + (i % 30),  # 60-89
            'RESPRATE_FIRST': 12 + (i % 10),  # 12-21
            'NTPROBNP_FIRST': 100.0 + i * 10,  # 100-1090
            'CREATININE_FIRST': 0.5 + i * 0.01,  # 0.5-1.49
            'BUN_FIRST': 10.0 + i * 0.5,  # 10-59.5
            'POTASSIUM_FIRST': 3.5 + (i % 20) * 0.1,  # 3.5-5.4
            'TOTAL_CHOLESTEROL_FIRST': 150.0 + i * 2,  # 150-348
        })

    population_df = pd.DataFrame(population_data)

    # Training: First 30 rows from population
    training_df = population_df.iloc[:30].copy()

    # Synthetic: Rows 30-79 from population (50 rows, none in training)
    # This creates 100% DDR: all in population, none in training
    synthetic_df = population_df.iloc[30:80].copy()

    # Save datasets
    population_df.to_csv(OUTPUT_DIR / "population_trivial.csv", index=False)
    training_df.to_csv(OUTPUT_DIR / "training_trivial.csv", index=False)
    synthetic_df.to_csv(OUTPUT_DIR / "synthetic_trivial.csv", index=False)

    print(f"  Population: {len(population_df)} rows")
    print(f"  Training: {len(training_df)} rows")
    print(f"  Synthetic: {len(synthetic_df)} rows")
    print(f"  Saved to: {OUTPUT_DIR}")
    print()
    print("Expected validation results:")
    print("  - DDR rate: 100% (50/50)")
    print("  - Training copy: 0% (0/50)")
    print("  - New hallucination: 0% (0/50)")
    print("  - Plausibility: 100% (all values valid)")


if __name__ == "__main__":
    generate_trivial_case()
    print("\n✓ Test data generation complete!")
