"""
Post-processing utilities for synthetic data generation.

This module provides functions to fix invalid categories in synthetic data
by mapping them to valid categories from the training/reference data.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def get_categorical_columns(sdtypes: Dict[str, str]) -> List[str]:
    """
    Extract categorical column names from sdtypes configuration.

    Args:
        sdtypes: Dictionary mapping column names to sdtype strings

    Returns:
        List of categorical column names
    """
    return [col for col, sdtype in sdtypes.items() if sdtype == 'categorical']


def normalize_categorical_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Normalize a categorical column to consistent string dtype.

    This handles cases where generators produce numeric types (int64, float64)
    for categorical columns, ensuring consistent string comparison.

    Args:
        df: DataFrame containing the column
        column: Column name to normalize

    Returns:
        Series with values converted to strings
    """
    series = df[column]

    # Convert to string, handling NaN/None properly
    # This works for int64, float64, object, etc.
    return series.astype(str).replace('nan', pd.NA).replace('None', pd.NA)


def identify_invalid_categories(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    categorical_columns: List[str]
) -> Dict[str, Set]:
    """
    Identify categories in synthetic data that don't exist in reference data.

    Normalizes columns to strings before comparison to handle dtype mismatches
    (e.g., when generators produce int64 for categorical columns).

    Args:
        synthetic_df: Synthetic dataframe to check
        reference_df: Reference dataframe with valid categories
        categorical_columns: List of categorical column names

    Returns:
        Dictionary mapping column name to set of invalid categories
    """
    invalid_categories = {}

    for col in categorical_columns:
        if col not in synthetic_df.columns or col not in reference_df.columns:
            logger.warning(f"Column '{col}' not found in both dataframes, skipping")
            continue

        # Normalize both columns to strings for consistent comparison
        # This handles int64, float64, object dtypes automatically
        ref_normalized = normalize_categorical_column(reference_df, col)
        syn_normalized = normalize_categorical_column(synthetic_df, col)

        valid_cats = set(ref_normalized.dropna().unique())
        synthetic_cats = set(syn_normalized.dropna().unique())
        invalid = synthetic_cats - valid_cats

        if invalid:
            invalid_categories[col] = invalid
            logger.debug(
                f"Column '{col}': Found {len(invalid)} invalid categories "
                f"(ref dtype: {reference_df[col].dtype}, syn dtype: {synthetic_df[col].dtype})"
            )

    return invalid_categories


def fix_invalid_categories_random(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    column: str,
    invalid_mask: pd.Series
) -> pd.Series:
    """
    Replace invalid categories with random valid ones (uniform distribution).

    Args:
        synthetic_df: Synthetic dataframe
        reference_df: Reference dataframe with valid categories
        column: Column name to fix
        invalid_mask: Boolean mask indicating invalid rows

    Returns:
        Series with fixed values (original dtype preserved)
    """
    # Get valid categories (using original dtype from reference)
    valid_cats = reference_df[column].dropna().unique()
    n_invalid = invalid_mask.sum()

    return np.random.choice(valid_cats, size=n_invalid)


def fix_invalid_categories_weighted(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    column: str,
    invalid_mask: pd.Series
) -> pd.Series:
    """
    Replace invalid categories using frequency-weighted sampling from reference.

    Args:
        synthetic_df: Synthetic dataframe
        reference_df: Reference dataframe with valid categories
        column: Column name to fix
        invalid_mask: Boolean mask indicating invalid rows

    Returns:
        Series with fixed values
    """
    # Get valid categories and their frequencies
    valid_freq = reference_df[column].value_counts(normalize=True)
    valid_cats = valid_freq.index.tolist()
    weights = valid_freq.values

    n_invalid = invalid_mask.sum()

    return np.random.choice(valid_cats, size=n_invalid, p=weights)


def fix_invalid_categories_knn(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    column: str,
    invalid_mask: pd.Series,
    categorical_columns: List[str],
    k: int = 5,
    distance_metric: str = 'hamming'
) -> pd.Series:
    """
    Replace invalid categories using K-Nearest Neighbors on other categorical columns.

    Finds K nearest valid rows based on similarity of OTHER categorical columns,
    then samples the target column value from neighbors weighted by distance.

    Args:
        synthetic_df: Synthetic dataframe
        reference_df: Reference dataframe with valid categories
        column: Column name to fix
        invalid_mask: Boolean mask indicating invalid rows
        categorical_columns: List of all categorical columns
        k: Number of neighbors
        distance_metric: Distance metric for KNN (default: 'hamming')

    Returns:
        Series with fixed values
    """
    # Get other columns (exclude target column)
    other_cols = [c for c in categorical_columns
                  if c != column and c in synthetic_df.columns and c in reference_df.columns]

    if len(other_cols) == 0:
        # Fallback to weighted if no other columns available
        logger.warning(f"No other categorical columns available for KNN on '{column}', using weighted fallback")
        return fix_invalid_categories_weighted(synthetic_df, reference_df, column, invalid_mask)

    # Encode all categorical columns for distance computation
    encoders = {}
    ref_encoded = pd.DataFrame(index=reference_df.index)
    syn_encoded = pd.DataFrame(index=synthetic_df.index)

    for col in other_cols:
        le = LabelEncoder()
        # Fit on combined vocabulary to handle edge cases
        combined_vocab = pd.concat([
            reference_df[col].astype(str).fillna('__NA__'),
            synthetic_df[col].astype(str).fillna('__NA__')
        ]).unique()
        le.fit(combined_vocab)
        encoders[col] = le

        # Transform reference and synthetic
        ref_encoded[col] = le.transform(reference_df[col].astype(str).fillna('__NA__'))
        syn_encoded[col] = le.transform(synthetic_df[col].astype(str).fillna('__NA__'))

    # Build KNN index on reference data
    k_actual = min(k, len(reference_df))  # Handle case where reference has fewer rows than k
    knn = NearestNeighbors(n_neighbors=k_actual, metric=distance_metric)
    knn.fit(ref_encoded[other_cols])

    # Get invalid rows
    invalid_indices = synthetic_df[invalid_mask].index
    fixed_values = pd.Series(index=invalid_indices, dtype=object)

    # For each invalid row, find K nearest neighbors
    for idx in invalid_indices:
        # Get features from other columns
        query = syn_encoded.loc[[idx], other_cols]

        # Find K nearest reference rows
        distances, indices = knn.kneighbors(query)
        neighbor_indices = indices[0]

        # Get target column values from neighbors
        neighbor_values = reference_df.iloc[neighbor_indices][column].values

        # Weight by inverse distance (closer neighbors weighted more)
        dist_array = distances[0]
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (dist_array + 1e-6)
        weights = weights / weights.sum()

        # Sample from neighbors
        fixed_values.loc[idx] = np.random.choice(neighbor_values, p=weights)

    return fixed_values


def fix_invalid_categories(
    synthetic_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    categorical_columns: List[str],
    method: str = 'knn',
    knn_neighbors: int = 5,
    distance_metric: str = 'hamming',
    fallback: str = 'weighted'
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Fix all invalid categories in synthetic data.

    This is the main entry point for category fixing. It:
    1. Identifies invalid categories per column
    2. Applies the specified method to fix them
    3. Returns fixed dataframe and metrics

    Args:
        synthetic_df: Synthetic dataframe to fix
        reference_df: Reference dataframe with valid categories
        categorical_columns: List of categorical column names
        method: Fixing method ('knn', 'weighted', 'random', 'none')
        knn_neighbors: Number of neighbors for KNN method
        distance_metric: Distance metric for KNN
        fallback: Fallback method for columns with 100% invalid

    Returns:
        Tuple of (fixed_dataframe, metrics_dict)
        metrics_dict contains: {column: num_fixed_values, ...}
    """
    if method == 'none':
        logger.info("Category fixing disabled (method='none')")
        return synthetic_df.copy(), {}

    # Identify invalid categories
    invalid_categories = identify_invalid_categories(
        synthetic_df, reference_df, categorical_columns
    )

    if not invalid_categories:
        logger.info("✓ No invalid categories found - all synthetic categories exist in reference")
        return synthetic_df.copy(), {}

    # Log summary
    total_invalid_cols = len(invalid_categories)
    logger.info(f"🔧 Fixing invalid categories in {total_invalid_cols} column(s) using method='{method}'")

    # Fix each column
    fixed_df = synthetic_df.copy()
    metrics = {}

    for col, invalid_cats in invalid_categories.items():
        # Normalize column for consistent comparison
        # (invalid_cats was identified using normalized strings)
        col_normalized = normalize_categorical_column(fixed_df, col)

        # Get mask for invalid rows (using normalized values)
        invalid_mask = col_normalized.isin(invalid_cats)
        n_invalid = invalid_mask.sum()

        if n_invalid == 0:
            continue

        # Check if column is 100% invalid (use fallback)
        total_rows = len(fixed_df)
        pct_invalid = (n_invalid / total_rows) * 100

        if pct_invalid == 100.0:
            logger.warning(
                f"Column '{col}': 100% invalid categories, using fallback method '{fallback}'"
            )
            current_method = fallback
        else:
            current_method = method

        # Apply fixing method
        logger.info(
            f"  {col}: {n_invalid}/{total_rows} ({pct_invalid:.1f}%) invalid values "
            f"(method={current_method})"
        )

        try:
            if current_method == 'knn':
                fixed_values = fix_invalid_categories_knn(
                    fixed_df, reference_df, col, invalid_mask,
                    categorical_columns, k=knn_neighbors, distance_metric=distance_metric
                )
            elif current_method == 'weighted':
                fixed_values = fix_invalid_categories_weighted(
                    fixed_df, reference_df, col, invalid_mask
                )
            elif current_method == 'random':
                fixed_values = fix_invalid_categories_random(
                    fixed_df, reference_df, col, invalid_mask
                )
            else:
                raise ValueError(f"Unknown method: {current_method}")

            # Apply fixes
            fixed_df.loc[invalid_mask, col] = fixed_values
            metrics[col] = int(n_invalid)  # Convert numpy int64 to Python int for JSON serialization

        except Exception as e:
            logger.error(f"Error fixing column '{col}': {e}")
            logger.warning(f"Skipping column '{col}', invalid categories will remain")
            metrics[col] = 0  # Mark as not fixed

    # Summary
    total_fixed = sum(metrics.values())
    logger.info(f"✓ Fixed {total_fixed} total invalid category values across {len(metrics)} columns")

    return fixed_df, metrics
