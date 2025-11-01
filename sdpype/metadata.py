"""Metadata detection and management for SDV."""

import json
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sdv.metadata import SingleTableMetadata

console = Console()


# Mapping from SDV computer_representation to pandas dtype
NUMERIC_DTYPE_MAP = {
    "Int8": "Int8",
    "Int16": "Int16",
    "Int32": "Int32",
    "Int64": "Int64",
    "UInt8": "UInt8",
    "UInt16": "UInt16",
    "UInt32": "UInt32",
    "UInt64": "UInt64",
    "Float": "float64",
}


def load_csv_with_metadata(
    csv_path: Path,
    metadata_path: Path,
    **read_csv_kwargs
) -> pd.DataFrame:
    """
    Load CSV with proper dtypes enforced from SDV metadata.

    This function ensures type consistency across multiple CSV files by using
    metadata as the single source of truth for column types. This is critical
    for avoiding string/number mismatches when comparing datasets (e.g., in
    ddr_metric or plausible_validator).

    Handles all SDV sdtypes:
    - categorical → str (ensures exact value matching, avoids "591" vs 591)
    - numerical → Int64, Float, etc. (from computer_representation)
    - boolean → bool
    - datetime → parsed with datetime_format
    - id, PII types (email, phone, etc.) → str (for exact comparison)

    Args:
        csv_path: Path to CSV file to load
        metadata_path: Path to SDV metadata JSON file (REQUIRED)
        **read_csv_kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        DataFrame with properly typed columns according to metadata

    Raises:
        FileNotFoundError: If csv_path or metadata_path doesn't exist
        ValueError: If metadata format is invalid

    Example:
        >>> df = load_csv_with_metadata(
        ...     csv_path=Path("data.csv"),
        ...     metadata_path=Path("metadata.json")
        ... )
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)

    # Extract columns metadata
    if "columns" not in metadata_dict:
        raise ValueError(f"Metadata file {metadata_path} missing 'columns' key")

    columns_meta = metadata_dict["columns"]

    # Build dtype mapping and parse configurations
    dtype_map: Dict[str, Any] = {}
    parse_dates_list = []
    date_format_dict: Dict[str, str] = {}
    boolean_columns = []

    for col_name, col_meta in columns_meta.items():
        sdtype = col_meta.get("sdtype")

        if sdtype == "categorical":
            # Force categorical to string to avoid type mismatches
            # (e.g., Region ID could be "591" in one file, 591 in another)
            dtype_map[col_name] = str

        elif sdtype == "numerical":
            # Use computer_representation if available, else float64
            comp_repr = col_meta.get("computer_representation", "Float")
            dtype_map[col_name] = NUMERIC_DTYPE_MAP.get(comp_repr, "float64")

        elif sdtype == "boolean":
            # Mark for post-processing (pandas may read as string)
            boolean_columns.append(col_name)
            # Don't set dtype - let pandas infer, then convert

        elif sdtype == "datetime":
            # Use parse_dates with datetime_format
            parse_dates_list.append(col_name)
            datetime_fmt = col_meta.get("datetime_format")
            if datetime_fmt:
                date_format_dict[col_name] = datetime_fmt

        elif sdtype == "id":
            # IDs should be treated as strings for exact comparison
            dtype_map[col_name] = str

        else:
            # All other sdtypes (PII types like email, phone, etc.)
            # treat as string for exact comparison purposes
            dtype_map[col_name] = str

    # Build read_csv arguments
    csv_kwargs = read_csv_kwargs.copy()

    if dtype_map:
        csv_kwargs["dtype"] = dtype_map

    if parse_dates_list:
        csv_kwargs["parse_dates"] = parse_dates_list

    # For pandas >= 2.0, use date_format parameter
    # For older versions, this will be ignored gracefully
    if date_format_dict:
        try:
            csv_kwargs["date_format"] = date_format_dict
        except TypeError:
            # Fallback for older pandas versions
            console.print(
                "⚠ Warning: date_format not supported in this pandas version. "
                "Datetimes will be parsed without explicit format.",
                style="yellow"
            )

    # Load CSV with proper types
    df = pd.read_csv(csv_path, **csv_kwargs)

    # Post-process boolean columns
    # (pandas might read "True"/"False" strings, need explicit conversion)
    for bool_col in boolean_columns:
        if bool_col in df.columns:
            # Handle common boolean representations
            df[bool_col] = df[bool_col].map({
                'True': True, 'true': True, 'TRUE': True, '1': True, 1: True,
                'False': False, 'false': False, 'FALSE': False, '0': False, 0: False,
                True: True, False: False,
                # Keep NaN as NaN
            })

    return df


def detect_metadata(
    data_path: Path,
    output_path: Optional[Path] = None,
    table_name: str = "data",
    show_summary: bool = True,
) -> SingleTableMetadata:
    """
    Auto-detect metadata from a dataset using SDV.
    
    Args:
        data_path: Path to the dataset file (CSV, parquet, etc.)
        output_path: Path to save metadata JSON (optional)
        table_name: Name for the table in metadata
        show_summary: Whether to display metadata summary
        
    Returns:
        SDV SingleTableMetadata object
    """
    console.print(f"📊 Loading dataset from {data_path}...")

    # Load data based on file extension
    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() in ['.parquet', '.pq']:
        df = pd.read_parquet(data_path)
    elif data_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    console.print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Auto-detect metadata
    console.print("🔍 Auto-detecting metadata...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    # Save metadata if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        console.print(f"💾 Saved metadata to {output_path}")

    # Display summary if requested
    if show_summary:
        _display_metadata_summary(metadata, df)

    return metadata


def _display_metadata_summary(metadata: SingleTableMetadata, df: pd.DataFrame):
    """Display a rich summary of detected metadata."""
    table = Table(title="📋 Detected Metadata Summary")
    table.add_column("Column", style="cyan")
    table.add_column("SDV Type", style="magenta")
    table.add_column("Data Type", style="green")
    table.add_column("Null Count", style="yellow")
    table.add_column("Sample Values", style="blue")

    for column_name, column_meta in metadata.columns.items():
        null_count = df[column_name].isnull().sum()
        sample_values = df[column_name].dropna().head(3).tolist()
        sample_str = str(sample_values)[:50] + "..." if len(str(sample_values)) > 50 else str(sample_values)

        table.add_row(
            column_name,
            column_meta.get('sdtype', 'unknown'),
            str(df[column_name].dtype),
            str(null_count),
            sample_str
        )
    
    console.print(table)
