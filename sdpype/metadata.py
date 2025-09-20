"""Metadata detection and management for SDV."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sdv.metadata import SingleTableMetadata

console = Console()


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
