"""
Batch Trace Script for SDPype

Processes multiple experiment folders in parallel, auto-discovering model IDs
and generating trace_chain outputs (HTML plots + summary index).

Usage:
    python batch_trace.py "./mimic_iii_baseline_*/"
    python batch_trace.py "./experiments_*/)" --max-generations 50
    python batch_trace.py "./exp_*/" --summary-output results/index.html
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from trace_chain import trace_chain, plot_chain_interactive, export_csv, export_json

app = typer.Typer(help="Batch process experiment folders with trace_chain")
console = Console()


def discover_model_id(folder: Path) -> Optional[str]:
    """
    Auto-discover model ID from folder's synthetic data CSV files.

    Looks for gen_0 CSV files in {folder}/data/synthetic/ and extracts
    the model ID from the filename.

    Args:
        folder: Path to experiment folder

    Returns:
        Model ID string if found, None otherwise
    """
    synthetic_dir = folder / "data" / "synthetic"

    if not synthetic_dir.exists():
        return None

    # Look for gen_0 CSV files: synthetic_data_*_gen_0_*.csv
    gen_0_files = list(synthetic_dir.glob("synthetic_data_*_gen_0_*.csv"))

    if not gen_0_files:
        return None

    # Extract model ID from filename
    # Pattern: synthetic_data_<MODEL_ID>_encoded.csv or similar
    filename = gen_0_files[0].name
    match = re.match(r"synthetic_data_(.+?)_(?:encoded|decoded)\.csv", filename)

    if match:
        return match.group(1)

    return None


def validate_folder(folder: Path) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate folder structure and auto-discover model ID.

    Args:
        folder: Path to experiment folder

    Returns:
        Tuple of (is_valid, model_id, error_message)
    """
    if not folder.exists():
        return False, None, f"Folder does not exist: {folder}"

    if not folder.is_dir():
        return False, None, f"Not a directory: {folder}"

    synthetic_dir = folder / "data" / "synthetic"
    if not synthetic_dir.exists():
        return False, None, f"Missing data/synthetic/ subdirectory in: {folder}"

    model_id = discover_model_id(folder)
    if not model_id:
        return False, None, f"Could not find gen_0 model ID in: {synthetic_dir}"

    return True, model_id, None


def process_folder(
    folder: Path,
    max_generations: int = 100,
    format: Optional[str] = None,
    output: Optional[str] = None,
) -> Tuple[bool, Optional[Dict]]:
    """
    Process a single experiment folder.

    Args:
        folder: Path to experiment folder
        max_generations: Max generations to trace

    Returns:
        Tuple of (success, result_dict)
    """
    is_valid, model_id, error = validate_folder(folder)

    if not is_valid:
        return False, {"folder": str(folder), "error": error}

    try:
        # Run trace_chain (reads k-anonymity from privacy_*.json if available)
        results = trace_chain(
            model_id=model_id,
            max_generations=max_generations,
            experiments_root=str(folder)
        )

        if not results:
            return False, {
                "folder": str(folder),
                "model_id": model_id,
                "error": "trace_chain returned no results"
            }

        # Generate HTML plot in folder (overwrite if exists)
        output_file = folder / f"{folder.name}.html"
        if output_file.exists():
            output_file.unlink()
        plot_chain_interactive(results, output_file=str(output_file))

        # Export in requested format if specified
        exported_files = []
        if format:
            output_dir = Path(output) if output else folder
            if format.lower() == "csv":
                csv_data = export_csv(results)
                csv_file = output_dir / f"{folder.name}.csv"
                csv_file.parent.mkdir(parents=True, exist_ok=True)
                if csv_file.exists():
                    csv_file.unlink()
                csv_file.write_text(csv_data)
                exported_files.append(str(csv_file))
            elif format.lower() == "json":
                json_data = export_json(results)
                json_file = output_dir / f"{folder.name}.json"
                json_file.parent.mkdir(parents=True, exist_ok=True)
                if json_file.exists():
                    json_file.unlink()
                json_file.write_text(json_data)
                exported_files.append(str(json_file))

        # Extract gen_0 metrics for summary
        gen_0_metrics = results[0]['metrics'] if results else {}

        return True, {
            "folder": str(folder),
            "folder_name": folder.name,
            "model_id": model_id,
            "generations_traced": len(results),
            "output_html": str(output_file),
            "exported_files": exported_files,
            "gen_0_metrics": {
                "alpha_precision": gen_0_metrics.get("alpha_precision", "N/A"),
                "prdc_avg": gen_0_metrics.get("prdc_avg", "N/A"),
                "detection_avg": gen_0_metrics.get("detection_avg", "N/A"),
            }
        }

    except Exception as e:
        return False, {
            "folder": str(folder),
            "model_id": model_id,
            "error": f"Error during processing: {str(e)}"
        }


def generate_summary_html(
    results: List[Dict],
    output_file: Optional[str] = None
) -> str:
    """
    Generate summary HTML with links to all results.

    Args:
        results: List of result dicts from process_folder()
        output_file: Output filename (default: batch_trace_results.html)

    Returns:
        Path to generated HTML file
    """
    if output_file is None:
        output_file = "batch_trace_results.html"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to ensure clean overwrite
    if output_path.exists():
        output_path.unlink()

    # Separate successful and failed results
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <title>Batch Trace Results</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }",
        "    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }",
        "    h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }",
        "    h2 { color: #555; margin-top: 30px; }",
        "    .summary { background: #f9f9f9; padding: 15px; border-radius: 4px; margin: 15px 0; }",
        "    .status-success { color: #28a745; font-weight: bold; }",
        "    .status-failed { color: #dc3545; font-weight: bold; }",
        "    table { width: 100%; border-collapse: collapse; margin: 15px 0; }",
        "    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
        "    th { background: #f0f0f0; font-weight: 600; }",
        "    tr:hover { background: #f9f9f9; }",
        "    a { color: #007bff; text-decoration: none; }",
        "    a:hover { text-decoration: underline; }",
        "    .metric { font-family: monospace; font-size: 0.9em; }",
        "    .error-box { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 12px; border-radius: 4px; margin: 10px 0; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h1>Batch Trace Results</h1>",
        f"    <div class='summary'>",
        f"      <p><strong>Total folders processed:</strong> {len(results)}</p>",
        f"      <p><span class='status-success'>✓ Successful:</span> {len(successful)}</p>",
        f"      <p><span class='status-failed'>✗ Failed:</span> {len(failed)}</p>",
        f"    </div>",
    ]

    if successful:
        html_parts.extend([
            "    <h2>Successful Results</h2>",
            "    <table>",
            "      <thead>",
            "        <tr>",
            "          <th>Folder</th>",
            "          <th>Model ID (last 20 chars)</th>",
            "          <th>Generations</th>",
            "          <th>Alpha Precision (Gen 0)</th>",
            "          <th>PRDC Avg (Gen 0)</th>",
            "          <th>Detection Avg (Gen 0)</th>",
            "          <th>Plot</th>",
            "        </tr>",
            "      </thead>",
            "      <tbody>",
        ])

        for result in successful:
            folder_name = result.get("folder_name", "")
            model_id = result.get("model_id", "")
            model_id_short = model_id[-20:] if len(model_id) > 20 else model_id
            generations = result.get("generations_traced", "N/A")
            output_html = result.get("output_html", "")
            gen_0 = result.get("gen_0_metrics", {})

            alpha = gen_0.get("alpha_precision", "N/A")
            prdc = gen_0.get("prdc_avg", "N/A")
            detection = gen_0.get("detection_avg", "N/A")

            # Format metrics nicely
            alpha_str = f"{alpha:.3f}" if isinstance(alpha, float) else str(alpha)
            prdc_str = f"{prdc:.3f}" if isinstance(prdc, float) else str(prdc)
            detection_str = f"{detection:.3f}" if isinstance(detection, float) else str(detection)

            html_parts.extend([
                "        <tr>",
                f"          <td><strong>{folder_name}</strong></td>",
                f"          <td><span class='metric'>{model_id_short}</span></td>",
                f"          <td>{generations}</td>",
                f"          <td>{alpha_str}</td>",
                f"          <td>{prdc_str}</td>",
                f"          <td>{detection_str}</td>",
                f"          <td><a href='{Path(output_html).name}' target='_blank'>View Plot →</a></td>",
                "        </tr>",
            ])

        html_parts.extend([
            "      </tbody>",
            "    </table>",
        ])

    if failed:
        html_parts.extend([
            "    <h2>Failed Results</h2>",
        ])

        for result in failed:
            folder = result.get("folder", "")
            error = result.get("error", "Unknown error")

            html_parts.extend([
                "    <div class='error-box'>",
                f"      <strong>{folder}</strong><br>",
                f"      {error}",
                "    </div>",
            ])

    html_parts.extend([
        "  </div>",
        "</body>",
        "</html>",
    ])

    html_content = "\n".join(html_parts)
    output_path.write_text(html_content)

    return str(output_path)


@app.command()
def main(
    pattern: str = typer.Argument(
        ...,
        help="Glob pattern for experiment folders (e.g., './mimic_iii_baseline_*/')"
    ),
    max_generations: int = typer.Option(
        100,
        "--max-generations",
        help="Maximum generations to trace per folder"
    ),
    summary_output: Optional[str] = typer.Option(
        "batch_trace_results.html",
        "--summary-output",
        "-s",
        help="Output path for summary HTML index"
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Export format (csv/json) for each folder"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for exported files (default: same as folder)"
    ),
) -> None:
    """
    Process multiple experiment folders with trace_chain.

    Auto-discovers model IDs, generates HTML plots in each folder,
    and creates a summary index with links to all results.

    Examples:
        python batch_trace.py "./mimic_iii_baseline_*/"
        python batch_trace.py "./exp_*/" --max-generations 50 -s results/index.html
        python batch_trace.py "./exp_*/" --format csv --output ./exports
        python batch_trace.py "./exp_*/" --format json
    """

    console.print(f"[cyan]Pattern:[/cyan] {pattern}")

    # Expand glob pattern
    folders = sorted(Path().glob(pattern))

    if not folders:
        console.print(f"[red]Error: No folders matched pattern: {pattern}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(folders)} folder(s)[/cyan]\n")

    results = []

    # Process each folder
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=len(folders))

        for folder in folders:
            progress.update(task, description=f"[cyan]Processing {folder.name}...")

            success, result = process_folder(folder, max_generations, format, output)
            results.append(result)

            if success:
                progress.print(f"[green]✓[/green] {folder.name}")
            else:
                error_msg = result.get("error", "Unknown error")
                progress.print(f"[red]✗[/red] {folder.name}: {error_msg}")
                raise typer.Exit(1)  # Fail fast

            progress.advance(task)

    console.print()

    # Generate summary HTML
    summary_path = generate_summary_html(results, summary_output)
    console.print(f"[green]✓ Summary generated:[/green] {summary_path}")

    # Print summary stats
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  [green]Successful:[/green] {successful}")
    console.print(f"  [red]Failed:[/red] {failed}")


if __name__ == "__main__":
    app()
