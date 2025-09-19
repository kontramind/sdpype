# sdpype/core/status.py - Repository status checking
"""
Basic repository status and experiment summary
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def show_repository_status():
    """Show basic repository status"""

    console.print("📊 SDPype Repository Status", style="bold cyan")

    # Show directory structure
    _show_directory_status()

    # Show file counts  
    _show_file_counts()

    # Show basic summary
    _show_basic_summary()

def _show_directory_status():
    """Show which key directories exist"""

    console.print("\n📁 Directory Structure:")

    key_dirs = [
        "experiments/",
        "experiments/data/",
        "experiments/data/raw/",
        "experiments/data/processed/",
        "experiments/data/synthetic/",
        "experiments/models/",
        "experiments/metrics/",
    ]

    table = Table(show_header=True)
    table.add_column("Directory", style="cyan")
    table.add_column("Status", style="green")

    for dir_path in key_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            status = "✅ Exists"
        else:
            status = "❌ Missing"

        table.add_row(dir_path, status)

    console.print(table)

def _show_file_counts():
    """Show counts of important file types"""

    console.print("\n📋 File Counts:")

    file_patterns = [
        ("experiments/metrics/*.json", "📊 Experiment metrics"),
        ("experiments/models/*.pkl", "🤖 Trained models"),  
        ("experiments/data/raw/*.csv", "📁 Raw data files"),
        ("experiments/data/processed/*.csv", "🔄 Processed data"),
        ("experiments/data/synthetic/*.csv", "🎲 Synthetic data"),
        ("experiments/metrics/statistical_*.json", "📊 Statistical metrics"),
        ("experiments/metrics/statistical_*.txt", "📋 Statistical metric report"),
    ]

    table = Table(show_header=True)
    table.add_column("File Type", style="cyan")
    table.add_column("Count", style="magenta")

    for pattern, description in file_patterns:
        files = list(Path().glob(pattern))
        count = len(files)
        table.add_row(description, str(count))

    console.print(table)

def _show_basic_summary():
    """Show basic repository summary"""

    # Count key indicators
    metrics_count = len(list(Path().glob("experiments/metrics/*.json")))
    models_count = len(list(Path().glob("experiments/models/*.pkl")))

    console.print("\n📋 Repository Summary:")

    if metrics_count == 0 and models_count == 0:
        console.print("🔵 Status: Clean (no experiments)", style="blue")
        console.print("💡 Run 'sdpype setup' to initialize or 'sdpype pipeline' to start experiments")
    else:
        console.print(f"🟢 Status: Active ({metrics_count} experiments, {models_count} models)", style="green")
        console.print("💡 Run 'sdpype pipeline' to add more experiments or 'sdpype purge' to clean up")
