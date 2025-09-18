# sdpype/core/pipeline.py - Pipeline execution logic
"""
DVC pipeline execution functionality
"""

import subprocess
from pathlib import Path
from rich.console import Console

console = Console()


def _validate_repository():
    """Check if we're in a valid SDPype repository"""
    
    required_files = [
        "pyproject.toml",
        "dvc.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        console.print("❌ Not in a valid SDPype repository!", style="bold red")
        console.print("Missing files:", style="red")
        for file in missing_files:
            console.print(f"  • {file}")
        console.print("\n💡 Run this command from your SDPype project root", style="yellow")
        return False
    
    return True

def _validate_stage_name(stage_name: str) -> bool:
    """Validate that the stage name exists in dvc.yaml"""
    
    # Known stages from your pipeline
    known_stages = [
        "preprocess",
        "train_sdg", 
        "generate_synthetic",
        "statistical_similarity",
    ]
    
    if stage_name not in known_stages:
        console.print(f"❌ Unknown stage: '{stage_name}'", style="bold red")
        console.print("Available stages:", style="yellow")
        for stage in known_stages:
            console.print(f"  • {stage}")
        console.print("\n💡 Check your dvc.yaml file for exact stage names", style="yellow")
        return False

    return True

def _show_pipeline_summary():
    """Show brief summary after pipeline completion"""
    
    console.print("\n📊 Pipeline Summary:")
    
    # Check for common output files
    output_patterns = [
        ("experiments/models/*.pkl", "🤖 Models"),
        ("experiments/data/synthetic/*.csv", "🎲 Synthetic data"),
        ("experiments/metrics/*.json", "📈 Metrics"),
    ]
    
    for pattern, description in output_patterns:
        files = list(Path().glob(pattern))
        count = len(files)
        if count > 0:
            console.print(f"  {description}: {count} files")
    
    console.print("\n💡 Next steps:")
    console.print("  • Check outputs in experiments/ directory")
    console.print("  • Run evaluation stages if available")
    console.print("  • Use dvc exp show to compare results")

def _show_stage_summary(stage_name: str):
    """Show summary specific to the completed stage"""
    
    console.print(f"\n📊 Stage '{stage_name}' Summary:")
    
    # Stage-specific output patterns
    stage_outputs = {
        "preprocess": [
            ("experiments/data/processed/*.csv", "📁 Processed data"),
            ("experiments/metrics/preprocess_*.json", "📊 Preprocessing metrics")
        ],
        "train_sdg": [
            ("experiments/models/*.pkl", "🤖 Trained models"),
            ("experiments/metrics/training_*.json", "📈 Training metrics")
        ],
        "generate_synthetic": [
            ("experiments/data/synthetic/*.csv", "🎲 Synthetic data"),
            ("experiments/metrics/generation_*.json", "⚡ Generation metrics")
        ],
        "statistical_similarity": [
             ("experiments/metrics/statistical_similarity_*.json", "📊 Statistical similarity"),
            ("experiments/metrics/statistical_report_*.txt", "📋 Statistical reports")
         ]
    }

    patterns = stage_outputs.get(stage_name, [])
    if not patterns:
        console.print("  📁 Check experiments/ directory for outputs")
        return
    
    for pattern, description in patterns:
        files = list(Path().glob(pattern))
        if files:
            console.print(f"  {description}: {len(files)} files")
            # Show the most recent file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            console.print(f"    Latest: {latest_file}")


def run_pipeline_command(force: bool = False):
    """Run the complete DVC pipeline"""

    console.print("🚀 Running SDPype pipeline...", style="bold blue")

    # Check if we're in a valid repository
    if not _validate_repository():
        return False

    # Build DVC command
    cmd = ["dvc", "repro"]
    if force:
        cmd.append("--force")

    console.print(f"📋 Executing: {' '.join(cmd)}")

    try:
        # Run DVC pipeline
        with console.status("[bold green]Running pipeline stages...", spinner="dots"):
            result = subprocess.run(cmd, check=True)

        console.print("✅ Pipeline completed successfully!", style="bold green")

        # Show quick summary
        _show_pipeline_summary()
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"❌ Pipeline failed with exit code {e.returncode}", style="bold red")
        console.print("💡 Try running with --force or check your configuration", style="yellow")
        return False

    except FileNotFoundError:
        console.print("❌ DVC not found. Make sure DVC is installed:", style="bold red")
        console.print("   pip install dvc", style="yellow")
        return False


def run_stage_command(stage_name: str, force: bool = False):
    """Run a specific DVC pipeline stage"""

    console.print(f"🎯 Running stage: [bold cyan]{stage_name}[/bold cyan]", style="bold blue")

    # Check if we're in a valid repository
    if not _validate_repository():
        return False

    # Validate stage name
    if not _validate_stage_name(stage_name):
        return False

    # Build DVC command
    cmd = ["dvc", "repro", "-s", stage_name]
    if force:
        cmd.append("--force")

    console.print(f"📋 Executing: {' '.join(cmd)}")

    try:
        # Run DVC stage
        with console.status(f"[bold green]Running {stage_name} stage...", spinner="dots"):
            result = subprocess.run(cmd, check=True)

        console.print(f"✅ Stage '{stage_name}' completed successfully!", style="bold green")

        # Show stage-specific summary
        _show_stage_summary(stage_name)
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"❌ Stage '{stage_name}' failed with exit code {e.returncode}", style="bold red")
        console.print("💡 Check the error output above for details", style="yellow")
        return False

    except FileNotFoundError:
        console.print("❌ DVC not found. Make sure DVC is installed:", style="bold red")
        console.print("   pip install dvc", style="yellow")
        return False
