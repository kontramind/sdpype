# Enhanced sdpype/main.py - Monolithic structure with experiment versioning

"""
SDPype - Synthetic Data Pipeline CLI for Monolithic Repository
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="sdpype",
    help="🚀 Synthetic Data Pipeline - Monolithic Repository with Experiment Versioning",
    rich_markup_mode="rich"
)

# Create experiment sub-app
exp_app = typer.Typer(
    name="experiment",
    help="🔬 Experiment management commands",
    rich_markup_mode="rich"
)
app.add_typer(exp_app, name="exp")

console = Console()


@app.command("setup")
def setup_repository():
    """🏗️ Setup repository for experiments (creates directories and sample data)"""

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        console.print("❌ Not in SDPype repository root", style="bold red")
        console.print("💡 Run this command from your sdpype/ directory")
        raise typer.Exit(1)

    # Create experiments directory structure (if not exists)
    dirs = [
        "experiments/data/raw",
        "experiments/data/processed",
        "experiments/data/synthetic",
        "experiments/models",
        "experiments/metrics",
        "pipelines"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    console.print("✅ Directory structure created", style="green")

    # Initialize DVC if not already done
    if not Path(".dvc").exists():
        subprocess.run(["dvc", "init"])
        console.print("✅ DVC initialized", style="green")

    # Initialize Git if not already done
    if not Path(".git").exists():
        subprocess.run(["git", "init"])
        console.print("✅ Git initialized", style="green")

    # Create sample data
    _create_sample_data()

    # Create proper .dvcignore
    console.print("📝 Creating .dvcignore...")
    dvcignore_content = """# .dvcignore - What DVC should ignore when tracking files

# System files
.DS_Store
Thumbs.db
desktop.ini

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Temporary files
*.tmp
*.temp
*.log
*.bak
*.swp
*.swo

# Hidden files/directories (except .dvc)
.*
!.dvc
!.dvcignore

# Common data science temp files
*.pkl.tmp
*.csv.tmp
*.json.tmp
checkpoints/
temp/
tmp/

# Jupyter notebook checkpoints
.ipynb_checkpoints/

# Model training artifacts that shouldn't be tracked
logs/
tensorboard/
wandb/

# OS and editor files
*~
*.orig

# Large files that might accidentally be in data directories
*.zip
*.tar
*.tar.gz
*.rar
*.7z

# Database files (if any)
*.db
*.sqlite
*.sqlite3
"""

    with open(".dvcignore", "w") as f:
        f.write(dvcignore_content)
    console.print("✅ .dvcignore created")

    console.print(Panel.fit(
        "✅ Repository setup complete!\n\n"
        "Next steps:\n"
        "• Run: [bold]sdpype models[/bold] (see available models)\n"
        "• Run: [bold]dvc repro[/bold] (run full pipeline)\n"
        "• Run: [bold]sdpype exp run --name 'test' --seed 42[/bold] (run experiment)",
        title="🎉 Setup Complete"
    ))


# Add this to sdpype/main.py

@app.command("nuke")
def nuke_repository(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    keep_raw_data: bool = typer.Option(True, "--keep-raw-data", help="Preserve raw data files"),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Preserve DVC cache"),
):
    """🧹 Nuclear reset: Remove all experiments, models, and DVC state

    ⚠️  WARNING: This will DELETE all experiment results, trained models, and metrics!
    After running this command, you'll need to run 'sdpype setup' to restart.
    """

    console.print("🧹 Nuclear Reset - Complete Repository Cleanup", style="bold red")
    console.print("⚠️  This will permanently delete:", style="bold yellow")

    # List what will be deleted
    items_to_delete = [
        "📊 All experiment metrics (experiments/metrics/)",
        "🤖 All trained models (experiments/models/)",
        "📈 All processed data (experiments/data/processed/)",
        "🎲 All synthetic data (experiments/data/synthetic/)",
        "🔄 DVC pipeline lock files",
        "📋 DVC experiment history",
    ]

    if not keep_raw_data:
        items_to_delete.append("📁 Raw data files (experiments/data/raw/)")

    if not keep_cache:
        items_to_delete.append("💾 DVC cache (.dvc/cache/)")

    for item in items_to_delete:
        console.print(f"  • {item}")

    console.print("\n✅ Will be preserved:", style="bold green")
    preserved_items = [
        "🐍 Source code (sdpype/ folder)",
        "⚙️  Configuration (params.yaml, dvc.yaml)",
        "📋 Project files (pyproject.toml, README.md)",
    ]

    if keep_raw_data:
        preserved_items.append("📁 Raw data files (experiments/data/raw/)")

    if keep_cache:
        preserved_items.append("💾 DVC cache (.dvc/cache/)")

    for item in preserved_items:
        console.print(f"  • {item}")

    # Confirmation
    if not confirm:
        console.print(f"\n❓ Are you sure you want to proceed?", style="bold yellow")
        response = typer.prompt("Type 'NUKE' to confirm")
        if response != "NUKE":
            console.print("❌ Nuke cancelled", style="red")
            raise typer.Exit(0)

    console.print(f"\n🧹 Starting nuclear reset...")

    # 1. Clear DVC experiments
    console.print("🔄 Clearing DVC experiments...")
    try:
        result = subprocess.run(["dvc", "exp", "remove", "--all"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            console.print("  ✅ DVC experiments cleared")
        else:
            console.print("  ⚠️  No DVC experiments to clear")
    except Exception as e:
        console.print(f"  ⚠️  Could not clear DVC experiments: {e}")

    # 2. Remove DVC cache (optional)
    if not keep_cache:
        console.print("💾 Clearing DVC cache...")
        dvc_cache = Path(".dvc/cache")
        if dvc_cache.exists():
            import shutil
            shutil.rmtree(dvc_cache)
            console.print("  ✅ DVC cache cleared")
        else:
            console.print("  ⚠️  No DVC cache found")

    # 3. Remove pipeline lock files
    console.print("🔒 Removing pipeline lock files...")
    lock_files = ["dvc.lock"]
    for lock_file in lock_files:
        lock_path = Path(lock_file)
        if lock_path.exists():
            lock_path.unlink()
            console.print(f"  ✅ Removed {lock_file}")

    # 4. Clean experiment directories
    console.print("📁 Cleaning experiment directories...")

    directories_to_clean = [
        ("experiments/metrics", "📊 Metrics"),
        ("experiments/models", "🤖 Models"), 
        ("experiments/data/processed", "📈 Processed data"),
        ("experiments/data/synthetic", "🎲 Synthetic data"),
    ]

    if not keep_raw_data:
        directories_to_clean.append(("experiments/data/raw", "📁 Raw data"))

    for dir_path, description in directories_to_clean:
        dir_obj = Path(dir_path)
        if dir_obj.exists():
            import shutil
            shutil.rmtree(dir_obj)
            console.print(f"  ✅ Cleared {description}")
        else:
            console.print(f"  ⚠️  {description} directory not found")

    # 5. Clean any Hydra outputs (if they exist)
    console.print("🌊 Cleaning Hydra outputs...")
    hydra_dirs = ["outputs", ".hydra"]
    for hydra_dir in hydra_dirs:
        hydra_path = Path(hydra_dir)
        if hydra_path.exists():
            import shutil
            shutil.rmtree(hydra_path)
            console.print(f"  ✅ Removed {hydra_dir}/")

    # 6. Clean Python cache
    console.print("🐍 Cleaning Python cache...")
    import shutil
    cache_dirs = [
        "__pycache__",
        "sdpype/__pycache__", 
        ".pytest_cache",
    ]
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            console.print(f"  ✅ Removed {cache_dir}/")

    # More aggressive DVC experiment cleanup
    console.print("🔄 Deep cleaning DVC experiments...")
    cleanup_commands = [
        ["dvc", "exp", "remove", "--all-commits"],
        ["dvc", "exp", "gc", "--workspace", "--force"],
        ["dvc", "cache", "dir", "--unset"],  # Reset cache location
    ]

    for cmd in cleanup_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            console.print(f"  ✅ {' '.join(cmd)}")
        except Exception as e:
            console.print(f"  ⚠️  Failed: {' '.join(cmd)} - {e}")

    # Remove DVC lock files
    console.print("🔒 Removing all DVC state files...")
    dvc_files_to_remove = [
        ".dvc/cache",
        ".dvc/tmp", 
        "dvc.lock",
        ".dvcignore",
    ]

    for dvc_file in dvc_files_to_remove:
        file_path = Path(dvc_file)
        if file_path.exists():
            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
            console.print(f"  ✅ Removed {dvc_file}")

    console.print("\n🎉 Nuclear reset complete!", style="bold green")
    console.print("\n📋 Next steps:")
    console.print("  1. Run: [bold]uv run sdpype setup[/bold] (recreate experiment structure)")
    console.print("  2. Add your data to experiments/data/raw/")
    console.print("  3. Run: [bold]uv run dvc repro[/bold] (start experimenting)")

    console.print(f"\n✨ Repository is now in pristine state!", style="bold cyan")



# Add to sdpype/main.py

@app.command("params")
def show_parameters(
    current: bool = typer.Option(False, "--current", help="Show current parameter values"),
    experiments: bool = typer.Option(False, "--experiments", help="Show parameters across experiments"),
    diff: bool = typer.Option(False, "--diff", help="Show parameter differences"),
):
    """📋 Show tracked parameters"""

    if current:
        console.print("📋 Current Parameter Values:")
        import yaml
        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        # Show in organized sections
        sections = {
            "🧪 Experiment": params.get("experiment", {}),
            "🤖 Model (SDG)": params.get("sdg", {}),
            "🔄 Preprocessing": params.get("preprocessing", {}),
            "📊 Generation": params.get("generation", {}),
            "📈 Evaluation": params.get("evaluation", {}),
        }

        for section_name, section_data in sections.items():
            if section_data:
                console.print(f"\n{section_name}:")
                import json
                console.print(json.dumps(section_data, indent=2))

    elif experiments:
        console.print("📊 Parameters Across Experiments:")
        subprocess.run(["dvc", "exp", "show"])

    elif diff:
        console.print("🔍 Parameter Differences:")
        subprocess.run(["dvc", "params", "diff", "--all"])

    else:
        console.print("📋 Available parameter commands:")
        console.print("  sdpype params --current       # Current values")
        console.print("  sdpype params --experiments    # Across experiments") 
        console.print("  sdpype params --diff          # Show differences")


@app.command("pipeline")
def run_pipeline(
    config: Optional[str] = typer.Option(None, "--config", help="Config overrides (e.g., sdg=ctgan)"),
    force: bool = typer.Option(False, "--force", help="Force rerun"),
):
    """🚀 Run the complete pipeline"""
    
    console.print("Running SDPype pipeline...")
    
    cmd = ["dvc", "repro"]
    if force:
        cmd.append("--force")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        console.print("✅ Pipeline completed!", style="green")
    else:
        console.print("❌ Pipeline failed!", style="red")
        raise typer.Exit(1)


@app.command("stage")  
def run_stage(
    stage_name: str = typer.Argument(..., help="Stage to run"),
    force: bool = typer.Option(False, "--force", help="Force rerun"),
):
    """🎯 Run specific stage"""
    
    console.print(f"Running stage: [bold]{stage_name}[/bold]")
    
    cmd = ["dvc", "repro", "-s", stage_name]
    if force:
        cmd.append("--force")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        console.print("✅ Stage completed!", style="green")
    else:
        console.print("❌ Stage failed!", style="red")
        raise typer.Exit(1)


@app.command()
def status():
    """📊 Show pipeline and experiment status"""
    
    console.print("📊 Pipeline Status:")
    result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
    
    if "No changes" in result.stdout:
        console.print("✅ Pipeline is up to date", style="green")
    else:
        console.print(result.stdout)

    # Show experiments overview
    console.print("\n🔬 Experiments Overview:")
    _show_experiments_summary()


@app.command()
def models():
    """📋 List available SDG models"""

    console.print("🤖 Available Synthetic Data Generation Models")
    
    # Create table
    table = Table(title="SDG Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Library", style="magenta") 
    table.add_column("Type", style="green")
    table.add_column("Description", style="white")
    
    # SDV Models
    table.add_row("gaussian", "SDV", "Copula", "Fast, good for tabular data")
    table.add_row("ctgan", "SDV", "GAN", "High quality, slower training")
    
    # Synthcity Models (common ones)
    table.add_row("synthcity_ctgan", "Synthcity", "GAN", "CTGAN implementation")
    table.add_row("synthcity_tvae", "Synthcity", "VAE", "Tabular VAE") 
    table.add_row("synthcity_marginal_coupling", "Synthcity", "Copula", "Advanced copula method")
    table.add_row("synthcity_nflow", "Synthcity", "Flow", "Normalizing flows")
    table.add_row("synthcity_ddpm", "Synthcity", "Diffusion", "Diffusion model")
    
    console.print(table)
    
    console.print("\n💡 Usage examples:")
    console.print("  dvc repro --set-param sdg=gaussian")
    console.print("  sdpype exp run --name 'test' --config 'sdg=ctgan'")


@app.command("list-synthcity")
def list_synthcity_models():
    """📋 List all available Synthcity models (requires synthcity installed)"""
    
    try:
        from synthcity.plugins import Plugins
        
        plugins = Plugins()
        available_models = plugins.list()
        
        console.print("🔬 All Available Synthcity Models:")
        
        table = Table()
        table.add_column("Model Name", style="cyan")
        table.add_column("Type", style="green") 
        table.add_column("Status", style="magenta")
        
        for model_name in sorted(available_models):
            # Try to get model info
            try:
                model = plugins.get(model_name)
                model_type = getattr(model, 'type', 'Unknown')
                status = "✅ Available"
            except Exception as e:
                model_type = "Unknown"
                status = "❌ Error"
            
            table.add_row(model_name, model_type, status)
        
        console.print(table)
        
        console.print(f"\n💡 To use any model, create config/sdg/synthcity_<model_name>.yaml")
        console.print("   Then run: sdpype exp run --config 'sdg=synthcity_<model_name>'")
        
    except ImportError:
        console.print("❌ Synthcity not installed.", style="red")
        console.print("Install with: uv add synthcity", style="yellow")
    except Exception as e:
        console.print(f"❌ Error listing Synthcity models: {e}", style="red")


# EXPERIMENT MANAGEMENT COMMANDS

@exp_app.command("run")
def experiment_run(
    name: Optional[str] = typer.Option(None, "--name", help="Experiment name"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    config: Optional[str] = typer.Option(None, "--config", help="Config overrides"),
    queue: bool = typer.Option(False, "--queue", help="Queue experiment"),
    skip_statistical: bool = typer.Option(False, "--skip-statistical", help="Skip statistical similarity evaluation")  # ✨ NEW
):
    """🔬 Run experiment with versioning"""

    cmd = ["dvc", "exp", "run"]

    if name:
        cmd.extend(["-n", name])
    if seed:
        cmd.extend(["--set-param", f"experiment.seed={seed}"])
    if config:
        cmd.extend(["--set-param", config])
    if queue:
        cmd.append("--queue")

    # ✨ NEW: Handle statistical similarity flag
    if skip_statistical:
        cmd.extend(["--set-param", "evaluation.statistical_similarity.enabled=false"])
        console.print("⚠️  Skipping statistical similarity evaluation")
    else:
        cmd.extend(["--set-param", "evaluation.statistical_similarity.enabled=true"])
        console.print("📈 Including statistical similarity evaluation")

    console.print(f"🚀 Running experiment...")
    console.print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        console.print("✅ Experiment completed!", style="green")
    else:
        console.print("❌ Experiment failed!", style="red")
        raise typer.Exit(1)


@exp_app.command("list")
def experiment_list():
    """📋 List all experiments"""

    result = subprocess.run(
        ["dvc", "exp", "show", "--include-metrics", "--include-params"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        console.print("🔬 Experiments:")
        console.print(result.stdout)
    else:
        console.print("❌ Failed to list experiments", style="red")


@exp_app.command("summary")
def experiment_summary():
    """📈 Show experiment metrics summary"""

    _show_experiments_summary()


# UTILITY FUNCTIONS

def _create_sample_data():
    """Create sample dataset in monolithic structure"""

    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples).clip(18, 80).astype(int),
        'income': np.random.lognormal(10, 1, n_samples).clip(20000, 200000).astype(int),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    })

    # Save to monolithic experiments structure
    sample_file = Path("experiments/data/raw/sample_data.csv")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(sample_file, index=False)
    console.print(f"📊 Created sample data: {len(data)} rows at {sample_file}")


def _show_experiments_summary():
    """Show summary of completed experiments"""

    metrics_dir = Path("experiments/metrics")
    if not metrics_dir.exists():
        console.print("📋 No experiments found. Run: sdpype setup", style="yellow")
        return

    # Collect experiment results
    experiments = []

    for training_file in metrics_dir.glob("training_*.json"):
        try:
            with open(training_file) as f:
                training_data = json.load(f)

            seed = training_data.get("seed", "unknown")
            generation_file = metrics_dir / f"generation_{seed}.json"

            exp_data = {
                "seed": seed,
                "model": training_data.get("model_type", "unknown"),
                "library": training_data.get("library", "unknown"),
                "train_time": training_data.get("training_time", 0),
                "gen_time": 0,
                "samples": 0,
                "timestamp": training_data.get("timestamp", "unknown")
            }

            if generation_file.exists():
                with open(generation_file) as f:
                    gen_data = json.load(f)
                exp_data.update({
                    "gen_time": gen_data.get("generation_time", 0),
                    "samples": gen_data.get("samples_generated", 0)
                })

            experiments.append(exp_data)

        except Exception as e:
            console.print(f"⚠️  Error reading {training_file}: {e}")

    if not experiments:
        console.print("📋 No completed experiments found")
        console.print("💡 Run: dvc repro (to run basic pipeline)")
        console.print("💡 Or: sdpype exp run --name 'test' --seed 42")
        return

    # Display summary table
    table = Table(title="Experiment Summary")
    table.add_column("Seed", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Library", style="yellow")
    table.add_column("Train Time", style="green")
    table.add_column("Gen Time", style="blue")
    table.add_column("Samples", style="white")

    for exp in sorted(experiments, key=lambda x: str(x["seed"])):
        table.add_row(
            str(exp["seed"]),
            exp["model"],
            exp["library"],
            f"{exp['train_time']:.1f}s",
            f"{exp['gen_time']:.1f}s",
            str(exp["samples"])
        )

    console.print(table)
    console.print(f"\n📊 Total experiments: {len(experiments)}")


if __name__ == "__main__":
    app()
