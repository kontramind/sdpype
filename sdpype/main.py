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
        "experiments/configs",
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

    console.print(Panel.fit(
        "✅ Repository setup complete!\n\n"
        "Next steps:\n"
        "• Run: [bold]sdpype models[/bold] (see available models)\n"
        "• Run: [bold]dvc repro[/bold] (run full pipeline)\n"
        "• Run: [bold]sdpype exp run --name 'test' --seed 42[/bold] (run experiment)",
        title="🎉 Setup Complete"
    ))


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
