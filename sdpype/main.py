# Enhanced sdpype/main.py - Add synthcity model discovery

"""
SDPype - Synthetic Data Pipeline CLI with SDV and Synthcity support
"""

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="sdpype",
    help="🚀 Synthetic Data Pipeline with SDV and Synthcity support",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def init(
    project_name: str = typer.Argument("my-sdpype-project", help="Project name"),
    sample_data: bool = typer.Option(True, "--sample-data/--no-sample-data", help="Include sample data"),
):
    """🚀 Initialize new SDPype project"""
    
    project_path = Path(project_name)
    if project_path.exists():
        console.print(f"❌ Directory {project_name} already exists", style="bold red")
        raise typer.Exit(1)
    
    # Create structure
    project_path.mkdir()
    (project_path / "data" / "raw").mkdir(parents=True)
    (project_path / "data" / "processed").mkdir()
    (project_path / "data" / "synthetic").mkdir()
    (project_path / "models").mkdir()
    (project_path / "metrics").mkdir()
    (project_path / "config").mkdir()
    
    # Initialize DVC
    subprocess.run(["dvc", "init", "--no-scm"], cwd=project_path)
    
    # Create sample data if requested
    if sample_data:
        _create_sample_data(project_path)
    
    console.print(f"✅ Project [bold]{project_name}[/bold] initialized!", style="green")
    console.print(f"Next: cd {project_name} && sdpype models")
    console.print(f"Then: sdpype pipeline --config sdg=<model_name>")


@app.command("pipeline")
def run_pipeline(
    config: Optional[str] = typer.Option(None, "--config", help="Config overrides (e.g., sdg=synthcity_tvae)"),
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
    """📊 Show pipeline status"""
    
    result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
    
    if "No changes" in result.stdout:
        console.print("✅ Pipeline is up to date", style="green")
    else:
        console.print("📋 Pipeline status:")
        console.print(result.stdout)


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
    console.print("  sdpype pipeline --config sdg=gaussian")
    console.print("  sdpype pipeline --config sdg=synthcity_tvae") 
    console.print("  sdpype pipeline --config sdg=synthcity_ddpm")


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
        console.print("   Then run: sdpype pipeline --config sdg=synthcity_<model_name>")
        
    except ImportError:
        console.print("❌ Synthcity not installed. Install with: pip install synthcity", style="red")
    except Exception as e:
        console.print(f"❌ Error listing Synthcity models: {e}", style="red")


def _create_sample_data(project_path: Path):
    """Create simple sample dataset"""
    
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
    
    sample_file = project_path / "data" / "raw" / "sample_data.csv"
    data.to_csv(sample_file, index=False)
    console.print(f"📊 Created sample data: {len(data)} rows")


if __name__ == "__main__":
    app()
