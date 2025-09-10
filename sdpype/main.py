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

# Import serialization module for model info
from sdpype.serialization import (
    get_model_info, list_saved_models, validate_model,
    get_supported_libraries, SerializationError, ModelNotFoundError
)


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


# Create model management sub-app
model_app = typer.Typer(
    name="model",
    help="🤖 Model management commands",
    rich_markup_mode="rich"
)
app.add_typer(model_app, name="model")


# Create evaluation sub-app
eval_app = typer.Typer(
    name="eval",
    help="📊 Evaluation commands",
    rich_markup_mode="rich"
)
app.add_typer(eval_app, name="eval")


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
    console.print("\n🔬 Experiments & Models Overview:")
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


# MODEL MANAGEMENT COMMANDS

@model_app.command("list")
def model_list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    sort_by: str = typer.Option("seed", "--sort", help="Sort by: seed, timestamp, size, library")
):
    """📋 List all saved models"""

    try:
        models = list_saved_models()

        if not models:
            console.print("📋 No saved models found", style="yellow")
            console.print("💡 Run training first: dvc repro -s train_sdg")
            return

        # Sort models
        sort_keys = {
            "seed": lambda x: x.get("experiment_seed", 0),
            "timestamp": lambda x: x.get("saved_at", ""),
            "size": lambda x: x.get("file_size_mb", 0),
            "library": lambda x: x.get("library", "")
        }

        if sort_by in sort_keys:
            models.sort(key=sort_keys[sort_by])

        # Create table
        if verbose:
            table = Table(title="📋 All Saved Models (Detailed)")
            table.add_column("Seed", style="cyan")
            table.add_column("Model Type", style="magenta")
            table.add_column("Library", style="yellow")
            table.add_column("Size (MB)", style="blue", justify="right")
            table.add_column("Training Time", style="green", justify="right")
            table.add_column("Saved At", style="white")
            table.add_column("Experiment", style="cyan")

            for model in models:
                experiment_info = model.get("experiment", {})
                table.add_row(
                    str(model.get("experiment_seed", "?")),
                    model.get("model_type", "unknown"),
                    model.get("library", "unknown"),
                    f"{model.get('file_size_mb', 0):.1f}",
                    f"{model.get('training_time', 0):.1f}s",
                    model.get("saved_at", "unknown")[:19] if model.get("saved_at") else "unknown",
                    experiment_info.get("name", experiment_info.get("id", "unknown"))
                )
        else:
            table = Table(title="📋 All Saved Models")
            table.add_column("Seed", style="cyan")
            table.add_column("Experiment", style="yellow")
            table.add_column("Model", style="magenta")
            table.add_column("Library", style="yellow")
            table.add_column("Size (MB)", style="blue", justify="right")
            table.add_column("Status", style="green")

            for model in models:
                table.add_row(
                    str(model.get("experiment_seed", "?")),
                    model.get("experiment_name", "legacy") or "legacy",
                    model.get("model_type", "unknown"),
                    model.get("library", "unknown"),
                    f"{model.get('file_size_mb', 0):.1f}",
                    "✅ Available"
                )

        console.print(table)

        # Summary
        total_size = sum(model.get('file_size_mb', 0) for model in models)
        console.print(f"\n📊 Total: {len(models)} models, {total_size:.1f} MB storage")

    except Exception as e:
        console.print(f"❌ Error listing models: {e}", style="red")


@model_app.command("info")
def model_info(
    seed: int = typer.Argument(..., help="Experiment seed of the model"),
    experiment_name: str = typer.Option(..., "--name", help="Experiment name (required)"),
    show_config: bool = typer.Option(False, "--config", help="Show full configuration")
):
    """📊 Show detailed information about a specific model"""

    try:
        info = get_model_info(seed, experiment_name)

        # Create info panel
        experiment_info = info.get("experiment", {})

        details = f"""[bold cyan]Model Information[/bold cyan]

🎲 [bold]Experiment Seed:[/bold] {seed}
🤖 [bold]Model Type:[/bold] {info.get('model_type', 'unknown')}
📚 [bold]Library:[/bold] {info.get('library', 'unknown')}
📝 [bold]Experiment Name:[/bold] {info.get('experiment_name', 'legacy') or 'legacy'}
💾 [bold]File Size:[/bold] {info.get('file_size_mb', 0):.1f} MB
⏱️  [bold]Training Time:[/bold] {info.get('training_time', 0):.1f} seconds
📅 [bold]Saved At:[/bold] {info.get('saved_at', 'unknown')}
📂 [bold]File Path:[/bold] {info.get('file_path', 'unknown')}

[bold yellow]Experiment Details[/bold yellow]
🔬 [bold]Experiment ID:[/bold] {experiment_info.get('id', 'unknown')}
📋 [bold]Name:[/bold] {experiment_info.get('name', 'unknown')}
👤 [bold]Researcher:[/bold] {experiment_info.get('researcher', 'unknown')}
📊 [bold]Training Data Shape:[/bold] {info.get('training_data_shape', 'unknown')}
🗓️  [bold]Timestamp:[/bold] {experiment_info.get('timestamp', 'unknown')}"""
        console.print(Panel(details, title=f"Model {seed}", border_style="cyan"))

        # Show configuration if requested
        if show_config and "config" in info:
            console.print(f"\n📋 Full Configuration:")
            console.print(json.dumps(info["config"], indent=2))

    except ModelNotFoundError:
        console.print(f"❌ Model with seed {seed} not found", style="red")
        console.print("💡 Use 'sdpype model list' to see available models")
    except Exception as e:
        console.print(f"❌ Error getting model info: {e}", style="red")


@model_app.command("validate")
def model_validate(
    seed: int = typer.Argument(..., help="Experiment seed of the model"),
    experiment_name: str = typer.Option(..., "--name", help="Experiment name (required)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation info")
):
    """🔍 Validate model integrity and check if it can be loaded"""

    try:
        console.print(f"🔍 Validating model {experiment_name}_{seed}...")

        validation_result = validate_model(seed, experiment_name)

        if validation_result["valid"]:
            console.print("✅ Model file is valid", style="green")

            if validation_result["loadable"]:
                console.print("✅ Model can be loaded successfully", style="green")

                # Show library-specific validation
                info = validation_result.get("info", {})
                library = info.get("library", "unknown")

                if library == "sdv" and validation_result.get("has_sample_method"):
                    console.print("✅ SDV model has sample() method", style="green")
                elif library == "synthcity" and validation_result.get("has_generate_method"):
                    console.print("✅ Synthcity model has generate() method", style="green")

            else:
                console.print("❌ Model file exists but cannot be loaded", style="red")

        else:
            console.print("❌ Model validation failed", style="red")

        # Show detailed info if requested
        if verbose:
            console.print(f"\n📋 Validation Details:")
            for key, value in validation_result.items():
                if key != "info":  # Don't duplicate info
                    console.print(f"  {key}: {value}")

        # Show error if any
        if validation_result.get("error"):
            console.print(f"\n❌ Error: {validation_result['error']}", style="red")

    except ModelNotFoundError:
        console.print(f"❌ Model with seed {seed} not found", style="red")
        console.print("💡 Use 'sdpype model list' to see available models")
    except Exception as e:
        console.print(f"❌ Error validating model: {e}", style="red")


@model_app.command("status")
def model_status():
    """📊 Show overall model storage status and library support"""

    try:
        # Get library support info
        supported_libs = get_supported_libraries()

        console.print("📚 Library Support Status:")
        for lib, available in supported_libs.items():
            status = "✅ Available" if available else "❌ Not installed"
            style = "green" if available else "red"
            console.print(f"  • {lib}: {status}", style=style)

        # Get models summary
        models = list_saved_models()

        if not models:
            console.print("\n📋 No models found")
            return

        # Calculate statistics
        total_models = len(models)
        total_size_mb = sum(model.get('file_size_mb', 0) for model in models)

        # Group by library
        by_library = {}
        for model in models:
            lib = model.get('library', 'unknown')
            if lib not in by_library:
                by_library[lib] = {'count': 0, 'size_mb': 0}
            by_library[lib]['count'] += 1
            by_library[lib]['size_mb'] += model.get('file_size_mb', 0)

        # Group by model type
        by_type = {}
        for model in models:
            model_type = model.get('model_type', 'unknown')
            if model_type not in by_type:
                by_type[model_type] = {'count': 0, 'size_mb': 0}
            by_type[model_type]['count'] += 1
            by_type[model_type]['size_mb'] += model.get('file_size_mb', 0)

        console.print(f"\n📊 Model Storage Summary:")
        console.print(f"  • Total models: {total_models}")
        console.print(f"  • Total storage: {total_size_mb:.1f} MB")
        console.print(f"  • Average size: {total_size_mb/total_models:.1f} MB per model")

        # Show breakdown by library
        console.print(f"\n📚 By Library:")
        for lib, stats in by_library.items():
            console.print(f"  • {lib}: {stats['count']} models, {stats['size_mb']:.1f} MB")

        # Show breakdown by model type
        console.print(f"\n🤖 By Model Type:")
        for model_type, stats in by_type.items():
            console.print(f"  • {model_type}: {stats['count']} models, {stats['size_mb']:.1f} MB")

        # Storage recommendations
        console.print(f"\n💡 Storage Recommendations:")
        if total_size_mb > 1000:  # > 1GB
            console.print(f"  • Consider cleaning old models: sdpype model clean --older-than 30")
        if len(models) > 10:
            console.print(f"  • Keep only recent models: sdpype model clean --keep-latest 5")
        if total_size_mb < 100:  # < 100MB
            console.print(f"  • Storage usage is efficient ✅")

    except Exception as e:
        console.print(f"❌ Error getting model status: {e}", style="red")


# EVALUATION COMMANDS

@eval_app.command("downstream")
def eval_downstream(
    seed: Optional[int] = typer.Option(None, "--seed", help="Experiment seed (default: from params.yaml)"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column for ML tasks"),
    task_type: Optional[str] = typer.Option(None, "--task-type", help="'classification' or 'regression'"),
    models: Optional[str] = typer.Option(None, "--models", help="Comma-separated list of models to test"),
    force: bool = typer.Option(False, "--force", help="Force re-evaluation")
):
    """🎯 Run downstream task evaluation (ML performance comparison)"""
    
    console.print("🎯 Running Downstream Task Evaluation...")
    
    # Determine if we need parameter overrides
    has_overrides = any([seed, target, task_type, models])
    
    if has_overrides:
        # Use dvc exp run when we have parameter overrides
        cmd = ["dvc", "exp", "run", "-s", "evaluate_downstream"]
        console.print("📊 Using DVC experiments with parameter overrides...")
    else:
        # Use dvc repro when no parameter overrides
        cmd = ["dvc", "repro", "-s", "evaluate_downstream"]
        console.print("📊 Using DVC repro with current parameters...")
    
    if force:
        cmd.append("--force")
    
    # Add parameter overrides (only works with dvc exp run)
    if has_overrides:
        if seed:
            cmd.extend(["--set-param", f"experiment.seed={seed}"])
        
        if target:
            cmd.extend(["--set-param", f"evaluation.downstream_tasks.target_column={target}"])
        
        if task_type:
            cmd.extend(["--set-param", f"evaluation.downstream_tasks.task_type={task_type}"])
        
        if models:
            model_list = [m.strip() for m in models.split(",")]
            cmd.extend(["--set-param", f"evaluation.downstream_tasks.models={model_list}"])
        
        # Always enable downstream evaluation when using overrides
        cmd.extend(["--set-param", "evaluation.downstream_tasks.enabled=true"])
    
    console.print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        console.print("✅ Downstream evaluation completed!", style="green")
        
        # Show results summary
        _show_downstream_summary(seed)
        
    else:
        console.print("❌ Downstream evaluation failed!", style="red")
        console.print("\n💡 Troubleshooting tips:")
        console.print("  • Make sure you have run the full pipeline first: dvc repro")
        console.print("  • Check that synthetic data exists for the specified seed")
        console.print("  • Verify the target column exists in your data")
        raise typer.Exit(1)


@eval_app.command("results")
def eval_results(
    seed: Optional[int] = typer.Option(None, "--seed", help="Experiment seed"),
    show_report: bool = typer.Option(False, "--report", help="Show full text report"),
    compare_seeds: Optional[str] = typer.Option(None, "--compare", help="Compare multiple seeds (comma-separated)")
):
    """📊 Show evaluation results"""
    
    if compare_seeds:
        # Compare multiple experiments
        seeds = [int(s.strip()) for s in compare_seeds.split(",")]
        _compare_downstream_results(seeds)
    elif seed:
        # Show single experiment results
        _show_downstream_summary(seed, show_report)
    else:
        # Show all available results
        _show_all_evaluation_results()


@eval_app.command("status")
def eval_status():
    """📊 Show evaluation pipeline status"""
    
    console.print("📊 Evaluation Pipeline Status:")
    
    # Check which evaluation stages have been run
    metrics_dir = Path("experiments/metrics")
    if not metrics_dir.exists():
        console.print("📋 No evaluation results found", style="yellow")
        return
    
    # Collect evaluation status by seed
    seeds_status = {}
    
    for metrics_file in metrics_dir.glob("*.json"):
        # Parse filename to extract seed and evaluation type
        name = metrics_file.stem
        if "_" in name:
            filename_parts = name.split("_")

            # Simplified parsing for new naming scheme
            try:
                seed = int(filename_parts[-1])  # Last part is always seed

                # Map specific file patterns to evaluation types
                if name.startswith("quality_comparison_"):
                    eval_type = "quality_comparison"
                elif name.startswith("statistical_similarity_"):
                    eval_type = "statistical_similarity"
                elif name.startswith("downstream_performance_"):
                    eval_type = "downstream_performance"
                else:
                    # For other files, use old logic
                    eval_type = "_".join(filename_parts[:-1])

                if seed not in seeds_status:
                    seeds_status[seed] = {}

                seeds_status[seed][eval_type] = "✅"

            except ValueError:
                continue

    if not seeds_status:
        console.print("📋 No evaluation results found", style="yellow")
        return

    # Create status table
    table = Table(title="Evaluation Pipeline Status")
    table.add_column("Seed", style="cyan")
    table.add_column("Intrinsic Quality", style="yellow")
    table.add_column("Statistical Similarity", style="magenta")
    table.add_column("Downstream Tasks", style="green")
    table.add_column("Status", style="white")

    for seed in sorted(seeds_status.keys()):
        status = seeds_status[seed]

        intrinsic = status.get("quality_comparison", "❌")
        statistical = status.get("statistical_similarity", "❌")
        downstream = status.get("downstream_performance", "❌")

        # Overall status
        complete_count = sum(1 for s in [intrinsic, statistical, downstream] if s == "✅")
        if complete_count == 3:
            overall = "🎉 Complete"
        elif complete_count >= 2:
            overall = "⚠️  Partial"
        else:
            overall = "❌ Minimal"

        table.add_row(
            str(seed),
            intrinsic,
            statistical,
            downstream,
            overall
        )

    console.print(table)

    # Summary
    total_seeds = len(seeds_status)
    complete_seeds = sum(1 for status in seeds_status.values() 
                        if all(s == "✅" for s in [
                            status.get("quality_comparison", "❌"),
                            status.get("statistical_similarity", "❌"),
                            status.get("downstream_performance", "❌")
                        ]))
    
    console.print(f"\n📊 Summary:")
    console.print(f"  • Total experiments: {total_seeds}")
    console.print(f"  • Complete evaluations: {complete_seeds}")
    console.print(f"  • Partial evaluations: {total_seeds - complete_seeds}")
    
    if complete_seeds < total_seeds:
        console.print(f"\n💡 To complete missing evaluations:")
        console.print(f"  sdpype eval downstream --seed <SEED>")
        console.print(f"  dvc repro -s compare_quality")


# EXPERIMENT MANAGEMENT COMMANDS

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
    """Show summary of completed experiments with model status and downstream results"""

    metrics_dir = Path("experiments/metrics")
    if not metrics_dir.exists():
        console.print("📋 No experiments found. Run: sdpype setup", style="yellow")
        return

    # Collect experiment results
    experiments = []

    # Handle both old format (training_seed.json) and new format (training_name_seed.json)
    for training_file in metrics_dir.glob("training_*.json"):
        try:
            with open(training_file) as f:
                training_data = json.load(f)

            seed = training_data.get("seed", "unknown")
            generation_file = metrics_dir / f"generation_{seed}.json"

            # Extract experiment name from filename if using new format
            filename_stem = training_file.stem  # e.g., "training_baseline_46"
            filename_parts = filename_stem.split("_")

            if len(filename_parts) >= 3:  # training_name_seed format
                experiment_name = "_".join(filename_parts[1:-1])  # everything between "training" and seed
                generation_file = metrics_dir / f"generation_{experiment_name}_{seed}.json"
                downstream_file = metrics_dir / f"downstream_performance_{experiment_name}_{seed}.json"
            else:  # old training_seed format
                experiment_name = None
                generation_file = metrics_dir / f"generation_{seed}.json"
                downstream_file = metrics_dir / f"downstream_performance_{seed}.json"

            exp_data = {
                "seed": seed,
                "experiment_name": experiment_name or "legacy",
                "model": training_data.get("model_type", "unknown"),
                "library": training_data.get("library", "unknown"),
                "train_time": training_data.get("training_time", 0),
                "gen_time": 0,
                "samples": 0,
                "timestamp": training_data.get("timestamp", "unknown")
            }

            # Add model status information using serialization module
            try:
                # Only try to get model info if we have experiment name
                if experiment_name and experiment_name != "legacy":
                    model_info = get_model_info(seed, experiment_name)
                    exp_data.update({
                        "model_status": "✅ Available",
                        "model_size_mb": f"{model_info.get('file_size_mb', 0):.1f}",
                        "model_library": model_info.get('library', 'unknown'),
                    })
                else:
                    exp_data.update({
                       "model_status": "❌ Legacy",
                       "model_size_mb": "0.0",
                       "model_library": "unknown",
                    })
            except (SerializationError, ModelNotFoundError):
                exp_data.update({
                    "model_status": "❌ Missing",
                    "model_size_mb": "0.0",
                    "model_library": "unknown",
                })

            # Add downstream evaluation results if available
            if downstream_file.exists():
                try:
                    with open(downstream_file) as f:
                        downstream_data = json.load(f)
                    exp_data.update({
                        "downstream_utility": f"{downstream_data.get('overall_utility_score', 0):.3f}",
                        "downstream_status": "✅ Complete"
                    })
                except Exception:
                    exp_data.update({"downstream_utility": "0.000", "downstream_status": "❌ Error"})
            else:
                exp_data.update({"downstream_utility": "-", "downstream_status": "⚠️  Missing"})

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
    table = Table(title="Experiments & Models Summary")
    table.add_column("Seed", style="cyan")
    table.add_column("Experiment", style="yellow")
    table.add_column("Model", style="magenta")
    table.add_column("Library", style="yellow")
    table.add_column("Model Status", style="white")
    table.add_column("Size (MB)", style="blue", justify="right")
    table.add_column("ML Utility", style="magenta", justify="right")
    table.add_column("Train Time", style="green")
    table.add_column("Gen Time", style="green")
    table.add_column("Samples", style="white", justify="right")

    for exp in sorted(experiments, key=lambda x: str(x["seed"])):
        table.add_row(
            str(exp["seed"]),
            exp.get("experiment_name", "legacy"),
            exp["model"],
            exp.get("model_library", exp["library"]),  # Prefer model info over training metrics
            exp.get("model_status", "Unknown"),
            exp.get("model_size_mb", "0.0"),
            exp.get("downstream_utility", "-"),
            f"{exp['train_time']:.1f}s",
            f"{exp['gen_time']:.1f}s",
            str(exp["samples"])
        )

    console.print(table)

    # Add summary statistics
    total_experiments = len(experiments)
    available_models = len([exp for exp in experiments if exp.get("model_status") == "✅ Available"])
    completed_downstream = len([exp for exp in experiments if exp.get("downstream_status") == "✅ Complete"])
    missing_models = total_experiments - available_models
    total_size_mb = sum(float(exp.get("model_size_mb", 0)) for exp in experiments)

    console.print(f"\n📊 Summary:")
    console.print(f"  • Total experiments: {total_experiments}")
    console.print(f"  • Available models: {available_models}")
    console.print(f"  • Downstream evaluations: {completed_downstream}")
    if missing_models > 0:
        console.print(f"  • Missing models: {missing_models}", style="yellow")
    console.print(f"  • Total model storage: {total_size_mb:.1f} MB")

    # Show average utility score for completed downstream evaluations
    completed_utilities = [float(exp.get("downstream_utility", 0)) for exp in experiments if exp.get("downstream_utility") != "-" and exp.get("downstream_utility") != "0.000"]
    if completed_utilities:
        avg_utility = sum(completed_utilities) / len(completed_utilities)
        console.print(f"  • Average ML utility: {avg_utility:.3f}")

    # Show hint about missing models
    if missing_models > 0:
        console.print(f"\n💡 Some models are missing. Run training again:", style="yellow")
        missing_seeds = [str(exp["seed"]) for exp in experiments if exp.get("model_status") == "❌ Missing"]
        console.print(f"  dvc repro --set-param experiment.seed={missing_seeds[0]}")

    # Show hint about missing downstream evaluations
    missing_downstream = total_experiments - completed_downstream
    if missing_downstream > 0:
        console.print(f"\n💡 Complete downstream evaluations:", style="yellow")
        console.print(f"  sdpype eval downstream --seed <SEED>")


# UTILITY FUNCTIONS FOR EVALUATION

def _show_downstream_summary(seed: Optional[int], show_report: bool = False):
    """Show summary of downstream evaluation results"""

    # Determine experiment name by looking for downstream files with this seed
    experiment_name = None
    if seed is None:
        # Get seed from params.yaml
        import yaml
        try:
            with open("params.yaml") as f:
                params = yaml.safe_load(f)
            seed = params.get("experiment", {}).get("seed", None)
            experiment_name = params.get("experiment", {}).get("name", None)
        except:
            console.print("❌ Could not determine seed", style="red")
            return
    else:
        # Try to find experiment name from existing files
        metrics_dir = Path("experiments/metrics")
        downstream_files = list(metrics_dir.glob(f"downstream_performance_*_{seed}.json"))
        if downstream_files:
            # Extract experiment name from filename: downstream_performance_name_seed.json
            filename_parts = downstream_files[0].stem.split("_")
            if len(filename_parts) >= 4:  # downstream_performance_name_seed
                experiment_name = "_".join(filename_parts[2:-1])  # everything between "performance" and seed

    # Check if results exist
    if experiment_name:
        results_file = Path(f"experiments/metrics/downstream_performance_{experiment_name}_{seed}.json")
        report_file = Path(f"experiments/metrics/downstream_report_{experiment_name}_{seed}.txt")
    else:
        # Try old format first, then look for any file with this seed
        results_file = Path(f"experiments/metrics/downstream_performance_{seed}.json")
        report_file = Path(f"experiments/metrics/downstream_report_{seed}.txt")

        if not results_file.exists():
            # Look for new format files
            metrics_dir = Path("experiments/metrics")
            possible_files = list(metrics_dir.glob(f"downstream_performance_*_{seed}.json"))
            if possible_files:
                results_file = possible_files[0]
                report_file = metrics_dir / f"downstream_report_{possible_files[0].stem.replace('downstream_performance_', '')}.txt"

    if not results_file.exists():
        console.print(f"❌ No downstream results found for seed {seed}", style="red")
        if experiment_name:
            console.print(f"💡 Looking for: downstream_performance_{experiment_name}_{seed}.json", style="yellow")
        console.print("💡 Run: sdpype eval downstream")
        return

    try:
        with open(results_file) as f:
            results = json.load(f)

        metadata = results["metadata"]
        overall_utility = results["overall_utility_score"]

        console.print(f"\n🎯 Downstream Evaluation Results (Seed {seed})")
        console.print(f"Task: {metadata['task_type']} on '{metadata['target_column']}'")
        console.print(f"Overall Utility Score: {overall_utility:.3f}")

        # Show model results
        table = Table(title="Model Performance Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Original", style="green")
        table.add_column("Synthetic", style="yellow")
        table.add_column("Utility", style="magenta")

        task_type = metadata["task_type"]
        primary_metric = "accuracy" if task_type == "classification" else "r2"

        for model_name in metadata["models_evaluated"]:
            orig_perf = results["original_performance"].get(model_name, {}).get(primary_metric, 0.0)
            synth_perf = results["synthetic_performance"].get(model_name, {}).get(primary_metric, 0.0)
            utility = results["utility_scores"].get(model_name, 0.0)

            table.add_row(
                model_name,
                f"{orig_perf:.3f}",
                f"{synth_perf:.3f}",
                f"{utility:.3f}"
            )

        console.print(table)

        # Show interpretation
        if overall_utility >= 0.9:
            console.print("🎉 Excellent synthetic data quality!", style="green")
        elif overall_utility >= 0.8:
            console.print("✅ Good synthetic data quality", style="green")
        elif overall_utility >= 0.7:
            console.print("⚠️  Moderate synthetic data quality", style="yellow")
        else:
            console.print("❌ Poor synthetic data quality", style="red")

        # Show full report if requested
        if show_report and report_file.exists():
            console.print("\n📋 Full Report:")
            console.print("-" * 60)
            with open(report_file) as f:
                console.print(f.read())

    except Exception as e:
        console.print(f"❌ Error reading results: {e}", style="red")


def _compare_downstream_results(seeds: List[int]):
    """Compare downstream results across multiple seeds"""

    console.print(f"📊 Comparing Downstream Results Across Seeds: {seeds}")

    all_results = {}

    for seed in seeds:
        # Try both old and new format files
        results_file = Path(f"experiments/metrics/downstream_performance_{seed}.json")
        experiment_name = "legacy"  # Default for old format

        if not results_file.exists():
            # Look for new format files
            metrics_dir = Path("experiments/metrics")
            possible_files = list(metrics_dir.glob(f"downstream_performance_*_{seed}.json"))
            if possible_files:
                results_file = possible_files[0]
                # Extract experiment name from filename: downstream_performance_name_seed.json
                filename_parts = results_file.stem.split("_")
                if len(filename_parts) >= 4:  # downstream_performance_name_seed
                    experiment_name = "_".join(filename_parts[2:-1])  # everything between "performance" and seed

        if results_file.exists():
            try:
                with open(results_file) as f:
                    results_data = json.load(f)
                    results_data["_experiment_name"] = experiment_name  # Store experiment name with results
                    all_results[seed] = results_data
            except Exception as e:
                console.print(f"⚠️  Error reading seed {seed}: {e}", style="yellow")
        else:
            console.print(f"⚠️  No results found for seed {seed}", style="yellow")

    if not all_results:
        console.print("❌ No valid results found", style="red")
        return

    # Create comparison table
    table = Table(title="Downstream Performance Comparison")
    table.add_column("Seed", style="cyan")
    table.add_column("Experiment", style="yellow")
    table.add_column("Task Type", style="yellow")
    table.add_column("Overall Utility", style="green")
    table.add_column("Best Model", style="magenta")
    table.add_column("Status", style="white")

    for seed, results in all_results.items():
        metadata = results["metadata"]
        overall_utility = results["overall_utility_score"]
        experiment_name = results.get("_experiment_name", "unknown")

        # Find best model
        if results["utility_scores"]:
            best_model = max(results["utility_scores"].items(), key=lambda x: x[1])
            best_model_name = best_model[0]
            best_utility = best_model[1]
        else:
            best_model_name = "None"
            best_utility = 0.0

        # Status
        if overall_utility >= 0.8:
            status = "✅ Good"
        elif overall_utility >= 0.7:
            status = "⚠️  OK"
        else:
            status = "❌ Poor"

        table.add_row(
            str(seed),
            experiment_name,
            metadata["task_type"],
            f"{overall_utility:.3f}",
            f"{best_model_name} ({best_utility:.3f})",
            status
        )

    console.print(table)


def _show_all_evaluation_results():
    """Show overview of all evaluation results"""

    metrics_dir = Path("experiments/metrics")
    if not metrics_dir.exists():
        console.print("📋 No evaluation results found", style="yellow")
        return

    # Find all downstream results (both old and new format)
    downstream_files = list(metrics_dir.glob("downstream_performance_*.json"))

    if not downstream_files:
        console.print("📋 No downstream evaluation results found", style="yellow")
        console.print("💡 Run: sdpype eval downstream")
        return

    console.print(f"📊 Found {len(downstream_files)} downstream evaluation results")

    # Extract seeds and show comparison
    seeds = []
    for file in downstream_files:
        try:
            # Handle both old format (downstream_performance_seed.json) and new format (downstream_performance_name_seed.json)
            filename_parts = file.stem.split("_")
            if len(filename_parts) >= 3:  # At least downstream_performance_seed
                seed = int(filename_parts[-1])  # Last part is always seed
                seeds.append(seed)
        except ValueError:
            continue

    if seeds:
        # Remove duplicates and sort
        unique_seeds = sorted(list(set(seeds)))
        _compare_downstream_results(unique_seeds)


def _display_experiment_comparison_table(all_results: dict, focus_metric: str):
    """Display formatted comparison table for named experiments"""

    table = Table(title="Experiment Comparison")
    table.add_column("Experiment", style="cyan", no_wrap=True)

    # Add columns based on focus metric
    if focus_metric in ["quality", "all"]:
        table.add_column("Quality↑", style="green")
    if focus_metric in ["statistical", "all"]:
        table.add_column("Statistical↑", style="blue")
    if focus_metric in ["downstream", "all"]:
        table.add_column("Downstream↑", style="magenta")
    if focus_metric == "all":
        table.add_column("Combined↑", style="yellow")
        table.add_column("Rank", style="red")

    # Calculate scores for each experiment
    experiment_scores = []

    for exp_name, results in all_results.items():
        if not results:  # Skip experiments with no data
            continue

        scores = {'experiment': exp_name}

        # Quality score
        if 'quality' in results and focus_metric in ["quality", "all"]:
            scores['quality'] = results['quality']['overall_score_comparison']['quality_preservation_rate']

        # Statistical score
        if 'statistical' in results and focus_metric in ["statistical", "all"]:
            scores['statistical'] = results['statistical']['overall_similarity_score']

        # Downstream score
        if 'downstream' in results and focus_metric in ["downstream", "all"]:
            scores['downstream'] = results['downstream']['overall_utility_score']

        # Combined score (weighted average)
        if focus_metric == "all":
            quality_score = scores.get('quality', 0)
            statistical_score = scores.get('statistical', 0)
            downstream_score = scores.get('downstream', 0)

            # Weight: 30% quality, 30% statistical, 40% downstream
            combined = (0.3 * quality_score + 0.3 * statistical_score + 0.4 * downstream_score)
            scores['combined'] = combined

        experiment_scores.append(scores)

    # Sort by combined score or specific metric
    if focus_metric == "all":
        experiment_scores.sort(key=lambda x: x.get('combined', 0), reverse=True)
    else:
        experiment_scores.sort(key=lambda x: x.get(focus_metric, 0), reverse=True)

    # Add rows to table
    for i, exp_scores in enumerate(experiment_scores, 1):
        row_data = [exp_scores['experiment']]

        if focus_metric in ["quality", "all"] and 'quality' in exp_scores:
            row_data.append(f"{exp_scores['quality']:.1%}")
        if focus_metric in ["statistical", "all"] and 'statistical' in exp_scores:
            row_data.append(f"{exp_scores['statistical']:.3f}")
        if focus_metric in ["downstream", "all"] and 'downstream' in exp_scores:
            row_data.append(f"{exp_scores['downstream']:.3f}")
        if focus_metric == "all":
            if 'combined' in exp_scores:
                row_data.append(f"{exp_scores['combined']:.3f}")
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
            row_data.append(rank_emoji)

        table.add_row(*row_data)

    console.print(table)

    # Show winner summary
    if len(experiment_scores) >= 2:
        winner = experiment_scores[0]
        console.print(f"\n🏆 Overall Winner: {winner['experiment']}", style="bold green")

        if focus_metric == "all" and 'combined' in winner:
            console.print(f"Combined Score: {winner['combined']:.3f}")
        elif focus_metric != "all" and focus_metric in winner:
            console.print(f"{focus_metric.title()} Score: {winner[focus_metric]:.3f}")


def _generate_experiment_comparison_report(all_results: dict) -> str:
    """Generate detailed comparison report"""

    report = []
    report.append("# Experiment Comparison Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Experiments compared: {', '.join(all_results.keys())}")
    report.append("")

    for exp_name, results in all_results.items():
        if not results:  # Skip experiments with no data
            continue

        report.append(f"## {exp_name}")

        if 'quality' in results:
            qp = results['quality']['overall_score_comparison']['quality_preservation_rate']
            report.append(f"- Quality Preservation: {qp:.1%}")

        if 'statistical' in results:
            ss = results['statistical']['overall_similarity_score']
            report.append(f"- Statistical Similarity: {ss:.3f}")

        if 'downstream' in results:
            du = results['downstream']['overall_utility_score']
            report.append(f"- Downstream Utility: {du:.3f}")

            # Best model details
            best_model = max(results['downstream']['model_results'].items(),
                           key=lambda x: x[1]['performance_score'])
            report.append(f"- Best ML Model: {best_model[0]} ({best_model[1]['performance_score']:.3f})")

        report.append("")

    return "\n".join(report)


@eval_app.command("compare-experiments")
def compare_named_experiments(
    experiments: str = typer.Argument(..., help="Comma-separated experiment_name_seed pairs (e.g., 'ctgan_baseline_1,tvae_baseline_1')"),
    metric: str = typer.Option("all", "--metric", "-m", help="Focus on specific metric (quality, statistical, downstream, all)"),
    save_report: bool = typer.Option(False, "--save", help="Save comparison report")
):
    """🆚 Compare experiments using experiment_name_seed format"""

    exp_list = [exp.strip() for exp in experiments.split(",")]

    console.print(f"🔍 Comparing experiments: {', '.join(exp_list)}")

    # Parse and validate experiment names
    parsed_experiments = []
    for exp in exp_list:
        # Parse experiment_name_seed format
        parts = exp.rsplit('_', 1)  # Split from right to get last part as seed
        if len(parts) != 2:
            console.print(f"❌ Invalid format: {exp}. Use format: experiment_name_seed", style="red")
            continue

        exp_name, seed = parts[0], parts[1]

        try:
            seed_int = int(seed)
            parsed_experiments.append((exp, exp_name, seed_int))
        except ValueError:
            console.print(f"❌ Invalid seed in {exp}: {seed}", style="red")
            continue

    if not parsed_experiments:
        console.print("❌ No valid experiments to compare!", style="red")
        return

    # Load results for each experiment
    all_results = {}
    metrics_dir = Path("experiments/metrics")

    if not metrics_dir.exists():
        console.print("❌ No metrics directory found!", style="red")
        console.print("💡 Run evaluations first: uv run sdpype stage compare_quality")
        return

    for exp_full, exp_name, seed_int in parsed_experiments:
        console.print(f"📊 Loading results for: {exp_full}")

        # Load results
        results = {}

        # Quality comparison
        quality_file = metrics_dir / f"quality_comparison_{exp_name}_{seed_int}.json"
        if quality_file.exists():
            with open(quality_file) as f:
                results['quality'] = json.load(f)
                console.print(f"  ✅ Quality data loaded")

        # Statistical similarity
        statistical_file = metrics_dir / f"statistical_similarity_{exp_name}_{seed_int}.json"
        if statistical_file.exists():
            with open(statistical_file) as f:
                results['statistical'] = json.load(f)
                console.print(f"  ✅ Statistical data loaded")

        # Downstream performance
        downstream_file = metrics_dir / f"downstream_performance_{exp_name}_{seed_int}.json"
        if downstream_file.exists():
            with open(downstream_file) as f:
                results['downstream'] = json.load(f)
                console.print(f"  ✅ Downstream data loaded")

        if not results:
            console.print(f"  ⚠️  No evaluation data found for {exp_full}")

        all_results[exp_full] = results

    # Validate that we have some results to compare
    experiments_with_data = [exp for exp, results in all_results.items() if results]

    if not experiments_with_data:
        console.print("❌ No evaluation results found for any experiment!", style="red")
        console.print("💡 Run evaluations first:")
        console.print("  uv run sdpype stage compare_quality")
        console.print("  uv run sdpype eval downstream --target <column>")
        return

    if len(experiments_with_data) == 1:
        console.print("⚠️  Only one experiment has data - comparison needs at least 2", style="yellow")
        console.print(f"Available: {experiments_with_data[0]}")
        return

    console.print(f"✅ Found data for {len(experiments_with_data)} experiments")

    # Create comparison table
    _display_experiment_comparison_table(all_results, metric)

    # Save report if requested
    if save_report:
        report = _generate_experiment_comparison_report(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"experiments/comparison_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write(report)

        console.print(f"📄 Comparison report saved: {report_path}")


if __name__ == "__main__":
    app()
