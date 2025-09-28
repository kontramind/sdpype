# sdpype/core/experiment.py - Experiment management with purge
"""
Core experiment management functionality
"""

import subprocess
import shutil
from pathlib import Path

def purge_repository(confirm: bool = False, keep_raw_data: bool = True, keep_cache: bool = False):
    """Purge all experiments, models, and DVC state (DESTRUCTIVE!)"""
    
    # Import console utilities
    from sdpype.utils.console import console, print_warning, print_success, confirm_dangerous_action, print_completion_panel
    
    print_warning("🧹 Repository Purge", "This will permanently delete experiment data!")
    
    # List what will be deleted
    items_to_delete = [
        "📊 All experiment metrics (experiments/metrics/)",
        "🤖 All trained models (experiments/models/)", 
        "🎲 All synthetic data (experiments/data/synthetic/)",
        "🔄 DVC pipeline lock files",
        "📋 DVC experiment history",
        "🐍 Python cache files"
    ]
    
    if not keep_raw_data:
        items_to_delete.append("📁 Raw data files (experiments/data/raw/)")
    
    if not keep_cache:
        items_to_delete.append("💾 DVC cache (.dvc/cache/)")
    
    console.print("\nThis will delete:")
    for item in items_to_delete:
        console.print(f"  • {item}")
    
    preserved_items = [
        "🐍 Source code (sdpype/ folder)",
        "⚙️ Configuration (params.yaml, dvc.yaml)",
        "📈 Processed data (experiments/data/processed/)",
        "📋 Project files (pyproject.toml, README.md)"
    ]
    
    if keep_raw_data:
        preserved_items.append("📁 Raw data files")
    if keep_cache:
        preserved_items.append("💾 DVC cache")
    
    console.print("\n✅ Will be preserved:", style="bold green")
    for item in preserved_items:
        console.print(f"  • {item}")
    
    # Confirmation
    if not confirm:
        if not confirm_dangerous_action("Repository Purge", items_to_delete):
            console.print("❌ Cancelled")
            return
    
    # Perform the purge
    console.print("\n🧹 Starting purge...", style="bold red")
    
    _purge_experiment_data(keep_raw_data)
    _purge_dvc_state(keep_cache) 
    _clean_python_cache()
    
    print_completion_panel(
        "Purge Complete",
        [
            "Run: [bold]sdpype setup[/bold] (recreate structure)",
            "Add your data to experiments/data/raw/", 
            "Start fresh with your experiments!"
        ],
        "✨ Repository has been purged!"
    )

def _purge_experiment_data(keep_raw_data: bool):
    """Remove experiment data directories"""
    
    from sdpype.utils.console import console
    
    console.print("📊 Removing experiment data...")
    
    dirs_to_remove = [
        "experiments/metrics",
        "experiments/models",
        "experiments/data/synthetic"
    ]
    
    if not keep_raw_data:
        dirs_to_remove.append("experiments/data/raw")
    
    for dir_path in dirs_to_remove:
        path = Path(dir_path)
        if path.exists():
            shutil.rmtree(path)
            console.print(f"  ✅ Removed {dir_path}/")

def _purge_dvc_state(keep_cache: bool):
    """Remove DVC state and lock files"""
    
    from sdpype.utils.console import console
    
    console.print("🔄 Cleaning DVC state...")
    
    # Remove DVC lock and temporary files
    dvc_files_to_remove = [
        "dvc.lock",
        ".dvcignore"
    ]
    
    for dvc_file in dvc_files_to_remove:
        file_path = Path(dvc_file)
        if file_path.exists():
            file_path.unlink()
            console.print(f"  ✅ Removed {dvc_file}")
    
    # Remove DVC directories
    dvc_dirs_to_remove = [".dvc/tmp"]
    if not keep_cache:
        dvc_dirs_to_remove.append(".dvc/cache")
    
    for dvc_dir in dvc_dirs_to_remove:
        dir_path = Path(dvc_dir)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            console.print(f"  ✅ Removed {dvc_dir}/")
    
    # Clean DVC experiments
    console.print("  🔄 Cleaning DVC experiment history...")
    cleanup_commands = [
        ["dvc", "exp", "remove", "--all-commits"],
        ["dvc", "exp", "gc", "--workspace", "--force"],
    ]

    for cmd in cleanup_commands:
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"    ✅ {' '.join(cmd)}")
        except subprocess.CalledProcessError:
            console.print(f"    ⚠️  Failed: {' '.join(cmd)}")

def _clean_python_cache():
    """Clean Python cache directories"""
    
    from sdpype.utils.console import console
    
    console.print("🐍 Cleaning Python cache...")
    
    cache_dirs = [
        "__pycache__",
        "sdpype/__pycache__",
        ".pytest_cache", 
        "outputs",  # Hydra outputs
        ".hydra"
    ]
    
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            console.print(f"  ✅ Removed {cache_dir}/")
