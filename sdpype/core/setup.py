# sdpype/core/setup.py - Repository setup logic
"""
Repository setup and initialization functions
"""

import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

from sdpype.utils.console import console

def setup_repository_command():
    """Setup repository for experiments (creates directories and sample data)"""
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        console.print("❌ Not in SDPype repository root", style="bold red")
        console.print("💡 Run this command from your sdpype/ directory")
        raise SystemExit(1)

    create_directory_structure()
    initialize_version_control()
    create_sample_data()
    create_dvc_ignore()
    show_completion_message()

def create_directory_structure():
    """Create the experiments directory structure"""
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

def initialize_version_control():
    """Initialize DVC and Git if needed"""
    
    if not Path(".dvc").exists():
        subprocess.run(["dvc", "init"])
        console.print("✅ DVC initialized", style="green")

    if not Path(".git").exists():
        subprocess.run(["git", "init"])  
        console.print("✅ Git initialized", style="green")

def create_sample_data():
    """Create sample dataset in experiments structure"""
    
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples).clip(18, 80).astype(int),
        'income': np.random.lognormal(10, 1, n_samples).clip(20000, 200000).astype(int),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    })

    sample_file = Path("experiments/data/raw/sample_data.csv")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(sample_file, index=False)
    console.print(f"📊 Created sample data: {len(data)} rows at {sample_file}")

def create_dvc_ignore():
    """Create .dvcignore file with proper exclusions"""
    
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

def show_completion_message():
    """Show setup completion message"""
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "✅ Repository setup complete!\n\n"
        "Next steps:\n"
        "• Run: [bold]sdpype models[/bold] (see available models)\n"
        "• Run: [bold]dvc repro[/bold] (run full pipeline)\n"
        "• Run: [bold]sdpype exp run --name 'test' --seed 42[/bold] (run experiment)",
        title="🎉 Setup Complete"
    ))
    