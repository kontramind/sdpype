# sdpype/core/pipeline.py - Pipeline execution logic
"""
DVC pipeline execution functionality
"""

import shutil
import subprocess
from pathlib import Path
from rich.console import Console

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import tempfile
import os

console = Console()


def _resolve_params_templates():
    """Auto-resolve experiment name templates in params.yaml before DVC execution"""

    params_file = Path("params.yaml")
    if not params_file.exists():
        console.print("⚠️  params.yaml not found", style="yellow")
        return False

    # Backup original params.yaml
    backup_file = Path("params.yaml.backup")
    try:
        shutil.copy2(params_file, backup_file)
        console.print("📋 Backed up params.yaml", style="dim")
    except Exception as e:
        console.print(f"⚠️  Could not backup params.yaml: {e}", style="yellow")
        # Continue anyway - not critical

    # Load with round-trip preservation (order, comments, quotes) - like pypyr scripts
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=0)

    with params_file.open("r", encoding="utf-8") as f:
        params = yaml.load(f)

    def set_in(d, path, value):
        """Safely set nested key path like ['experiment','name'] even if missing."""
        cur = d
        for k in path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = CommentedMap()
            cur = cur[k]
        cur[path[-1]] = value

    # Check if we need to resolve templates
    experiment = params.get('experiment', {})
    name = experiment.get('name')
    name_template = experiment.get('name_template')

    # If name is empty/None or contains templates, resolve it
    if not name or (isinstance(name, str) and '{' in name and '}' in name):
        # Use template or auto-generate
        template = name_template or name or "{sdg.library}_{sdg.model_type}_{seed}"

        # Resolve template
        try:
            # Create a custom formatting class that supports dot notation
            class DotDict:
                def __init__(self, d):
                    self.__dict__.update(d)
                    for k, v in d.items():
                        if isinstance(v, dict):
                            setattr(self, k, DotDict(v))

            # Convert CommentedMaps to regular dicts for dot notation
            def to_dict(obj):
                if hasattr(obj, 'items'):  # Handle CommentedMap and regular dict
                    return {k: to_dict(v) for k, v in obj.items()}
                else:
                    return obj

            # Create dot-accessible objects
            sdg_dict = to_dict(params.get('sdg', {}))
            experiment_dict = to_dict(params.get('experiment', {}))

            resolved_name = template.format(
                sdg=DotDict(sdg_dict),
                experiment=DotDict(experiment_dict)
            )
            console.print(f"🏷️  Auto-resolving template: {template} → {resolved_name}", style="cyan")

            # Update params using safe nested setting
            set_in(params, ["experiment", "name"], resolved_name)

            # Remove template to avoid confusion (if it exists)
            if 'experiment' in params and 'name_template' in params['experiment']:
                del params['experiment']['name_template']

            # Atomic write to avoid partial files (like pypyr scripts)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=str(params_file.parent), encoding="utf-8") as tf:
                yaml.dump(params, tf)
                tmp_name = tf.name
            os.replace(tmp_name, params_file)

            # DEBUG: Verify the file was actually updated
            with open(params_file, 'r') as f:
                verify_content = f.read()
                console.print(f"🐛 DEBUG: File contents after write:", style="dim")
                console.print(f"{verify_content[:300]}...", style="dim")

        except KeyError as e:
            console.print(f"❌ Template error: missing placeholder {e}", style="red")
            return False
        except Exception as e:
            console.print(f"❌ Error updating params.yaml: {e}", style="red")
            return False
    else:
        console.print(f"✅ Experiment name already resolved: {name}", style="green")

    return True

def _restore_params_backup():
    """Restore original params.yaml from backup"""
    backup_file = Path("params.yaml.backup")
    params_file = Path("params.yaml")

    if backup_file.exists():
        shutil.move(backup_file, params_file)
        console.print("📋 Restored original params.yaml", style="dim")

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
        "detection_evaluation",
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
            ("experiments/metrics/statistical_similarity_*.json", "📊 Statistical metrics"),
            ("experiments/metrics/statistical_report_*.txt", "📋 Statistical metric reports")
         ],
        "detection_evaluation": [
            ("experiments/metrics/detection_evaluation_*.json", "🔍 Detection metrics"),
            ("experiments/metrics/detection_report_*.txt", "📋 Detection evaluation reports")
        ],
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

    # Auto-resolve experiment name templates BEFORE calling DVC
    console.print("🔍 Checking experiment configuration...", style="blue")
    if not _resolve_params_templates():
        console.print("❌ Failed to resolve experiment templates", style="red")
        return False

    # Build DVC command - DVC will now read the resolved params.yaml
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
        # Restore original params.yaml
        _restore_params_backup()
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"❌ Pipeline failed with exit code {e.returncode}", style="bold red")
        console.print("💡 Try running with --force or check your configuration", style="yellow")
        # Restore original params.yaml even on failure
        _restore_params_backup()
        return False

    except FileNotFoundError:
        console.print("❌ DVC not found. Make sure DVC is installed:", style="bold red")
        console.print("   pip install dvc", style="yellow")
        # Restore original params.yaml even on failure
        _restore_params_backup()
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
