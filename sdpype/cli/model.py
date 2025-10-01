# sdpype/cli/model.py
"""Model management CLI commands"""

from omegaconf import OmegaConf, DictConfig
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
from pathlib import Path

console = Console()

model_app = typer.Typer(
    help="🤖 Manage trained models",
    rich_markup_mode="rich",
    no_args_is_help=True
)

@model_app.command("list")
def list_models(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Custom model directory")
):
    """📋 List all trained models"""

    from sdpype.serialization import list_saved_models, ModelNotFoundError

    try:
        model_dir_path = Path(model_dir) if model_dir else None
        models = list_saved_models(model_dir_path)

        if not models:
            console.print("📭 No trained models found", style="yellow")
            console.print("💡 Run [bold]sdpype pipeline[/bold] to train your first model")
            return

        console.print(f"🤖 Found {len(models)} trained model(s):\n")

        if verbose:
            _show_detailed_model_list(models)
        else:
            _show_compact_model_list(models)

        console.print(f"\n💡 Use [bold]sdpype model info <model_id>[/bold] for details (e.g., 'sdpype model info sdv_copula_gan_test_123')")
        
    except Exception as e:
        console.print(f"❌ Error listing models: {e}", style="red")
        raise typer.Exit(1)

@model_app.command("info")
def model_info(
    model_id: str = typer.Argument(..., help="Model identifier (experiment_name_seed, e.g. 'sdv_copula_gan_test_123')"),
    export: bool = typer.Option(False, "--export", help="Export params.yaml to file"),
    export_file: Optional[str] = typer.Option(None, "--export-file", help="Custom filename for export (default: params_{model_id}.yaml)"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Custom model directory")
):
    """📊 Show detailed information about a specific model"""

    from sdpype.serialization import get_model_info, ModelNotFoundError

    try:
        experiment_name, seed = _parse_model_id(model_id)
        model_dir_path = Path(model_dir) if model_dir else None
        info = get_model_info(seed, experiment_name, model_dir_path)

        # Determine export filename
        export_filename = None
        if export:
            export_filename = export_file if export_file else f"params_{model_id}.yaml"

        _show_model_details(info, seed, experiment_name, export_file=export_filename)

    except ModelNotFoundError as e:
        console.print(f"❌ {e}", style="red")
        console.print("\n💡 Use [bold]sdpype model list[/bold] to see available models")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"❌ Invalid model ID format: {e}", style="red")
        console.print("💡 Use format: experiment_name_seed (e.g., 'baseline_42')")
        raise typer.Exit(1)        
    except Exception as e:
        console.print(f"❌ Error getting model info: {e}", style="red")
        raise typer.Exit(1)

# TODO: Future extensions to add:
# - model validate: Check model file integrity  
# - model clean: Remove trained models with safety checks

def _parse_model_id(model_id: str) -> tuple[str, int]:
    """
    Parse model_id into experiment_name and seed (with new structure)
    
    Args:
        model_id: Format 'library_model_refhash_roothash_trnhash_gen_N_cfghash_seed'
                 (e.g., 'synthcity_ctgan_0cf8e0f5_852ba944_de360e6d_gen_1_68b7c931_51')
        
    Returns:
        tuple: (experiment_name, seed)
        Note: experiment_name includes everything except cfghash and seed
    """
    parts = model_id.split('_')
    # New format has at least 9 parts: library_model_refhash_roothash_trnhash_gen_N_cfghash_seed
    if len(parts) < 9:
        raise ValueError(f"Model ID format invalid (expected at least 9 components): {model_id}")

    try:
        seed = int(parts[-1])
        config_hash = parts[-2]
        generation_num = parts[-3]  # Should be a number
        gen_marker = parts[-4]      # Should be "gen"

        # Validate "gen" marker
        if gen_marker != "gen":
            raise ValueError(f"Expected 'gen' marker at position -4, got: {gen_marker}")

        # Validate generation number
        try:
            int(generation_num)
        except ValueError:
            raise ValueError(f"Invalid generation number: {generation_num}")

        # Experiment name is everything except: _gen_N_cfghash_seed
        experiment_name = '_'.join(parts[:-4])

        # Validate config hash format (8 hex characters)
        if len(config_hash) != 8 or not all(c in '0123456789abcdef' for c in config_hash.lower()):
            raise ValueError(f"Invalid config hash format in model ID: {config_hash}")

        return experiment_name, seed
    except ValueError:
        raise ValueError(f"Invalid model ID format. Expected: experiment_name_config_hash_seed, got: {model_id}")


def _extract_dataset_name(model: dict) -> str:
    """Extract dataset filename from model metadata"""
    try:
        # Get the full config from model metadata
        params = model.get('params', {})
        data_config = params.get('data', {})
        training_file = data_config.get('training_file', '')

        if training_file:
            # Extract just the filename from the full path
            from pathlib import Path
            return Path(training_file).name
        else:
            return "Unknown"

    except Exception:
        return "Unknown"

def _show_compact_model_list(models):
    """Show models in a compact table format"""

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Experiment", style="cyan")
    # table.add_column("Seed", style="magenta")
    # table.add_column("Model", style="green")
    table.add_column("Dataset", style="yellow")
    table.add_column(
        "Model ID\n(library_model_refhash_roothash_trnhash_gen_N_cfghash_seed)", 
        style="blue",
        header_style="blue dim"
    )
    # table.add_column("Size", style="yellow")
    table.add_column("Created", style="dim")

    for model in models:
        name = model.get('experiment_name', 'unknown')
        seed = str(model.get('experiment_seed', '?'))
        # model_type = f"{model.get('library', '?')}/{model.get('model_type', '?')}"
        dataset_name = _extract_dataset_name(model)
        model_id = f"{name}_{model.get('config_hash', '?')}_{seed}"
        # size = f"{model.get('file_size_mb', 0):.1f} MB"

        # Format timestamp
        created = model.get('saved_at', 'unknown')
        if created != 'unknown' and 'T' in created:
            # Extract date from ISO timestamp
            created = created.split('T')[0]

        table.add_row(name, dataset_name, model_id, created)

    console.print(table)

def _show_detailed_model_list(models):
    """Show models with detailed information"""

    for i, model in enumerate(models):
        if i > 0:
            console.print()  # Space between models

        name = model.get('experiment_name', 'unknown')
        seed = model.get('experiment_seed', '?')

        model_id = f"{name}_{seed}"
        title = f"🤖 Model: {model_id}"

        details = []
        details.append(f"📁 Path: {model.get('file_path', 'unknown')}")
        details.append(f"📏 Size: {model.get('file_size_mb', 0):.2f} MB")

        if 'library' in model and 'model_type' in model:
            details.append(f"🔧 Type: {model['library']}/{model['model_type']}")

        dataset_name = _extract_dataset_name(model)
        details.append(f"📊 Dataset: {dataset_name}")  # NEW LINE

        if 'saved_at' in model:
            details.append(f"🕒 Created: {model['saved_at']}")

        if 'training_time' in model:
            details.append(f"⏱️  Training: {model['training_time']:.1f}s")

        console.print(Panel.fit(
            "\n".join(details),
            title=title,
            border_style="blue"
        ))

def _show_model_details(info, seed, experiment_name, export_file=None):
    """Show detailed information for a single model"""

    model_id = f"{experiment_name}_{seed}"
    title = f"🤖 Model Details: {model_id}"

    details = []
    details.append(f"📁 File: {info.get('file_path', 'unknown')}")
    details.append(f"📏 Size: {info.get('file_size_mb', 0):.2f} MB")

    if 'library' in info and 'model_type' in info:
        details.append(f"🔧 Type: {info['library']}/{info['model_type']}")

    if 'saved_at' in info:
        details.append(f"🕒 Created: {info['saved_at']}")

    if 'training_time' in info:
        details.append(f"⏱️  Training Time: {info['training_time']:.1f}s")

    if 'experiment_info' in info:
        exp_info = info['experiment_info']
        if 'description' in exp_info:
            details.append(f"📝 Description: {exp_info['description']}")
        if 'researcher' in exp_info:
            details.append(f"👤 Researcher: {exp_info['researcher']}")

    # Display lineage information if available
    if 'lineage' in info:
        lineage = info['lineage']
        details.append("")  # Blank line for separation
        details.append("🌳 Lineage Information:")
        details.append(f"  Generation: {lineage.get('generation', 'unknown')}")
        if lineage.get('parent_model_id'):
            parent_id = lineage['parent_model_id']
            # Truncate if too long
            if len(parent_id) > 50:
                parent_id = parent_id[:47] + "..."
            details.append(f"  Parent: {parent_id}")
        details.append(f"  Root Hash: {lineage.get('root_training_hash', 'unknown')}")
        details.append(f"  Reference Hash: {lineage.get('reference_hash', 'unknown')}")

    console.print(Panel.fit(
        "\n".join(details),
        title=title,
        border_style="blue"
    ))

    if 'params' in info:
        console.print("\n⚙️ Training Parameters (params.yaml):")

        # Handle export if requested
        if export_file is not None:            
            _export_params_to_file(info['params'], export_file, model_id)

        params = info['params']
        # Show complete params.yaml structure
        config_lines = []
        for section, section_data in params.items():
            if section == 'defaults':  # Skip Hydra defaults
                continue
                
            config_lines.append(f"{section}:")
            
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        config_lines.append(f"  {key}:")
                        for subkey, subvalue in value.items():
                            config_lines.append(f"    {subkey}: {subvalue}")
                    elif isinstance(value, list):
                        config_lines.append(f"  {key}: {value}")
                    else:
                        config_lines.append(f"  {key}: {value}")
            else:
                config_lines.append(f"  {section_data}")
            
            config_lines.append("")  # Empty line between sections
        
        console.print(Panel.fit(
            "\n".join(config_lines[:-1]),  # Remove last empty line
            title="Complete params.yaml Configuration",
            border_style="green"
        ))

def _export_params_to_file(params_data, filename, model_id):
    """Export params data to a YAML file"""

    try:
        if not params_data:
            console.print("⚠️  No params data to export", style="yellow")
            return
        
        # Convert to OmegaConf if it's a plain dict, then save with native method
        if isinstance(params_data, dict):
            config = OmegaConf.create(params_data)
        elif isinstance(params_data, DictConfig):
            config = params_data
        else:
            # Fallback: convert to OmegaConf
            config = OmegaConf.create(params_data)
        
        # Use OmegaConf's native YAML export
        OmegaConf.save(config, filename)
        console.print(f"📁 Exported params.yaml to: [bold green]{filename}[/bold green]")
        console.print(f"💡 Use this file to reproduce experiment {model_id}")

    except Exception as e:
        console.print(f"❌ Failed to export params: {e}", style="red")
