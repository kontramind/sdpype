# sdpype/core/models.py - Curated model discovery and information
"""
Curated list of tested synthetic data generation models with working hyperparameters
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Curated list of tested models with working configurations
CURATED_MODELS = {
    "sdv": {
        "gaussian_copula": {
            "type": "Copula",
            "description": "Fast Gaussian copula model, good for tabular data",
            "tested": True,
            "hyperparams": {
                "enforce_min_max_values": True,
                "enforce_rounding": True,
                "locales": ['en_US'],
                "default_distribution": 'beta',
                "numerical_distributions": {
                    "column_name_1": 'gaussian_kde',
                    "column_name_2": 'gaussian_kde',
                }
            }
        },
        "ctgan": {
            "type": "GAN", 
            "description": "CTGAN - high quality generative adversarial network",
            "tested": True,
            "hyperparams": {
                "enforce_min_max_values": True,
                "enforce_rounding": True,
                "locales": ['en_US'],
                "epochs": 300,
                "cuda": True,
                "verbose": False,
                "batch_size": 500,
                "discriminator_dim": [256, 256],
                "discriminator_decay": 1e-6,
                "discriminator_lr": 0.0002,
                "discriminator_steps": 1,
                "embedding_dim": 128,
                "generator_decay": 1e-6,
                "generator_dim": [256, 256],
                "generator_lr": 2e-4,
                "log_frequency": True,
                "pac": 10
            }
        },
        "tvae": {
            "type": "VAE",
            "description": "Tabular Variational Autoencoder", 
            "tested": True,
            "hyperparams": {
                "enforce_min_max_values": True,
                "enforce_rounding": True,
                "epochs": 300,
                "cuda": True,
                "verbose": False,
                "batch_size": 500,
                "compress_dims": [128, 128],
                "decompress_dims": [128, 128],
                "embedding_dim": 128,
                "l2scale": 1e-5,
                "loss_factor": 2
            }
        },
        "copula_gan": {
            "type": "Copula + GAN",
            "description": "A mix of classic (statistical) and GAN-based (deep learning) methods.", 
            "tested": True,
            "hyperparams": {
                "enforce_min_max_values": True,
                "enforce_rounding": True,
                "locales": ['en_US'],
                "default_distribution": 'beta',
                "numerical_distributions": {
                    "column_name_1": 'gaussian_kde',
                    "column_name_2": 'gaussian_kde',
                },
                "epochs": 300,
                "cuda": True,
                "verbose": False,
                "batch_size": 500,
                "discriminator_dim": [256, 256],
                "discriminator_decay": 1e-6,
                "discriminator_lr": 0.0002,
                "discriminator_steps": 1,
                "embedding_dim": 128,
                "generator_decay": 1e-6,
                "generator_dim": [256, 256],
                "generator_lr": 2e-4,
                "log_frequency": True,
                "pac": 10                
            }
        },
    },
    "synthcity": {
    "ctgan": {
        "type": "GAN",
        "description": "Synthcity's CTGAN implementation",
        "tested": True,
        "hyperparams": {
            # Training iterations - EXACT DEFAULTS FROM SOURCE CODE
            "n_iter": 2000,  # Note: your current "n_epochs" maps to this
            "batch_size": 200,  # Source default: 200

            # Generator architecture - EXACT DEFAULTS
            "generator_n_layers_hidden": 2,  # Source default: 2
            "generator_n_units_hidden": 500,  # Source default: 500
            "generator_nonlin": "relu",  # Source default: "relu"
            "generator_dropout": 0.1,  # Source default: 0.1
            "generator_opt_betas": [0.5, 0.999],  # Source default: (0.5, 0.999)

            # Discriminator architecture - EXACT DEFAULTS
            "discriminator_n_layers_hidden": 2,  # Source default: 2
            "discriminator_n_units_hidden": 500,  # Source default: 500
            "discriminator_nonlin": "leaky_relu",  # Source default: "leaky_relu"
            "discriminator_n_iter": 1,  # Source default: 1
            "discriminator_dropout": 0.1,  # Source default: 0.1
            "discriminator_opt_betas": [0.5, 0.999],  # Source default: (0.5, 0.999)

            # Learning rates and regularization - EXACT DEFAULTS
            "lr": 0.001,  # Source default: 1e-3 (same as 0.001)
            "weight_decay": 0.001,  # Source default: 1e-3 (same as 0.001)

            # Training stability - EXACT DEFAULTS
            "clipping_value": 1,  # Source default: 1 (int)
            "lambda_gradient_penalty": 10,  # Source default: 10 (float)

            # Data encoding - EXACT DEFAULTS
            "encoder_max_clusters": 10,  # Source default: 10
            "adjust_inference_sampling": False,  # Source default: False

            # Early stopping and monitoring - EXACT DEFAULTS
            "patience": 5,  # Source default: 5
            "n_iter_print": 50,  # Source default: 50
            "n_iter_min": 100,  # Source default: 100

            # Core plugin settings - EXACT DEFAULTS
            "compress_dataset": False,  # Source default: False
            "sampling_patience": 500,  # Source default: 500
            "random_state": 0,  # Source default: 0

            # Advanced parameters (usually left as None/default)
            # "encoder": None,  # Source default: None
            # "dataloader_sampler": None,  # Source default: None
            # "patience_metric": None,  # Source default: None
            # "workspace": "workspace",  # Source default: Path("workspace")
            # "device": "auto",  # Source default: DEVICE constant
            }
        },
        "ddpm": {
            "type": "Diffusion",
            "description": "Denoising Diffusion Probabilistic Model",
            "tested": True,
            "hyperparams": {
               # Core training parameters - EXACT DEFAULTS FROM SOURCE CODE
               "n_iter": 1000,  # Source default: 1000 (not "n_epochs")
               "lr": 0.002,  # Source default: 0.002 (not 0.001)
               "weight_decay": 1e-4,  # Source default: 1e-4
               "batch_size": 1024,  # Source default: 1024 (not 500)

               # Task configuration
               "is_classification": False,  # Source default: False

               # Diffusion process parameters
               "num_timesteps": 1000,  # Source default: 1000
               "gaussian_loss_type": "mse",  # Source default: "mse", options: "mse", "kl"
               "scheduler": "cosine",  # Source default: "cosine", options: "cosine", "linear"

               # Model architecture
             "model_type": "mlp",  # Source default: "mlp", options: "mlp" only (resnet/tabnet not implemented)
               "model_params": {  # Source default model_params for MLP
                   "n_layers_hidden": 3,  # From docstring default
                   "n_units_hidden": 256,  # From docstring default
                   "dropout": 0.0  # From docstring default
               },
               "dim_embed": 128,  # Source default: 128

               # Data encoding
               "continuous_encoder": "quantile",  # Source default: "quantile"
               "cont_encoder_params": {},  # Source default: {}

               # Training monitoring and validation
               "log_interval": 100,  # Source default: 100
               "validation_size": 0,  # Source default: 0 (no validation split)

               # Core plugin settings - EXACT DEFAULTS
               "random_state": 0,  # Source default: 0
               "compress_dataset": False,  # Source default: False
               "sampling_patience": 500,  # Source default: 500

               # Advanced parameters (usually left as default)
               # "callbacks": [],  # Source default: ()
               # "validation_metric": None,  # Source default: None
               # "workspace": "workspace",  # Source default: Path("workspace")
               # "device": "auto",  # Source default: DEVICE constant
            }
        },
        "bayesian_network": {
            "type": "Probabilistic",
            "description": "Bayesian Network using probabilistic graphical models (pgmpy backend)",
            "tested": True,
            "hyperparams": {
                # Structure learning parameters - EXACT DEFAULTS FROM SOURCE CODE
                "struct_learning_n_iter": 1000,  # Source default: 1000
                "struct_learning_search_method": "tree_search",  # Source default: "tree_search"
                "struct_learning_score": "k2",  # Source default: "k2"
                "struct_max_indegree": 4,  # Source default: 4

                # Data encoding parameters - EXACT DEFAULTS
                "encoder_max_clusters": 10,  # Source default: 10
                "encoder_noise_scale": 0.1,  # Source default: 0.1

                # Core plugin settings - EXACT DEFAULTS
                "random_state": 0,  # Source default: 0
                "compress_dataset": False,  # Source default: False
                "sampling_patience": 500,  # Source default: 500

                # Advanced parameters (usually left as default)
                # "workspace": "workspace",  # Source default: Path("workspace")
            }
        }
    }
}

def show_available_models(library_filter: str = None, show_params: str = None):
    """Show curated list of available models"""
    
    console.print("🤖 Available Synthetic Data Generation Models", style="bold cyan")
    
    # Check library availability
    _show_library_status()
    
    # Filter models if requested
    if library_filter:
        if library_filter not in CURATED_MODELS:
            console.print(f"❌ Unknown library: {library_filter}", style="red")
            console.print(f"Available libraries: {', '.join(CURATED_MODELS.keys())}", style="yellow")
            return
        models_to_show = {library_filter: CURATED_MODELS[library_filter]}
    else:
        models_to_show = CURATED_MODELS
    
    # Show models table
    _show_models_table(models_to_show)
    
    # Show specific model parameters if requested
    if show_params:
        _show_specific_model_params(show_params, models_to_show)
    else:
        # Show usage examples
        _show_usage_examples()

def _show_library_status():
    """Check and show which libraries are available"""
    
    console.print("\n📚 Library Status:")
    
    # Test SDV availability
    try:
        import sdv
        sdv_status = f"✅ Available (v{sdv.__version__})"
        sdv_style = "green"
    except ImportError:
        sdv_status = "❌ Not installed"
        sdv_style = "red"
    
    # Test Synthcity availability  
    try:
        import synthcity
        # Try to get version from the version module first
        try:
            from synthcity.version import __version__ as synthcity_version
        except ImportError:
            # Fallback to direct attribute
            synthcity_version = getattr(synthcity, '__version__', 'unknown')
        synthcity_status = f"✅ Available (v{synthcity_version})"
        synthcity_style = "green"
    except ImportError:
        synthcity_status = "❌ Not installed"
        synthcity_style = "red"
    
    table = Table(show_header=True)
    table.add_column("Library", style="cyan")
    table.add_column("Status", style="white")
    
    table.add_row("SDV", f"[{sdv_style}]{sdv_status}[/{sdv_style}]")
    table.add_row("Synthcity", f"[{synthcity_style}]{synthcity_status}[/{synthcity_style}]")
    
    console.print(table)

def _show_models_table(models_to_show: dict):
    """Display models in a formatted table"""
    
    console.print("\n🎯 Curated Models:")
    
    table = Table(show_header=True)
    table.add_column("Library", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Description", style="white")
    
    for library_name, models in models_to_show.items():
        for model_name, model_info in models.items():
            # Determine status
            if model_info["tested"]:
                status = "✅ Tested"
                status_style = "green"
            else:
                status = "🔶 Untested"
                status_style = "yellow"
            
            table.add_row(
                library_name,
                model_name,
                model_info["type"],
                f"[{status_style}]{status}[/{status_style}]",
                model_info["description"]
            )
    
    console.print(table)

def _show_specific_model_params(model_request: str, models_to_show: dict):
    """Show parameters for a specific model"""
    
    console.print(f"\n⚙️ Parameters for: {model_request}")
    
    # Parse model request: "ctgan" or "sdv/ctgan"
    if "/" in model_request:
        # Specific library/model format
        library, model = model_request.split("/", 1)
        if library in models_to_show and model in models_to_show[library]:
            _display_model_params(library, model, models_to_show[library][model])
        else:
            console.print(f"❌ Model not found: {model_request}", style="red")
            _suggest_available_models(models_to_show)
    else:
        # Model name only - search across all libraries
        found_models = []
        for library, models in models_to_show.items():
            if model_request in models:
                found_models.append((library, model_request, models[model_request]))
        
        if found_models:
            for library, model, model_info in found_models:
                _display_model_params(library, model, model_info)
        else:
            console.print(f"❌ Model not found: {model_request}", style="red")
            _suggest_available_models(models_to_show)

def _display_model_params(library: str, model: str, model_info: dict):
    """Display parameters for a single model"""
    
    title = f"📋 {library}/{model}"
    
    if not model_info["tested"]:
        console.print(f"⚠️ Note: {library}/{model} is untested", style="yellow")
    
    # Generate YAML dynamically from hyperparams
    yaml_content = _generate_yaml_config(library, model, model_info["hyperparams"])
    
    console.print(Panel.fit(
        yaml_content,
        title=title,
        border_style="blue"
    ))

def _generate_yaml_config(library: str, model: str, hyperparams: dict) -> str:
    """Generate params.yaml configuration from hyperparams dictionary"""
    
    # Determine model_type based on library and model
    if library == "sdv":
        model_type = model
    else:
        model_type = f"{library}_{model}"
    
    # Start with base config
    lines = [
        "sdg:",
        f"  model_type: {model_type}",
        f"  library: {library}"
    ]
    
    # Add parameters if any exist
    if hyperparams:
        lines.append("  parameters:")
        for key, value in hyperparams.items():
            lines.extend(_format_yaml_value(key, value, indent=4))
    
    return "\n".join(lines)

def _format_yaml_value(key: str, value, indent: int = 0) -> list:
    """Format a single value for YAML output with proper indentation"""
    
    indent_str = " " * indent
    
    if isinstance(value, str):
        return [f"{indent_str}{key}: {value}"]
    
    elif isinstance(value, (int, float, bool)):
        return [f"{indent_str}{key}: {value}"]
    
    elif isinstance(value, list):
        if not value:
            return [f"{indent_str}{key}: []"]
        elif len(value) == 1:
            # Single item list - use inline format
            # Format the item properly (add quotes if string)
            item = f"'{value[0]}'" if isinstance(value[0], str) else value[0]
            return [f"{indent_str}{key}: [{item}]"]
        else:
            # Multi-item list - use block format
            lines = [f"{indent_str}{key}:"]
            for item in value:
                lines.append(f"{indent_str}  - {item}")
            return lines
    
    elif isinstance(value, dict):
        if not value:
            return [f"{indent_str}{key}: {{}}"]
        else:
            # Dictionary - use block format
            lines = [f"{indent_str}{key}:"]
            for dict_key, dict_value in value.items():
                # Format both key and value properly
                formatted_key = f"'{dict_key}'" if isinstance(dict_key, str) and ' ' in dict_key else dict_key
                formatted_value = f"'{dict_value}'" if isinstance(dict_value, str) else dict_value
                lines.append(f"{indent_str}  {formatted_key}: {formatted_value}")
            return lines
    
    else:
        # Fallback for other types
        return [f"{indent_str}{key}: {value}"]

def _suggest_available_models(models_to_show: dict):
    """Show available models when a requested model is not found"""
    
    console.print("\nAvailable models:", style="yellow")
    for library, models in models_to_show.items():
        for model in models.keys():
            console.print(f"  • {model} (or {library}/{model})")

def _show_usage_examples():
    """Show basic usage examples"""
    
    console.print("\n💡 Usage Examples:")
    
    examples = [
        "sdpype models --library sdv           # Show only SDV models",
        "sdpype models --library synthcity     # Show only Synthcity models", 
        "sdpype models --params ctgan          # Show params for any 'ctgan' model",
        "sdpype models --params sdv/ctgan      # Show params for SDV's ctgan specifically",
        "",
        "# Copy the params.yaml example, then run:",
        "sdpype pipeline                       # Train with selected model"
    ]
    
    for example in examples:
        if example:
            console.print(f"  {example}")
        else:
            console.print()

def get_model_config_template(library: str, model: str) -> str:
    """Get a template configuration for a specific model"""
    
    if library not in CURATED_MODELS:
        return None
    
    if model not in CURATED_MODELS[library]:
        return None
    
    # Generate YAML dynamically
    hyperparams = CURATED_MODELS[library][model]["hyperparams"]
    return _generate_yaml_config(library, model, hyperparams)
