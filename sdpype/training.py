# Enhanced sdpype/training.py - Using new serialization module
"""
Enhanced SDG training module using centralized serialization
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sdv.metadata import SingleTableMetadata

# SDV models
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer

# Synthcity models and serialization
from synthcity.plugins import Plugins

# Import new serialization module
from sdpype.serialization import save_model, create_model_metadata

# Synthpop imports
from synthpop.method import CARTMethod


def _get_config_hash() -> str:
    """Get config hash from temporary file created during pipeline execution"""
    try:
        if Path('.sdpype_config_hash').exists():
            with open('.sdpype_config_hash', 'r') as f:
                return f.read().strip()
        return "nohash"
    except Exception:
        return "nohash"


def create_sdv_model(cfg: DictConfig, metadata: SingleTableMetadata):
    """Create SDV model with metadata"""
    # SDV v1+ requires metadata
    # Get model parameters from config, filtering for SDV-compatible params  
    # Use OmegaConf.to_container to properly convert nested structures
    all_params = OmegaConf.to_container(cfg.sdg.parameters, resolve=True) if cfg.sdg.parameters else {}
    
    # Filter out Synthcity-specific parameters that SDV doesn't understand
    synthcity_only_params = {
        'n_iter', 'generator_n_layers_hidden', 'generator_n_units_hidden',
        'generator_nonlin', 'generator_opt_betas', 'discriminator_n_layers_hidden',
        'discriminator_n_units_hidden', 'discriminator_nonlin', 'discriminator_n_iter',
        'discriminator_opt_betas', 'clipping_value', 'lambda_gradient_penalty',
        'encoder_max_clusters', 'adjust_inference_sampling', 'patience',
        'n_iter_print', 'n_iter_min', 'compress_dataset', 'sampling_patience',
        # DDPM-specific parameters
        'is_classification', 'num_timesteps', 'gaussian_loss_type', 'scheduler',
        'model_type', 'model_params', 'dim_embed', 'continuous_encoder',
        'cont_encoder_params', 'log_interval', 'validation_size', 'callbacks',
        'validation_metric',
        # Bayesian Network-specific parameters
        'struct_learning_n_iter', 'struct_learning_search_method', 'struct_learning_score',
        'struct_max_indegree', 'encoder_noise_scale',
        # RTVAE-specific parameters
        'n_units_embedding', 'robust_divergence_beta', 'data_encoder_max_clusters',
        'decoder_n_layers_hidden', 'decoder_n_units_hidden', 'decoder_nonlin', 'decoder_dropout',
        'encoder_n_layers_hidden', 'encoder_n_units_hidden', 'encoder_nonlin', 'encoder_dropout',
    }
    model_params = {k: v for k, v in all_params.items() if k not in synthcity_only_params}

    match cfg.sdg.model_type:
        case "gaussiancopula":
            return GaussianCopulaSynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                locales=model_params.get("locales", ['en_US']),
                default_distribution=model_params.get("default_distribution", 'beta'),
                numerical_distributions=model_params.get("numerical_distributions", {})
            )
        case "ctgan":
            return CTGANSynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                locales=model_params.get("locales", ['en_US']),
                epochs=model_params.get("epochs", 300),
                batch_size=model_params.get("batch_size", 500),
                verbose=model_params.get("verbose", False),
                cuda=model_params.get("cuda", True),
                discriminator_dim=model_params.get("discriminator_dim", (256, 256)),
                discriminator_decay=model_params.get("discriminator_decay", 1e-06),
                discriminator_lr=model_params.get("discriminator_lr", 0.0002),
                discriminator_steps=model_params.get("discriminator_steps", 1),
                generator_lr=model_params.get("generator_lr", 0.0002),
                embedding_dim=model_params.get("embedding_dim", 128),
                generator_decay=model_params.get("generator_decay", 1e-06),
                generator_dim=model_params.get("generator_dim", (256, 256)),
                log_frequency=model_params.get("log_frequency", True),
                pac=model_params.get("pac", 10)
            )

        case "tvae":
            return TVAESynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                epochs=model_params.get("epochs", 300),
                batch_size=model_params.get("batch_size", 500),
                verbose=model_params.get("verbose", False),
                cuda=model_params.get("cuda", True),
                compress_dims=model_params.get("compress_dims", (128, 128)),
                decompress_dims=model_params.get("decompress_dims", (128, 128)),
                embedding_dim=model_params.get("embedding_dim", 128),
                l2scale=model_params.get("l2scale", 1e-5),
                loss_factor=model_params.get("loss_factor", 2)
            )

        case "copulagan":
            return CopulaGANSynthesizer(
                metadata,
                enforce_min_max_values=model_params.get("enforce_min_max_values", True),
                enforce_rounding=model_params.get("enforce_rounding", True),
                locales=model_params.get("locales", ['en_US']),
                default_distribution=model_params.get("default_distribution", 'beta'),
                numerical_distributions=model_params.get("numerical_distributions", {}),
                epochs=model_params.get("epochs", 300),
                batch_size=model_params.get("batch_size", 500),
                verbose=model_params.get("verbose", False),
                cuda=model_params.get("cuda", True),
                discriminator_dim=model_params.get("discriminator_dim", (256, 256)),
                discriminator_decay=model_params.get("discriminator_decay", 1e-06),
                discriminator_lr=model_params.get("discriminator_lr", 0.0002),
                discriminator_steps=model_params.get("discriminator_steps", 1),
                generator_lr=model_params.get("generator_lr", 0.0002),
                embedding_dim=model_params.get("embedding_dim", 128),
                generator_decay=model_params.get("generator_decay", 1e-06),
                generator_dim=model_params.get("generator_dim", (256, 256)),
                log_frequency=model_params.get("log_frequency", True),
                pac=model_params.get("pac", 10)
            )

        case _:
            raise ValueError(f"Unknown SDV model: {cfg.sdg.model_type}")


def create_synthcity_model(cfg: DictConfig, data_shape):
    """Create Synthcity model with detailed parameter handling"""
    model_name = cfg.sdg.model_type

    # Get model parameters from config, filtering for Synthcity-compatible params
    all_params = OmegaConf.to_container(cfg.sdg.parameters, resolve=True) if cfg.sdg.parameters else {}

    # Filter out SDV-specific parameters that Synthcity doesn't understand
    sdv_only_params = {'epochs', 'verbose', 'cuda', 'enforce_min_max_values', 'enforce_rounding', 'locales'}
    model_params = {k: v for k, v in all_params.items() if k not in sdv_only_params}

    # Create synthcity plugin with detailed parameter mapping
    try:
        match model_name:
            case "ctgan":
                # Map common parameters that might have different names
                # Handle epochs -> n_iter mapping
                if 'epochs' in all_params and 'n_iter' not in model_params:
                    model_params['n_iter'] = all_params['epochs']

                model = Plugins().get("ctgan",
                    # Training configuration
                    n_iter=model_params.get("n_iter", 2000),
                    batch_size=model_params.get("batch_size", 200),
                    random_state=model_params.get("random_state", cfg.experiment.seed),

                    # Generator architecture
                    generator_n_layers_hidden=model_params.get("generator_n_layers_hidden", 2),
                    generator_n_units_hidden=model_params.get("generator_n_units_hidden", 500),
                    generator_nonlin=model_params.get("generator_nonlin", "relu"),
                    generator_dropout=model_params.get("generator_dropout", 0.1),
                    generator_opt_betas=tuple(model_params.get("generator_opt_betas", [0.5, 0.999])),

                    # Discriminator architecture
                    discriminator_n_layers_hidden=model_params.get("discriminator_n_layers_hidden", 2),
                    discriminator_n_units_hidden=model_params.get("discriminator_n_units_hidden", 500),
                    discriminator_nonlin=model_params.get("discriminator_nonlin", "leaky_relu"),
                    discriminator_n_iter=model_params.get("discriminator_n_iter", 1),
                    discriminator_dropout=model_params.get("discriminator_dropout", 0.1),
                    discriminator_opt_betas=tuple(model_params.get("discriminator_opt_betas", [0.5, 0.999])),

                    # Learning rates and regularization
                    lr=model_params.get("lr", 1e-3),
                    weight_decay=model_params.get("weight_decay", 1e-3),

                    # Training stability
                    clipping_value=model_params.get("clipping_value", 1),
                    lambda_gradient_penalty=model_params.get("lambda_gradient_penalty", 10),

                    # Data encoding and handling
                    encoder_max_clusters=model_params.get("encoder_max_clusters", 10),
                    adjust_inference_sampling=model_params.get("adjust_inference_sampling", False),

                    # Early stopping and monitoring
                    patience=model_params.get("patience", 5),
                    n_iter_print=model_params.get("n_iter_print", 50),
                    n_iter_min=model_params.get("n_iter_min", 100),

                    # Core plugin settings
                    compress_dataset=model_params.get("compress_dataset", False),
                    sampling_patience=model_params.get("sampling_patience", 500),

                    # Advanced parameters (only if explicitly provided)
                    **{k: v for k, v in model_params.items() if k in [
                        'encoder', 'dataloader_sampler', 'patience_metric', 'workspace'
                    ] and v is not None}
                )
                return model

            case "marginaldistributions":
                # Simple baseline model with no hyperparameters
                # Samples from marginal distributions independently
                model = Plugins().get("marginal_distributions",
                    # Core plugin settings
                    random_state=model_params.get("random_state", cfg.experiment.seed),
                    sampling_patience=model_params.get("sampling_patience", 500),
                )
                return model

            case "ddpm":
                # Map common parameters that might have different names
                # Handle epochs -> n_iter mapping
                if 'epochs' in all_params and 'n_iter' not in model_params:
                    model_params['n_iter'] = all_params['epochs']

                model = Plugins().get("ddpm",
                    # Core training parameters
                    n_iter=model_params.get("n_iter", 1000),
                    lr=model_params.get("lr", 0.002),
                    weight_decay=model_params.get("weight_decay", 1e-4),
                    batch_size=model_params.get("batch_size", 1024),
                    random_state=model_params.get("random_state", cfg.experiment.seed),

                    # Task configuration
                    is_classification=model_params.get("is_classification", False),

                    # Diffusion process parameters
                    num_timesteps=model_params.get("num_timesteps", 1000),
                    gaussian_loss_type=model_params.get("gaussian_loss_type", "mse"),
                    scheduler=model_params.get("scheduler", "cosine"),

                    # Model architecture
                    model_type=model_params.get("model_type", "mlp"),
                    model_params=model_params.get("model_params", {
                        "n_layers_hidden": 3,
                        "n_units_hidden": 256,
                        "dropout": 0.0
                    }),
                    dim_embed=model_params.get("dim_embed", 128),

                    # Data encoding
                    continuous_encoder=model_params.get("continuous_encoder", "quantile"),
                    cont_encoder_params=model_params.get("cont_encoder_params", {}),

                    # Training monitoring and validation
                    log_interval=model_params.get("log_interval", 100),
                    validation_size=model_params.get("validation_size", 0),

                    # Core plugin settings
                    compress_dataset=model_params.get("compress_dataset", False),
                    sampling_patience=model_params.get("sampling_patience", 500),

                    # Advanced parameters (only if explicitly provided)
                    **{k: v for k, v in model_params.items() if k in [
                        'callbacks', 'validation_metric', 'workspace'
                    ] and v is not None}
                )
                return model

            case "rtvae":
                # Map common parameters that might have different names
                # Handle epochs -> n_iter mapping
                if 'epochs' in all_params and 'n_iter' not in model_params:
                    model_params['n_iter'] = all_params['epochs']

                model = Plugins().get("rtvae",
                    # Training configuration
                    n_iter=model_params.get("n_iter", 1000),
                    batch_size=model_params.get("batch_size", 200),
                    random_state=model_params.get("random_state", cfg.experiment.seed),
                    n_units_embedding=model_params.get("n_units_embedding", 500),

                    # Decoder architecture
                    decoder_n_layers_hidden=model_params.get("decoder_n_layers_hidden", 3),
                    decoder_n_units_hidden=model_params.get("decoder_n_units_hidden", 500),
                    decoder_nonlin=model_params.get("decoder_nonlin", "leaky_relu"),
                    decoder_dropout=model_params.get("decoder_dropout", 0),

                    # Encoder architecture
                    encoder_n_layers_hidden=model_params.get("encoder_n_layers_hidden", 3),
                    encoder_n_units_hidden=model_params.get("encoder_n_units_hidden", 500),
                    encoder_nonlin=model_params.get("encoder_nonlin", "leaky_relu"),
                    encoder_dropout=model_params.get("encoder_dropout", 0.1),

                    # Learning rates and regularization
                    lr=model_params.get("lr", 1e-3),
                    weight_decay=model_params.get("weight_decay", 1e-5),

                    # Robust divergence parameter (key feature of RTVAE)
                    robust_divergence_beta=model_params.get("robust_divergence_beta", 2),

                    # Data encoding
                    data_encoder_max_clusters=model_params.get("data_encoder_max_clusters", 10),

                    # Early stopping and monitoring
                    n_iter_print=model_params.get("n_iter_print", 50),
                    n_iter_min=model_params.get("n_iter_min", 100),
                    patience=model_params.get("patience", 5),

                    # Core plugin settings (workspace can be provided if needed)
                )
                return model

            case "bayesiannetwork":
                # Map common parameters that might have different names
                # Handle epochs -> struct_learning_n_iter mapping (if needed)
                if 'epochs' in all_params and 'struct_learning_n_iter' not in model_params:
                    model_params['struct_learning_n_iter'] = all_params['epochs']

                model = Plugins().get("bayesiannetwork",
                    # Structure learning parameters
                    struct_learning_n_iter=model_params.get("struct_learning_n_iter", 1000),
                    struct_learning_search_method=model_params.get("struct_learning_search_method", "tree_search"),
                    struct_learning_score=model_params.get("struct_learning_score", "k2"),
                    struct_max_indegree=model_params.get("struct_max_indegree", 4),

                    # Data encoding parameters
                    encoder_max_clusters=model_params.get("encoder_max_clusters", 10),
                    encoder_noise_scale=model_params.get("encoder_noise_scale", 0.1),

                    # Core plugin settings
                    random_state=model_params.get("random_state", cfg.experiment.seed),
                    compress_dataset=model_params.get("compress_dataset", False),
                    sampling_patience=model_params.get("sampling_patience", 500),

                    # Advanced parameters (only if explicitly provided)
                    **{k: v for k, v in model_params.items() if k in [
                        'workspace'
                    ] and v is not None}
                )
                return model

            case _:
                # Fallback to generic plugin creation for other Synthcity models
                model = Plugins().get(model_name, **model_params)
                return model

    except Exception as e:
        raise ValueError(f"Failed to create Synthcity model '{model_name}': {e}")

def create_synthpop_model(cfg: DictConfig, metadata):
    """Create Synthpop model with metadata conversion"""
    model_type = cfg.sdg.model_type

    # Convert SDPype/SDV metadata to synthpop format
    synthpop_metadata = {}
    for column_name, column_info in metadata.columns.items():
        sdtype = column_info.get('sdtype', 'unknown')
        if sdtype == "numerical":
            synthpop_metadata[column_name] = "numerical"
        elif sdtype == "categorical":
            synthpop_metadata[column_name] = "categorical"
        elif sdtype == "boolean":
            synthpop_metadata[column_name] = "boolean"
        elif sdtype == "datetime":
            synthpop_metadata[column_name] = "datetime"
        else:
            # Default fallback
            synthpop_metadata[column_name] = "numerical"
            print(f"⚠️  Unknown sdtype '{sdtype}' for column '{column_name}', defaulting to numerical")

    # Get parameters from config
    model_params = OmegaConf.to_container(cfg.sdg.parameters, resolve=True) if cfg.sdg.parameters else {}

    if model_type == "cart":
        return CARTMethod(
            metadata=synthpop_metadata,
            smoothing=model_params.get("smoothing", False),
            proper=model_params.get("proper", False),
            minibucket=model_params.get("minibucket", 5),
            random_state=cfg.experiment.seed,
            tree_params=model_params.get("tree_params", {})
        ), synthpop_metadata
    else:
        raise ValueError(f"Unknown synthpop model: {model_type}")

def create_experiment_hash(cfg: DictConfig) -> str:
    """Create unique hash for experiment configuration"""
    # Include key config elements in hash
    hash_dict = {
        "sdg": OmegaConf.to_container(cfg.sdg, resolve=True),
        "seed": cfg.experiment.seed,
        "data_file": cfg.data.training_file
    }

    hash_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:8]


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Train synthetic data generator with unified .pkl output"""
    
    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.seed)
    
    # Create experiment metadata
    experiment_hash = create_experiment_hash(cfg)
    experiment_id = f"{cfg.sdg.model_type}_{cfg.experiment.seed}_{experiment_hash}"

    print(f"🚀 Training {cfg.sdg.model_type} model (library: {cfg.sdg.library})")
    print(f"📋 Experiment: {experiment_id}")
    print(f"🎲 Seed: {cfg.experiment.seed}")

    # Get config hash for file paths
    config_hash = _get_config_hash()

    # Load training data directly from config path
    data_file = cfg.data.training_file
    if not Path(data_file).exists():
        print(f"❌ Training data not found: {data_file}")
        print("💡 Check your data.training_file path in params.yaml!")
        raise FileNotFoundError(f"Training data file not found: {data_file}")

    metadata_file = cfg.data.metadata_file
    if not Path(metadata_file).exists():
        print(f"❌ Metadata for processed data not found: {metadata_file}")
        print("💡 Check your data.metadata_file path in params.yaml!")
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    training_data = pd.read_csv(data_file)
    print(f"📊 Training data: {training_data.shape}")

    metadata = SingleTableMetadata.load_from_json(metadata_file)

    # Create model based on library
    library = cfg.sdg.library

    if library == "sdv":
        print(f"🔧 Creating SDV {cfg.sdg.model_type} model...")
        model = create_sdv_model(cfg, metadata)
    elif library == "synthcity":
        print(f"🔧 Creating Synthcity {cfg.sdg.model_type} model...")
        model = create_synthcity_model(cfg, training_data.shape)
    elif library == "synthpop":
        print(f"🔧 Creating Synthpop {cfg.sdg.model_type} model...")
        model, synthpop_metadata = create_synthpop_model(cfg, metadata)
    else:
        raise ValueError(f"Unknown library: {library}")

    # Train model
    print(f"⏳ Training {library} {cfg.sdg.model_type} model...")
    start_time = time.time()

    if library == "synthpop":
        # Use SDPype preprocessed data directly
        print(f"🔄 Using SDPype preprocessing for synthpop...")
        model._sdpype_metadata = synthpop_metadata
        model.fit(training_data)
    elif library == "sdv":
        model.fit(training_data)
        model._set_random_state(cfg.experiment.seed)
    else:
        model.fit(training_data)

    training_time = time.time() - start_time

    print(f"⏱️  Training completed in {training_time:.1f}s")

    # Create standardized metadata using serialization module
    metadata = create_model_metadata(
        cfg, cfg.sdg.model_type, library, cfg.experiment.seed,
        training_time, training_data, experiment_id, experiment_hash
    )

    # Save model using new serialization module
    model_filename = save_model(
        model, metadata, library, cfg.experiment.seed, cfg.experiment.name
    )

    # Save training metrics (same as before)
    metrics = {
        "experiment_id": experiment_id,
        "experiment_hash": experiment_hash,
        "seed": cfg.experiment.seed,
        "config_hash": config_hash,
        "library": library,
        "model_type": cfg.sdg.model_type,
        "training_time": training_time,
        "training_rows": len(training_data),
        "training_columns": len(training_data.columns),
        "timestamp": datetime.now().isoformat(),
        "data_source": data_file,
        "model_output": model_filename,
        "model_parameters": OmegaConf.to_container(cfg.sdg.parameters, resolve=True) if cfg.sdg.parameters else {}
    }

    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    metrics_filename = f"experiments/metrics/training_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📊 Metrics saved: {metrics_filename}")
    print("✅ Model training completed")

    # Print training summary
    print(f"\n📈 Training Summary:")
    print(f"  Library: {library}")
    print(f"  Model: {cfg.sdg.model_type}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Training data: {len(training_data):,} rows × {len(training_data.columns)} columns")
    print(f"  Model file: {model_filename}")


if __name__ == "__main__":
    main()
