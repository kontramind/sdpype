"""
Enhanced preprocessing module for monolithic SDPype with experiment versioning
"""

import json
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def _get_config_hash() -> str:
    """Get config hash from temporary file created during pipeline execution"""
    try:
        if Path('.sdpype_config_hash').exists():
            with open('.sdpype_config_hash', 'r') as f:
                return f.read().strip()
        return "nohash"
    except Exception:
        return "nohash"


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Preprocess training and reference data with experiment versioning"""

    training_file_path = Path(cfg.data.training_file)
    reference_file_path = Path(cfg.data.reference_file)

    if not training_file_path.exists():
        raise FileNotFoundError(f"❌ Required training data file not found: {cfg.data.training_file}")

    if not reference_file_path.exists():
        raise FileNotFoundError(f"❌ Required reference data file not found: {cfg.data.reference_file}")

    metadata_file_path = Path(cfg.data.metadata_file)
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {cfg.data.metadata_file}")
    
    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.seed)
    
    config_hash = _get_config_hash()
    print(f"🔄 Starting preprocessing for {cfg.experiment.name} (seed: {cfg.experiment.seed})")
    print(f"🎲 Experiment seed: {cfg.experiment.seed}")

    # Load data (updated path for monolithic structure)
    # Load both datasets
    training_data = pd.read_csv(training_file_path)
    reference_data = pd.read_csv(reference_file_path)
    print(f"Loaded training data: {training_data.shape}")
    print(f"Loaded reference data: {reference_data.shape}")

    # Load metadata (shared between training and reference)
    metadata = SingleTableMetadata.load_from_json(metadata_file_path)
    print(f"Loaded metadata with {len(metadata.columns)} columns")

    if cfg.preprocessing.enabled:

        # Encode categorical features - some generators/metrics work only with numerical values
        if "encode_categorical" in cfg.preprocessing.steps:
            categorical_columns = list(filter(
                lambda col: metadata.columns[col].get("sdtype") == "categorical", 
                metadata.columns.keys()
            ))
            encoding_method = cfg.preprocessing.steps.encode_categorical
            print(f"Using {encoding_method} encoding for categorical columns")

            match encoding_method:
                case "label":
                    # Label encoding
                    label_encoders = {}
                    for col in categorical_columns:
                        le = LabelEncoder()
                        # Fit on training data, apply to both datasets
                        le.fit(training_data[col])
                        training_data[col] = le.transform(training_data[col])
                        reference_data[col] = le.transform(reference_data[col])
                        label_encoders[col] = le
                        print(f"  Label encoded {col}: {training_data[col].nunique()} unique values")
                        # Update metadata: keep semantic type as categorical, but change representation
                        metadata.update_column(col, sdtype='categorical')

                case "onehot":
                    # One-hot encoding
                    if categorical_columns:
                        # Create one-hot encoded columns and update metadata
                        training_encoded_dfs = []
                        reference_encoded_dfs = []
                        new_columns_metadata = {}
                        for col in categorical_columns:
                            # Fit on training data
                            training_one_hot = pd.get_dummies(training_data[col], prefix=col, dtype=int)
                            # Apply same columns to reference data
                            reference_one_hot = pd.get_dummies(reference_data[col], prefix=col, dtype=int)
                            # Align columns (in case reference has different categories)
                            reference_one_hot = reference_one_hot.reindex(columns=training_one_hot.columns, fill_value=0)
                            
                            training_encoded_dfs.append(training_one_hot)
                            reference_encoded_dfs.append(reference_one_hot)
                            print(f"  One-hot encoded {col}: {training_one_hot.shape[1]} columns created")

                            # Track new binary columns for metadata (semantic type: boolean)
                            for new_col in training_one_hot.columns:
                                new_columns_metadata[new_col] = {'sdtype': 'boolean'}

                        # Drop original categorical columns and add one-hot encoded ones
                        training_data = training_data.drop(columns=categorical_columns)
                        reference_data = reference_data.drop(columns=categorical_columns)
                        training_data = pd.concat([training_data] + training_encoded_dfs, axis=1)
                        reference_data = pd.concat([reference_data] + reference_encoded_dfs, axis=1)

                        # Update metadata: remove original categorical columns, add new binary columns
                        for col in categorical_columns:
                            metadata.remove_column(col)
                        for new_col, col_metadata in new_columns_metadata.items():
                            metadata.add_column(new_col, **col_metadata)

                case "frequency":
                    # Frequency encoding
                    for col in categorical_columns:
                        # Calculate frequency map from training data
                        freq_map = training_data[col].value_counts().to_dict()
                        # Apply to both datasets
                        training_data[col] = training_data[col].map(freq_map)
                        reference_data[col] = reference_data[col].map(freq_map).fillna(0)  # Fill unknown categories with 0
                        print(f"  Frequency encoded {col}: min_freq={training_data[col].min()}, max_freq={training_data[col].max()}")
                        # Update metadata: categorical -> numerical (frequency counts are truly numerical)
                        metadata.update_column(col, sdtype='numerical', computer_representation='Int64')

                case _:
                    raise ValueError(f"Unknown encoding method: {encoding_method}. Use 'label', 'onehot', or 'frequency'")

        # Handle missing values
        if "handle_missing" in cfg.preprocessing.steps:
            method = cfg.preprocessing.steps.handle_missing
            if method == "mean":
                # Use training data means for both datasets
                numeric_cols = training_data.select_dtypes(include=['number']).columns
                means = training_data[numeric_cols].mean()
                training_data[numeric_cols] = training_data[numeric_cols].fillna(means)
                reference_data[numeric_cols] = reference_data[numeric_cols].fillna(means)
        
        # Remove outliers  
        if cfg.preprocessing.steps.get("remove_outliers", False):
            # Apply outlier removal to both datasets using training data thresholds
            numeric_cols = training_data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                # Calculate thresholds from training data
                Q1 = training_data[col].quantile(0.25)
                Q3 = training_data[col].quantile(0.75)
                IQR = Q3 - Q1
                # Apply to both datasets
                training_mask = ~((training_data[col] < (Q1 - 1.5 * IQR)) | (training_data[col] > (Q3 + 1.5 * IQR)))
                reference_mask = ~((reference_data[col] < (Q1 - 1.5 * IQR)) | (reference_data[col] > (Q3 + 1.5 * IQR)))
                training_data = training_data[training_mask]
                reference_data = reference_data[reference_mask]

        # Scale numeric features
        if cfg.preprocessing.steps.get("scale_numeric") == "standard":
            print("Applying standard scaling to numerical columns in both datasets...")            
            # Get only truly numerical columns from metadata, exclude categorical/boolean/datetime
            numerical_columns = []
            excluded_columns = []

            # Use training data to determine scaling parameters
            # Apply same scaling to both datasets for consistency
            for col in training_data.columns:
                if col in metadata.columns:
                    sdtype = metadata.columns[col].get("sdtype", "unknown")
                    if sdtype == "numerical":
                        numerical_columns.append(col)
                    else:
                        excluded_columns.append((col, sdtype))
                else:
                    # Column not in metadata, assume numerical for safety
                    numerical_columns.append(col)

            print(f"Scaling {len(numerical_columns)} numerical columns using standard scaler")
            if numerical_columns:
                print(f"  Scaling columns: {numerical_columns}")
                # Fit scaler on training data
                scaler = StandardScaler()
                scaler.fit(training_data[numerical_columns])

                # Apply to both datasets
                training_data[numerical_columns] = scaler.transform(training_data[numerical_columns])
                reference_data[numerical_columns] = scaler.transform(reference_data[numerical_columns])

            if excluded_columns:
                print(f"  Preserving non-numerical columns: {[(col, dtype) for col, dtype in excluded_columns]}")

        print(f"Processed training data: {training_data.shape}")
        print(f"Processed reference data: {reference_data.shape}")
    else:
        print("Preprocessing disabled - using raw data")
    
    # Save processed datasets separately
    Path("experiments/data/processed").mkdir(parents=True, exist_ok=True)
    training_output_file = f"experiments/data/processed/training_data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.csv"
    reference_output_file = f"experiments/data/processed/reference_data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.csv"
    training_data.to_csv(training_output_file, index=False)
    reference_data.to_csv(reference_output_file, index=False)
    print(f"📁 Saved training data: {training_output_file}")
    print(f"📁 Saved reference data: {reference_output_file}")

    # Save metadata (shared between both datasets)
    metadata_output_file = f"experiments/data/processed/data_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}_metadata.json"
    if Path(metadata_output_file).exists():
        Path(metadata_output_file).unlink()

    metadata.save_to_json(metadata_output_file)
    print(f"🔍 Updated metadata saved: {metadata_output_file}")

    # Save metrics with information about both datasets
    metrics = {
        "experiment_seed": cfg.experiment.seed,
        "experiment_name": cfg.experiment.name,
        "timestamp": datetime.now().isoformat(),
        "training_original_rows": len(pd.read_csv(cfg.data.training_file)),
        "reference_original_rows": len(pd.read_csv(cfg.data.reference_file)),
        "training_processed_rows": len(training_data),
        "reference_processed_rows": len(reference_data),
        "preprocessing_enabled": cfg.preprocessing.enabled,
        "training_output_file": training_output_file,
        "reference_output_file": reference_output_file
    }
    
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    metrics_file = f"experiments/metrics/preprocess_{cfg.experiment.name}_{config_hash}_{cfg.experiment.seed}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Metrics saved: {metrics_file}")
    print("✅ Preprocessing completed")


if __name__ == "__main__":
    main()
