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


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Preprocess data with experiment versioning"""

    data_file_path = Path(cfg.data.input_file)
    if not data_file_path.exists():
        raise FileNotFoundError(f"❌ Required input data file not found: {cfg.data.input_file}")

    metadata_file_path = Path(cfg.data.metadata_file)
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {cfg.data.metadata_file}")
    
    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.seed)
    
    print(f"🔄 Starting preprocessing for {cfg.experiment.name} (seed: {cfg.experiment.seed})")
    print(f"🎲 Experiment seed: {cfg.experiment.seed}")

    # Load data (updated path for monolithic structure)
    data = pd.read_csv(data_file_path)
    print(f"Loaded data: {data.shape}")

    metadata = SingleTableMetadata.load_from_json(metadata_file_path)
    
    if cfg.preprocessing.enabled:

        # Encode categorical features - some generators/metrics work only with numerical values
        if "encode_categorical" in cfg.preprocessing.steps:
            categorical_columns = list(filter(lambda col: metadata.columns[col].get("sdtype") == "categorical", metadata.columns.keys()))

            encoding_method = cfg.preprocessing.steps.encode_categorical
            print(f"Using {encoding_method} encoding for categorical columns")

            match encoding_method:
                case "label":
                    # Label encoding
                    label_encoders = {}
                    for col in categorical_columns:
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col])
                        label_encoders[col] = le
                        print(f"  Label encoded {col}: {data[col].nunique()} unique values")
                        # Update metadata: keep semantic type as categorical, but change representation
                        metadata.update_column(col, sdtype='categorical')

                case "onehot":
                    # One-hot encoding
                    if categorical_columns:
                        # Create one-hot encoded columns and update metadata
                        encoded_dfs = []
                        new_columns_metadata = {}
                        for col in categorical_columns:
                            one_hot = pd.get_dummies(data[col], prefix=col, dtype=int)
                            encoded_dfs.append(one_hot)
                            print(f"  One-hot encoded {col}: {one_hot.shape[1]} columns created")

                            # Track new binary columns for metadata (semantic type: boolean)
                            for new_col in one_hot.columns:
                                new_columns_metadata[new_col] = {'sdtype': 'boolean'}

                        # Drop original categorical columns and add one-hot encoded ones
                        data = data.drop(columns=categorical_columns)
                        data = pd.concat([data] + encoded_dfs, axis=1)

                        # Update metadata: remove original categorical columns, add new binary columns
                        for col in categorical_columns:
                            metadata.remove_column(col)
                        for new_col, col_metadata in new_columns_metadata.items():
                            metadata.add_column(new_col, **col_metadata)

                case "frequency":
                    # Frequency encoding
                    for col in categorical_columns:
                        # Calculate frequency map
                        freq_map = data[col].value_counts().to_dict()
                        # Replace values with their frequencies
                        data[col] = data[col].map(freq_map)
                        print(f"  Frequency encoded {col}: min_freq={data[col].min()}, max_freq={data[col].max()}")
                        # Update metadata: categorical -> numerical (frequency counts are truly numerical)
                        metadata.update_column(col, sdtype='numerical', computer_representation='Int64')

                case _:
                    raise ValueError(f"Unknown encoding method: {encoding_method}. Use 'label', 'onehot', or 'frequency'")

        # Handle missing values
        if "handle_missing" in cfg.preprocessing.steps:
            method = cfg.preprocessing.steps.handle_missing
            if method == "mean":
                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Remove outliers  
        if cfg.preprocessing.steps.get("remove_outliers", False):
            numeric_cols = data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]

        # Scale numeric features
        if cfg.preprocessing.steps.get("scale_numeric") == "standard":
            # Get only truly numerical columns from metadata, exclude categorical/boolean/datetime
            numerical_columns = []
            excluded_columns = []

            for col in data.columns:
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
                scaler = StandardScaler()
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

            if excluded_columns:
                print(f"  Preserving non-numerical columns: {[(col, dtype) for col, dtype in excluded_columns]}")

        print(f"Processed data: {data.shape}")
    else:
        print("Preprocessing disabled - using raw data")
    
    # Save processed data (updated path + experiment versioning)
    Path("experiments/data/processed").mkdir(parents=True, exist_ok=True)
    output_file = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
    data.to_csv(output_file, index=False)
    print(f"📁 Saved: {output_file}")

    # Save updated metadata (NEW)
    metadata_output_file = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}_metadata.json"
    if Path(metadata_output_file).exists():
        Path(metadata_output_file).unlink()

    metadata.save_to_json(metadata_output_file)
    print(f"🔍 Updated metadata saved: {metadata_output_file}")

    # Save metrics (updated path + experiment versioning + enhanced metadata)
    metrics = {
        "experiment_seed": cfg.experiment.seed,
        "experiment_name": cfg.experiment.name,
        "timestamp": datetime.now().isoformat(),
        "original_rows": len(pd.read_csv(cfg.data.input_file)),
        "processed_rows": len(data),
        "preprocessing_enabled": cfg.preprocessing.enabled,
        "output_file": output_file
    }
    
    Path("experiments/metrics").mkdir(parents=True, exist_ok=True)
    metrics_file = f"experiments/metrics/preprocess_{cfg.experiment.name}_{cfg.experiment.seed}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Metrics saved: {metrics_file}")
    print("✅ Preprocessing completed")


if __name__ == "__main__":
    main()
