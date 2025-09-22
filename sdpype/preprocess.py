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

    from sdv.metadata import SingleTableMetadata
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
                    for col in categorical_columns:
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col])
                        print(f"  Label encoded {col}: {data[col].nunique()} unique values")

                case "onehot":
                    # One-hot encoding
                    if categorical_columns:
                        # Create one-hot encoded columns
                        encoded_dfs = []
                        for col in categorical_columns:
                            one_hot = pd.get_dummies(data[col], prefix=col, dtype=int)
                            encoded_dfs.append(one_hot)
                            print(f"  One-hot encoded {col}: {one_hot.shape[1]} columns created")

                        # Drop original categorical columns and add one-hot encoded ones
                        data = data.drop(columns=categorical_columns)
                        data = pd.concat([data] + encoded_dfs, axis=1)

                case "frequency":
                    # Frequency encoding
                    for col in categorical_columns:
                        # Calculate frequency map
                        freq_map = data[col].value_counts().to_dict()
                        # Replace values with their frequencies
                        data[col] = data[col].map(freq_map)
                        print(f"  Frequency encoded {col}: min_freq={data[col].min()}, max_freq={data[col].max()}")

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
            numeric_cols = data.select_dtypes(include=['number']).columns
            scaler = StandardScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        print(f"Processed data: {data.shape}")
    else:
        print("Preprocessing disabled - using raw data")
    
    # Save processed data (updated path + experiment versioning)
    Path("experiments/data/processed").mkdir(parents=True, exist_ok=True)
    output_file = f"experiments/data/processed/data_{cfg.experiment.name}_{cfg.experiment.seed}.csv"
    data.to_csv(output_file, index=False)
    print(f"📁 Saved: {output_file}")
    
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
