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
from sklearn.preprocessing import StandardScaler


@hydra.main(version_base=None, config_path="../", config_name="params")
def main(cfg: DictConfig) -> None:
    """Preprocess data with experiment versioning"""
    
    # Set random seed for reproducibility
    np.random.seed(cfg.experiment.seed)
    
    print("🔄 Starting preprocessing...")
    print(f"🎲 Experiment seed: {cfg.experiment.seed}")

    # Load data (updated path for monolithic structure)
    data = pd.read_csv(cfg.data.input_file)
    print(f"Loaded data: {data.shape}")
    
    if cfg.preprocessing.enabled:
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
    output_file = f"experiments/data/processed/data_{cfg.experiment.seed}.csv"
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
    metrics_file = f"experiments/metrics/preprocess_{cfg.experiment.seed}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Metrics saved: {metrics_file}")
    print("✅ Preprocessing completed")


if __name__ == "__main__":
    main()
