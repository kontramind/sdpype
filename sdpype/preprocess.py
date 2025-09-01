"""
Minimal preprocessing module for SDPype
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Preprocess data"""
    
    print("🔄 Starting preprocessing...")
    
    # Load data
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
    
    # Save processed data
    Path("data/processed").mkdir(exist_ok=True)
    data.to_csv("data/processed/data.csv", index=False)
    
    # Save metrics
    metrics = {
        "original_rows": len(pd.read_csv(cfg.data.input_file)),
        "processed_rows": len(data),
        "preprocessing_enabled": cfg.preprocessing.enabled
    }
    
    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/preprocess.json", "w") as f:
        json.dump(metrics, f)
    
    print("✅ Preprocessing completed")


if __name__ == "__main__":
    main()
