# sdpype/core/setup.py - Repository setup logic
"""
Repository setup and initialization functions
"""

import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata

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
    create_sample_data_with_preprocessing()
    create_dvc_ignore()
    show_completion_message()

def create_directory_structure():
    """Create the experiments directory structure"""
    dirs = [
        "experiments/data/raw",
        "experiments/data/processed",
        "experiments/data/synthetic",
        "experiments/checkpoints",
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

def preprocess_data(data: pd.DataFrame, fit_scalers: bool = True, scalers_dict: dict = None):
    """
    Apply preprocessing steps to data

    Note: Categorical columns are encoded but NOT scaled.
    Only originally numeric columns are scaled.

    Args:
        data: DataFrame to preprocess
        fit_scalers: Whether to fit new scalers (True for first dataset, False for second)
        scalers_dict: Dictionary of fitted scalers (when fit_scalers=False)

    Returns:
        tuple: (processed_data, scalers_dict)
    """
    data = data.copy()

    if scalers_dict is None:
        scalers_dict = {}

    # Remember which columns were originally numeric before any transformations
    if fit_scalers:
        scalers_dict['original_numeric_columns'] = list(data.select_dtypes(include=[np.number]).columns)

    # 1. Handle missing values (mean for numeric, mode for categorical)
    console.print("  🔧 Handling missing values...")
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype in ['object', 'category']:
                # Fill categorical with mode
                mode_value = data[column].mode()[0] if not data[column].mode().empty else 'Unknown'
                data[column].fillna(mode_value, inplace=True)
            else:
                # Fill numeric with mean
                mean_value = data[column].mean()
                data[column].fillna(mean_value, inplace=True)

    # 2. Encode categorical variables (label encoding)
    console.print("  🔧 Encoding categorical variables...")
    categorical_columns = data.select_dtypes(include=['object']).columns

    for column in categorical_columns:
        if fit_scalers:
            # Fit new encoder
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            scalers_dict[f'{column}_encoder'] = le
        else:
            # Use existing encoder
            if f'{column}_encoder' in scalers_dict:
                le = scalers_dict[f'{column}_encoder']
                # Handle unseen categories
                unique_values = set(data[column].astype(str))
                known_values = set(le.classes_)
                unknown_values = unique_values - known_values

                if unknown_values:
                    console.print(f"  ⚠️  Unknown categories in {column}: {unknown_values}")
                    # Map unknown values to a default (first class)
                    data[column] = data[column].astype(str).map(
                        lambda x: x if x in known_values else le.classes_[0]
                    )

                data[column] = le.transform(data[column].astype(str))

    # 3. Remove outliers (IQR method for numeric columns)
    console.print("  🔧 Removing outliers...")
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_count = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
        if outliers_count > 0:
            console.print(f"    📊 Removing {outliers_count} outliers from {column}")
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    # 4. Scale numeric variables (standard scaling)
    console.print("  🔧 Scaling originally numeric variables (excluding encoded categoricals)...")
    # Only scale columns that were originally numeric, not encoded categorical ones
    original_numeric_columns = scalers_dict.get('original_numeric_columns', [])
    columns_to_scale = [col for col in original_numeric_columns if col in data.columns]

    if len(columns_to_scale) > 0:
        console.print(f"    📊 Scaling columns: {columns_to_scale}")
        if fit_scalers:
            # Fit new scaler
            scaler = StandardScaler()
            data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
            scalers_dict['numeric_scaler'] = scaler
        else:
            # Use existing scaler
            if 'numeric_scaler' in scalers_dict:
                scaler = scalers_dict['numeric_scaler']
                data[columns_to_scale] = scaler.transform(data[columns_to_scale])
    else:
        console.print("    📊 No originally numeric columns to scale")

    return data, scalers_dict

def create_sample_data_with_preprocessing():
    """Create larger sample dataset with train/reference split and individual preprocessing"""

    console.print("📊 Creating sample dataset with preprocessing...")

    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 6000

    console.print(f"🎲 Generating {n_samples} records...")

    # Create more diverse sample data
    data = pd.DataFrame({
        'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'income': np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 250000).astype(int),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'],
                                    n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'region': np.random.choice(['North', 'South', 'East', 'West'],
                                  n_samples, p=[0.25, 0.25, 0.25, 0.25])
     })

    # Add some missing values to test missing value handling
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    data.loc[missing_indices[:len(missing_indices)//2], 'income'] = np.nan
    data.loc[missing_indices[len(missing_indices)//2:], 'education'] = np.nan

    # Add some outliers to test outlier removal
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    data.loc[outlier_indices, 'income'] = data.loc[outlier_indices, 'income'] * 5  # Create income outliers

    console.print(f"📊 Created dataset: {data.shape}")
    console.print(f"📊 Missing values: {data.isnull().sum().sum()}")

    # Split data randomly: 4K reference, 2K training (truly random, no stratification)
    console.print("🔄 Splitting data: 4K reference, 2K training...")

    reference_data, training_data = train_test_split(
        data,
        test_size=2000,  # 2K for training
        train_size=4000,  # 4K for reference
        random_state=42  # No stratification - truly random
    )

    console.print(f"📊 Reference data: {reference_data.shape}")
    console.print(f"📊 Training data: {training_data.shape}")

    # Preprocess each split individually
    console.print("\n🔧 Preprocessing reference data...")
    reference_processed, scalers_dict = preprocess_data(reference_data, fit_scalers=True)

    console.print(f"📊 Reference data after preprocessing: {reference_processed.shape}")

    console.print("\n🔧 Preprocessing training data...")
    training_processed, _ = preprocess_data(training_data, fit_scalers=False, scalers_dict=scalers_dict)

    console.print(f"📊 Training data after preprocessing: {training_processed.shape}")

    # Ensure processed directory exists
    processed_dir = Path("experiments/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save processed datasets
    ref_file = processed_dir / "ref_sample_data.csv"
    trn_file = processed_dir / "trn_sample_data.csv"

    reference_processed.to_csv(ref_file, index=False)
    training_processed.to_csv(trn_file, index=False)

    console.print(f"💾 Saved reference data: {ref_file}")
    console.print(f"💾 Saved training data: {trn_file}")

    # Create and save metadata using SDV
    console.print("🔍 Creating metadata...")

    # Use reference data to create metadata (larger dataset)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(reference_processed)

    # Manually set appropriate data types based on original columns
    # After preprocessing, everything becomes numeric, but we need to specify semantic types
    column_mapping = {
        'age': {'sdtype': 'numerical'},
        'income': {'sdtype': 'numerical'},
        'score': {'sdtype': 'numerical'},
        'education': {'sdtype': 'categorical'},  # Was categorical, now encoded as numerical
        'category': {'sdtype': 'categorical'},   # Was categorical, now encoded as numerical
        'region': {'sdtype': 'categorical'}      # Was categorical, now encoded as numerical
    }

    for column, config in column_mapping.items():
        if column in metadata.columns:
            metadata.update_column(column, **config)

    # Save metadata
    metadata_file = processed_dir / "sample_data_metadata.json"
    metadata.save_to_json(metadata_file)

    console.print(f"🔍 Saved metadata: {metadata_file}")

    # Save preprocessing info for reference
    preprocessing_info = {
        "preprocessing_applied": True,
        "steps": {
            "encode_categorical": "label",
            "handle_missing": "mean",
            "remove_outliers": True,
            "scale_numeric": "standard"
        },
        "original_shape": data.shape,
        "reference_shape": reference_processed.shape,
        "training_shape": training_processed.shape,
        "categorical_columns_original": list(data.select_dtypes(include=['object']).columns),
        "numeric_columns_original": list(data.select_dtypes(include=[np.number]).columns),
        "split_strategy": "random (no stratification)",
        "seed": 42
    }

    preprocessing_info_file = processed_dir / "preprocessing_info.json"
    with open(preprocessing_info_file, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)

    console.print(f"📋 Saved preprocessing info: {preprocessing_info_file}")

    # Print summary
    console.print("\n📈 Preprocessing Summary:")
    console.print(f"  📊 Original dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    console.print(f"  📊 Reference dataset: {reference_processed.shape[0]} rows (after outlier removal)")
    console.print(f"  📊 Training dataset: {training_processed.shape[0]} rows (after outlier removal)")
    console.print(f"  🔧 Categorical columns encoded: {len(data.select_dtypes(include=['object']).columns)}")
    console.print(f"  🔧 Missing values handled: {data.isnull().sum().sum()} values")
    console.print("  ✅ Both datasets preprocessed individually")

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
        "📊 Created 6K sample dataset with preprocessing:\n"
        "  • 4K reference data (experiments/data/processed/ref_sample_data.csv)\n"
        "  • 2K training data (experiments/data/processed/trn_sample_data.csv)\n"
        "  • Metadata file (experiments/data/processed/sample_data_metadata.json)\n\n"
        "🔧 Preprocessing applied to each split individually:\n"
        "  • Categorical encoding (label)\n"
        "  • Missing value imputation (mean/mode)\n"
        "  • Outlier removal (IQR method)\n"
        "  • Numeric scaling (standard)\n\n"
        "Next steps:\n"
        "• Run: [bold]sdpype models[/bold] (see available models)\n"
        "• Run: [bold]sdpype pipeline[/bold] (run full pipeline)\n"
        "• Check: [bold]params.yaml[/bold] (verify data file paths)",
        title="🎉 Setup Complete"
    ))
    