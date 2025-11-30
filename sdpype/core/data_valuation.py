"""
Data Valuation Module
Uses Data Shapley to identify hallucinations in synthetic training data
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings

# OpenDataVal imports
from opendataval.dataval import DataShapley
from opendataval.model import ClassifierSkLearnWrapper
from opendataval.dataloader import DataFetcher

warnings.filterwarnings('ignore')


class LGBMDataValuator:
    """
    Data Shapley valuator for LGBM models

    Measures the marginal contribution of each synthetic training point
    to the model's performance on real test data.

    Negative Shapley values indicate harmful "hallucinations" that degrade
    performance on real data.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        lgbm_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize the data valuator

        Parameters:
        -----------
        X_train : pd.DataFrame
            Synthetic training features
        y_train : pd.Series
            Synthetic training labels
        X_test : pd.DataFrame
            Real test features
        y_test : pd.Series
            Real test labels
        lgbm_params : dict, optional
            LightGBM parameters (if None, uses defaults)
        random_state : int
            Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state

        # Default LGBM parameters if not provided
        if lgbm_params is None:
            self.lgbm_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'random_state': random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
        else:
            # Process provided parameters
            self.lgbm_params = self._process_lgbm_params(lgbm_params.copy())

        self.shapley_values = None

    def _process_lgbm_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process loaded LGBM parameters and handle special cases

        Parameters:
        -----------
        params : dict
            Raw parameters from training JSON

        Returns:
        --------
        dict : Processed LGBM parameters ready for LGBMClassifier
        """
        processed_params = params.copy()

        # Set required parameters
        processed_params['objective'] = 'binary'
        processed_params['metric'] = 'auc'
        processed_params['random_state'] = self.random_state
        processed_params['n_jobs'] = -1
        processed_params['verbosity'] = -1

        # Handle imbalance_method (convert to LGBM params)
        if 'imbalance_method' in processed_params:
            imbalance_method = processed_params.pop('imbalance_method')

            if imbalance_method == 'scale_pos_weight':
                # Calculate scale_pos_weight from training data
                neg_count = (self.y_train == 0).sum()
                pos_count = (self.y_train == 1).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                processed_params['scale_pos_weight'] = scale_pos_weight
            elif imbalance_method == 'is_unbalance':
                processed_params['is_unbalance'] = True
            # else: 'none' - no special handling

        # Remove early_stopping_rounds (not used in sklearn interface during fit)
        if 'early_stopping_rounds' in processed_params:
            processed_params.pop('early_stopping_rounds')

        # Remove optimal_threshold if present (not an LGBM param)
        if 'optimal_threshold' in processed_params:
            processed_params.pop('optimal_threshold')

        return processed_params

    def _create_lgbm_sklearn_wrapper(self):
        """
        Create sklearn-compatible LGBM classifier for opendataval

        Returns:
        --------
        lgb.LGBMClassifier : Sklearn-compatible LGBM model
        """
        from lightgbm import LGBMClassifier

        # Extract parameters for sklearn interface
        params = self.lgbm_params.copy()

        return LGBMClassifier(**params)

    def _auc_metric(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        AUC metric for opendataval

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities (2D array for binary classification)

        Returns:
        --------
        float : AUROC score
        """
        # Handle both 1D and 2D probability arrays
        if len(y_pred_proba.shape) == 2:
            y_pred_proba = y_pred_proba[:, 1]

        return roc_auc_score(y_true, y_pred_proba)

    def compute_shapley_values(
        self,
        num_samples: int = 100,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Compute Data Shapley values for each training point

        Uses Truncated Monte Carlo Sampling (TMCS) to approximate Shapley values.

        Parameters:
        -----------
        num_samples : int
            Number of random samples for TMCS approximation
            Higher = more accurate but slower (default: 100)
        show_progress : bool
            Show progress during computation

        Returns:
        --------
        np.ndarray : Shapley value for each training point
        """
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn

        console = Console()

        # Create sklearn-compatible model
        lgbm_model = self._create_lgbm_sklearn_wrapper()

        # Wrap model for opendataval
        pred_model = ClassifierSkLearnWrapper(
            model=lgbm_model,
            num_classes=2
        )

        # Create custom DataFetcher with our data
        # opendataval expects data in specific format
        fetcher = self._create_data_fetcher()

        if show_progress:
            console.print(f"\n[bold cyan]Computing Data Shapley Values[/bold cyan]")
            console.print(f"  Training samples: {len(self.X_train):,}")
            console.print(f"  Test samples: {len(self.X_test):,}")
            console.print(f"  Monte Carlo samples: {num_samples}")
            console.print(f"  Metric: AUROC on real test data\n")

        # Compute Data Shapley values
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Computing Shapley values...", total=None)

                dataval_evaluator = DataShapley(
                    num_rand_samp=num_samples
                ).train(
                    fetcher=fetcher,
                    pred_model=pred_model,
                    metric=self._auc_metric
                )

                self.shapley_values = dataval_evaluator.evaluate_data_values()
        else:
            dataval_evaluator = DataShapley(
                num_rand_samp=num_samples
            ).train(
                fetcher=fetcher,
                pred_model=pred_model,
                metric=self._auc_metric
            )

            self.shapley_values = dataval_evaluator.evaluate_data_values()

        if show_progress:
            console.print(f"\n[bold green]✓ Shapley values computed![/bold green]")
            console.print(f"  Mean value: {self.shapley_values.mean():.6f}")
            console.print(f"  Std value: {self.shapley_values.std():.6f}")
            console.print(f"  Min value: {self.shapley_values.min():.6f}")
            console.print(f"  Max value: {self.shapley_values.max():.6f}")

            # Count negative values (potential hallucinations)
            negative_count = (self.shapley_values < 0).sum()
            negative_pct = 100 * negative_count / len(self.shapley_values)
            console.print(f"\n  Negative values (potential hallucinations): {negative_count:,} ({negative_pct:.1f}%)")

        return self.shapley_values

    def _create_data_fetcher(self) -> DataFetcher:
        """
        Create DataFetcher for opendataval

        Returns:
        --------
        DataFetcher : Custom data fetcher with our train/test split
        """
        # Convert to numpy arrays
        X_train_np = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        y_train_np = self.y_train.values if isinstance(self.y_train, pd.Series) else self.y_train
        X_test_np = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
        y_test_np = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test

        # Create fetcher with custom data
        # Note: opendataval's DataFetcher expects (X, y, X_val, y_val, X_test, y_test)
        # We use synthetic data as training, real data as test
        fetcher = DataFetcher(
            dataset_name='custom_synthetic',
            train_count=len(X_train_np),
            valid_count=0,  # No validation split needed
            random_state=self.random_state
        )

        # Manually set the data
        fetcher.datapoints = (
            X_train_np,
            y_train_np,
            None,  # No validation set
            None,
            X_test_np,
            y_test_np
        )

        return fetcher

    def save_results(
        self,
        output_path: Path,
        include_features: bool = True
    ) -> Path:
        """
        Save Shapley values to CSV

        Parameters:
        -----------
        output_path : Path
            Output CSV file path
        include_features : bool
            If True, includes all features in the output CSV
            If False, only includes Shapley value and index

        Returns:
        --------
        Path : Path to saved CSV file
        """
        if self.shapley_values is None:
            raise ValueError("Must call compute_shapley_values() first")

        # Create output dataframe
        if include_features:
            # Include all features
            results_df = self.X_train.copy()
            results_df['target'] = self.y_train.values
        else:
            # Only index
            results_df = pd.DataFrame()

        # Add Shapley value
        results_df['shapley_value'] = self.shapley_values

        # Add index for reference
        results_df['sample_index'] = range(len(self.shapley_values))

        # Sort by Shapley value (most harmful first)
        results_df = results_df.sort_values('shapley_value', ascending=True)

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        return output_path


def run_data_valuation(
    train_file: Path,
    test_file: Path,
    target_column: str = "IS_READMISSION_30D",
    num_samples: int = 100,
    random_state: int = 42,
    output_dir: Path = Path("experiments/data_valuation"),
    lgbm_params: Optional[Dict[str, Any]] = None,
    encoding_config: Optional[Path] = None,
    include_features: bool = True
) -> Dict[str, Any]:
    """
    Run data valuation analysis on synthetic training data

    Parameters:
    -----------
    train_file : Path
        Path to synthetic training data CSV
    test_file : Path
        Path to real test data CSV
    target_column : str
        Name of the target column
    num_samples : int
        Number of Monte Carlo samples for Shapley approximation
    random_state : int
        Random seed
    output_dir : Path
        Output directory for results
    lgbm_params : dict, optional
        LightGBM parameters (if None, uses defaults)
    encoding_config : Path, optional
        Path to RDT encoding config YAML
    include_features : bool
        If True, includes all features in output CSV

    Returns:
    --------
    dict : Results summary
    """
    from rich.console import Console

    console = Console()

    # Load data
    console.print("\n[bold cyan]Loading data...[/bold cyan]")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Validate target column
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data")

    # Separate features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Encode categorical columns (same as downstream.py)
    if encoding_config is not None:
        console.print(f"\n[bold cyan]Using RDT encoding from config...[/bold cyan]")
        console.print(f"  Config: {encoding_config}")

        from sdpype.encoding import load_encoding_config, RDTDatasetEncoder

        # Load encoding config
        enc_config = load_encoding_config(encoding_config)

        # Filter out target column
        if target_column in enc_config['sdtypes']:
            enc_config['sdtypes'] = {k: v for k, v in enc_config['sdtypes'].items() if k != target_column}
        if target_column in enc_config['transformers']:
            enc_config['transformers'] = {k: v for k, v in enc_config['transformers'].items() if k != target_column}

        # Create and fit encoder
        rdt_encoder = RDTDatasetEncoder(enc_config)
        rdt_encoder.fit(X_train)

        # Transform both datasets
        X_train = rdt_encoder.transform(X_train)
        X_test = rdt_encoder.transform(X_test)

        console.print(f"  ✓ Encoded {X_train.shape[1]} features using RDT transformers")
    else:
        # Simple LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

        if categorical_cols:
            console.print(f"\n[bold cyan]Encoding categorical features (LabelEncoder)...[/bold cyan]")

            for col in categorical_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                console.print(f"  ✓ {col}: {len(le.classes_)} categories")

    # Display data info
    console.print(f"\n[bold cyan]Data Summary:[/bold cyan]")
    console.print(f"  Training samples (synthetic): {len(X_train):,}")
    console.print(f"  Test samples (real): {len(X_test):,}")
    console.print(f"  Features: {X_train.shape[1]}")
    console.print(f"  Target: {target_column}")

    # Initialize valuator
    valuator = LGBMDataValuator(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        lgbm_params=lgbm_params,
        random_state=random_state
    )

    # Compute Shapley values
    shapley_values = valuator.compute_shapley_values(
        num_samples=num_samples,
        show_progress=True
    )

    # Save results
    console.print(f"\n[bold cyan]Saving results...[/bold cyan]")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"data_valuation_{timestamp}.csv"

    saved_path = valuator.save_results(
        output_path=output_path,
        include_features=include_features
    )

    console.print(f"[bold green]✓ Results saved to: {saved_path}[/bold green]")

    # Summary statistics
    console.print(f"\n[bold cyan]Summary Statistics:[/bold cyan]")
    console.print(f"  Total samples: {len(shapley_values):,}")
    console.print(f"  Mean Shapley value: {shapley_values.mean():.6f}")
    console.print(f"  Std Shapley value: {shapley_values.std():.6f}")
    console.print(f"  Min Shapley value: {shapley_values.min():.6f}")
    console.print(f"  Max Shapley value: {shapley_values.max():.6f}")

    negative_count = (shapley_values < 0).sum()
    negative_pct = 100 * negative_count / len(shapley_values)
    console.print(f"\n  [bold yellow]Negative values (potential hallucinations):[/bold yellow]")
    console.print(f"    Count: {negative_count:,} ({negative_pct:.1f}%)")
    console.print(f"    These samples may harm model performance on real data")

    # Return results
    return {
        'output_path': str(saved_path),
        'num_samples': len(shapley_values),
        'mean_shapley': float(shapley_values.mean()),
        'std_shapley': float(shapley_values.std()),
        'min_shapley': float(shapley_values.min()),
        'max_shapley': float(shapley_values.max()),
        'negative_count': int(negative_count),
        'negative_percentage': float(negative_pct)
    }
