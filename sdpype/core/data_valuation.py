"""
Data Valuation Module
Uses Data Shapley (Truncated Monte Carlo) to identify hallucinations in synthetic training data
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class LGBMDataValuator:
    """
    Data Shapley valuator for LGBM models using Truncated Monte Carlo Sampling

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
        random_state: int = 42,
        X_train_original: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the data valuator

        Parameters:
        -----------
        X_train : pd.DataFrame
            Synthetic training features (encoded)
        y_train : pd.Series
            Synthetic training labels
        X_test : pd.DataFrame
            Real test features (encoded)
        y_test : pd.Series
            Real test labels
        lgbm_params : dict, optional
            LightGBM parameters (if None, uses defaults)
        random_state : int
            Random seed for reproducibility
        X_train_original : pd.DataFrame, optional
            Original unencoded training features for output
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_original = X_train_original if X_train_original is not None else X_train
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

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
        self.shapley_std = None
        self.shapley_se = None
        self.shapley_ci_lower = None
        self.shapley_ci_upper = None

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

    def _train_and_evaluate(self, indices: np.ndarray) -> float:
        """
        Train LGBM on subset of data and evaluate on test set

        Parameters:
        -----------
        indices : np.ndarray
            Indices of training samples to use

        Returns:
        --------
        float : AUROC on test set
        """
        if len(indices) == 0:
            return 0.0

        # Get subset of training data
        X_subset = self.X_train.iloc[indices]
        y_subset = self.y_train.iloc[indices]

        # Train model
        model = LGBMClassifier(**self.lgbm_params)
        model.fit(X_subset, y_subset)

        # Evaluate on test set
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        auroc = roc_auc_score(self.y_test, y_pred_proba)

        return auroc

    def compute_shapley_values(
        self,
        num_samples: int = 100,
        max_coalition_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Compute Data Shapley values using Truncated Monte Carlo Sampling

        Parameters:
        -----------
        num_samples : int
            Number of random permutations to sample (default: 100)
            Higher = more accurate but slower
        max_coalition_size : int, optional
            Maximum coalition size per permutation (early truncation)
            If None, uses all training samples
            Lower = faster but less accurate
        show_progress : bool
            Show progress during computation

        Returns:
        --------
        np.ndarray : Shapley value for each training point
        """
        from rich.console import Console
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

        console = Console()
        n_train = len(self.X_train)

        # Determine coalition size
        if max_coalition_size is None:
            coalition_size = n_train
        else:
            coalition_size = min(max_coalition_size, n_train)

        # Initialize Shapley values and contribution tracking
        shapley_values = np.zeros(n_train)
        # Track all contributions for uncertainty quantification
        contributions_per_point = [[] for _ in range(n_train)]

        # Calculate total models to be trained
        total_models = num_samples * coalition_size

        if show_progress:
            console.print(f"\n[bold cyan]Computing Data Shapley Values[/bold cyan]")
            console.print(f"  Training samples: {n_train:,}")
            console.print(f"  Test samples: {len(self.X_test):,}")
            console.print(f"  Monte Carlo samples: {num_samples}")
            console.print(f"  Coalition size: {coalition_size:,}")
            if coalition_size < n_train:
                console.print(f"  [yellow]Early truncation enabled ({coalition_size}/{n_train} = {100*coalition_size/n_train:.1f}%)[/yellow]")
            console.print(f"  Total models to train: {total_models:,}")
            console.print(f"  Metric: AUROC on real test data\n")

        # Truncated Monte Carlo Shapley (TMCS)
        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    f"Training models: 0 / {total_models:,}",
                    total=num_samples
                )

                models_trained = 0
                for perm_idx in range(num_samples):
                    # Random permutation of training indices
                    perm = self.rng.permutation(n_train)

                    # Track marginal contributions for this permutation
                    prev_score = 0.0
                    contributions_this_perm = np.zeros(n_train)

                    # Truncate to max_coalition_size
                    for j in range(coalition_size):
                        idx = perm[j]

                        # Indices before current point (including current)
                        subset_indices = perm[:j+1]

                        # Train and evaluate with this subset
                        current_score = self._train_and_evaluate(subset_indices)

                        # Marginal contribution of adding point idx
                        marginal_contribution = current_score - prev_score
                        contributions_this_perm[idx] = marginal_contribution
                        shapley_values[idx] += marginal_contribution

                        prev_score = current_score
                        models_trained += 1

                    # Store contributions for uncertainty quantification
                    for idx in range(n_train):
                        contributions_per_point[idx].append(contributions_this_perm[idx])

                    # Update progress with models trained count
                    models_remaining = total_models - models_trained
                    progress.update(
                        task,
                        advance=1,
                        description=f"Training models: {models_trained:,} / {total_models:,} ({models_remaining:,} remaining)"
                    )
        else:
            for _ in range(num_samples):
                # Random permutation of training indices
                perm = self.rng.permutation(n_train)

                # Track marginal contributions for this permutation
                prev_score = 0.0
                contributions_this_perm = np.zeros(n_train)

                # Truncate to max_coalition_size
                for j in range(coalition_size):
                    idx = perm[j]

                    # Indices before current point (including current)
                    subset_indices = perm[:j+1]

                    # Train and evaluate with this subset
                    current_score = self._train_and_evaluate(subset_indices)

                    # Marginal contribution of adding point idx
                    marginal_contribution = current_score - prev_score
                    contributions_this_perm[idx] = marginal_contribution
                    shapley_values[idx] += marginal_contribution

                    prev_score = current_score

                # Store contributions for uncertainty quantification
                for idx in range(n_train):
                    contributions_per_point[idx].append(contributions_this_perm[idx])

        # Average over all permutations
        shapley_values /= num_samples

        # Compute uncertainty metrics for each data point
        shapley_std = np.zeros(n_train)
        shapley_se = np.zeros(n_train)
        shapley_ci_lower = np.zeros(n_train)
        shapley_ci_upper = np.zeros(n_train)

        for idx in range(n_train):
            contributions = np.array(contributions_per_point[idx])
            shapley_std[idx] = np.std(contributions, ddof=1)  # Sample std
            shapley_se[idx] = shapley_std[idx] / np.sqrt(num_samples)  # Standard error
            # 95% confidence interval (1.96 * SE)
            shapley_ci_lower[idx] = shapley_values[idx] - 1.96 * shapley_se[idx]
            shapley_ci_upper[idx] = shapley_values[idx] + 1.96 * shapley_se[idx]

        self.shapley_values = shapley_values
        self.shapley_std = shapley_std
        self.shapley_se = shapley_se
        self.shapley_ci_lower = shapley_ci_lower
        self.shapley_ci_upper = shapley_ci_upper

        if show_progress:
            console.print(f"\n[bold green]✓ Shapley values computed![/bold green]")
            console.print(f"  Mean value: {shapley_values.mean():.6f}")
            console.print(f"  Std value: {shapley_values.std():.6f}")
            console.print(f"  Min value: {shapley_values.min():.6f}")
            console.print(f"  Max value: {shapley_values.max():.6f}")
            console.print(f"\n[bold cyan]Uncertainty Metrics:[/bold cyan]")
            console.print(f"  Mean standard error: {shapley_se.mean():.6f}")
            console.print(f"  Max standard error: {shapley_se.max():.6f}")

            # Count reliably negative values (CI upper bound < 0)
            reliable_negative_count = (shapley_ci_upper < 0).sum()
            reliable_negative_pct = 100 * reliable_negative_count / len(shapley_values)

            # Count negative values (potential hallucinations)
            negative_count = (shapley_values < 0).sum()
            negative_pct = 100 * negative_count / len(shapley_values)

            console.print(f"\n  [bold yellow]Negative values (potential hallucinations):[/bold yellow]")
            console.print(f"    Total negative: {negative_count:,} ({negative_pct:.1f}%)")
            console.print(f"    Reliably negative (CI upper < 0): {reliable_negative_count:,} ({reliable_negative_pct:.1f}%)")
            console.print(f"    These are definitively harmful with 95% confidence")

        return shapley_values

    def save_results(
        self,
        output_path: Path,
        include_features: bool = True
    ) -> Path:
        """
        Save Shapley values with uncertainty metrics to CSV

        Parameters:
        -----------
        output_path : Path
            Output CSV file path
        include_features : bool
            If True, includes all features in the output CSV (decoded/original values)
            If False, only includes Shapley metrics and index

        Returns:
        --------
        Path : Path to saved CSV file

        Output columns:
        ---------------
        - [Original feature columns]: Unencoded/human-readable values (if include_features=True)
        - target: Target variable
        - shapley_value: Mean Shapley value
        - shapley_std: Standard deviation across permutations
        - shapley_se: Standard error (std / sqrt(n))
        - shapley_ci_lower: Lower bound of 95% CI
        - shapley_ci_upper: Upper bound of 95% CI
        - sample_index: Original index in training data
        """
        if self.shapley_values is None:
            raise ValueError("Must call compute_shapley_values() first")

        # Create output dataframe
        if include_features:
            # Include all features (use original unencoded data)
            results_df = self.X_train_original.copy()
            results_df['target'] = self.y_train.values
        else:
            # Only index
            results_df = pd.DataFrame()

        # Add Shapley value and uncertainty metrics
        results_df['shapley_value'] = self.shapley_values
        results_df['shapley_std'] = self.shapley_std
        results_df['shapley_se'] = self.shapley_se
        results_df['shapley_ci_lower'] = self.shapley_ci_lower
        results_df['shapley_ci_upper'] = self.shapley_ci_upper

        # Add index for reference
        results_df['sample_index'] = range(len(self.shapley_values))

        # Sort by Shapley value (most harmful first)
        results_df = results_df.sort_values('shapley_value', ascending=True)

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        return output_path


def _compute_proportion_ci(
    count: int,
    total: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion

    This is more accurate than the normal approximation, especially for
    small samples or proportions near 0 or 1.

    Parameters:
    -----------
    count : int
        Number of successes (e.g., negative Shapley values)
    total : int
        Total number of samples
    confidence_level : float
        Confidence level (default: 0.95 for 95% CI)

    Returns:
    --------
    tuple : (lower_bound, upper_bound) as percentages (0-100)
    """
    if total == 0:
        return (0.0, 0.0)

    # Observed proportion
    p = count / total

    # Z-score for confidence level
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)

    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    lower = max(0.0, (center - margin) * 100)
    upper = min(100.0, (center + margin) * 100)

    return (lower, upper)


def run_data_valuation(
    train_file: Path,
    test_file: Path,
    target_column: str = "IS_READMISSION_30D",
    num_samples: int = 100,
    max_coalition_size: Optional[int] = None,
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
    max_coalition_size : int, optional
        Maximum coalition size per permutation (early truncation)
        If None, uses all training samples
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

    # Store original unencoded data for output
    X_train_original = X_train.copy()

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
        random_state=random_state,
        X_train_original=X_train_original
    )

    # Compute Shapley values
    shapley_values = valuator.compute_shapley_values(
        num_samples=num_samples,
        max_coalition_size=max_coalition_size,
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

    # Compute 95% confidence interval for negative percentage
    ci_lower, ci_upper = _compute_proportion_ci(negative_count, len(shapley_values), confidence_level=0.95)

    # Count reliably negative values (CI upper bound < 0)
    reliable_negative_count = (valuator.shapley_ci_upper < 0).sum()
    reliable_negative_pct = 100 * reliable_negative_count / len(shapley_values)
    reliable_ci_lower, reliable_ci_upper = _compute_proportion_ci(
        reliable_negative_count, len(shapley_values), confidence_level=0.95
    )

    console.print(f"\n  [bold yellow]Negative values (potential hallucinations):[/bold yellow]")
    console.print(f"    Count: {negative_count:,} ({negative_pct:.1f}%)")
    console.print(f"    95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]  (Wilson score interval)")
    console.print(f"\n  [bold red]Reliably negative (CI upper < 0):[/bold red]")
    console.print(f"    Count: {reliable_negative_count:,} ({reliable_negative_pct:.1f}%)")
    console.print(f"    95% CI: [{reliable_ci_lower:.1f}%, {reliable_ci_upper:.1f}%]")
    console.print(f"    These samples are definitively harmful with 95% confidence")

    # Return results
    return {
        'output_path': str(saved_path),
        'num_samples': len(shapley_values),
        'mean_shapley': float(shapley_values.mean()),
        'std_shapley': float(shapley_values.std()),
        'min_shapley': float(shapley_values.min()),
        'max_shapley': float(shapley_values.max()),
        'mean_shapley_se': float(valuator.shapley_se.mean()),
        'max_shapley_se': float(valuator.shapley_se.max()),
        'negative_count': int(negative_count),
        'negative_percentage': float(negative_pct),
        'negative_percentage_ci_lower': float(ci_lower),
        'negative_percentage_ci_upper': float(ci_upper),
        'reliable_negative_count': int(reliable_negative_count),
        'reliable_negative_percentage': float(reliable_negative_pct),
        'reliable_negative_ci_lower': float(reliable_ci_lower),
        'reliable_negative_ci_upper': float(reliable_ci_upper),
        'confidence_level': 0.95
    }
