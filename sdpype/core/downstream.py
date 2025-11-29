"""
Downstream Task Training Module
Implements LGBM hyperparameter tuning with Bayesian Optimization for binary classification tasks.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')


class LGBMBayesianTuner:
    """
    LGBM hyperparameter tuner using Bayesian Optimization

    Optimizes hyperparameters using Optuna with Tree-structured Parzen Estimator (TPE)
    and k-fold cross-validation with AUROC as the primary metric.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: int = 5,
        n_trials: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the tuner

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (binary)
        n_folds : int
            Number of cross-validation folds (default: 5)
        n_trials : int
            Number of Bayesian optimization trials (default: 100)
        random_state : int
            Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.study = None
        self.best_threshold = 0.5  # Default threshold

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization

        Parameters:
        -----------
        trial : optuna.Trial
            A trial object from Optuna

        Returns:
        --------
        float : Mean AUROC across folds
        """

        # Define hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 4, 60),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 2**(-10), 2**0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 60),
            'n_estimators': 1000,  # Large number, will use early stopping
            'random_state': self.random_state,
            'n_jobs': -1
        }

        # Regularization parameters (helps with overfitting)
        params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 10.0)  # L1 regularization
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 10.0)  # L2 regularization

        # Feature sampling (column subsampling - helps generalization)
        params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 1.0)

        # Leaf constraints (prevents too-specific leaves)
        params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 50)

        # Class imbalance handling (mutually exclusive options)
        imbalance_method = trial.suggest_categorical(
            'imbalance_method',
            ['none', 'scale_pos_weight', 'is_unbalance']
        )

        if imbalance_method == 'scale_pos_weight':
            # Scale positive weight: ratio of negative to positive samples
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            params['scale_pos_weight'] = scale_pos_weight
        elif imbalance_method == 'is_unbalance':
            params['is_unbalance'] = True
        # else: no imbalance handling

        # Early stopping rounds
        early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 7, 30)

        # Perform k-fold cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train model with early stopping
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=0)  # Suppress output
                ]
            )

            # Predict on validation set
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)

            # Calculate AUROC
            auroc = roc_auc_score(y_val, y_pred)
            cv_scores.append(auroc)

        # Return mean AUROC across folds
        return np.mean(cv_scores)

    def tune(self, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Bayesian Optimization to find best hyperparameters

        Parameters:
        -----------
        timeout : int, optional
            Maximum time in seconds for optimization

        Returns:
        --------
        dict : Best hyperparameters found
        """

        # Create Optuna study with TPE sampler (Bayesian Optimization)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )

        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=timeout,
            show_progress_bar=False
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return self.best_params

    def get_best_model_params(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get the best parameters formatted for LightGBM model training

        Returns:
        --------
        tuple : (lgbm_params, preprocessing_flags)
        """
        if self.best_params is None:
            raise ValueError("Must run tune() before getting best parameters")

        # Extract early stopping
        early_stopping_rounds = self.best_params.get('early_stopping_rounds', 10)

        # Format parameters for LightGBM
        lgbm_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': self.best_params['boosting_type'],
            'num_leaves': self.best_params['num_leaves'],
            'max_depth': self.best_params['max_depth'],
            'learning_rate': self.best_params['learning_rate'],
            'min_child_samples': self.best_params['min_child_samples'],
            'n_estimators': 1000,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1,
            # Regularization
            'reg_alpha': self.best_params.get('reg_alpha', 0.0),
            'reg_lambda': self.best_params.get('reg_lambda', 0.0),
            # Feature sampling
            'feature_fraction': self.best_params.get('feature_fraction', 1.0),
            # Leaf constraints
            'min_data_in_leaf': self.best_params.get('min_data_in_leaf', 20),
        }

        # Add class imbalance handling if selected
        imbalance_method = self.best_params.get('imbalance_method', 'none')
        if imbalance_method == 'scale_pos_weight':
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            lgbm_params['scale_pos_weight'] = scale_pos_weight
        elif imbalance_method == 'is_unbalance':
            lgbm_params['is_unbalance'] = True

        preprocessing_flags = {
            'early_stopping_rounds': early_stopping_rounds
        }

        return lgbm_params, preprocessing_flags

    def train_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> lgb.Booster:
        """
        Train final model with best parameters

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation labels

        Returns:
        --------
        lgb.Booster : Trained LightGBM model
        """
        if self.best_params is None:
            raise ValueError("Must run tune() before training final model")

        lgbm_params, preprocessing_flags = self.get_best_model_params()

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        # Train model
        model = lgb.train(
            lgbm_params,
            train_data,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=preprocessing_flags['early_stopping_rounds']
                ),
                lgb.log_evaluation(period=0)
            ]
        )

        return model

    def find_optimal_threshold(
        self,
        model: lgb.Booster,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = 'f1'
    ) -> float:
        """
        Find optimal classification threshold on validation data

        Parameters:
        -----------
        model : lgb.Booster
            Trained model
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation labels
        metric : str
            Metric to optimize ('f1', 'precision', 'recall')

        Returns:
        --------
        float : Optimal threshold
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Get predicted probabilities
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)

        # Try thresholds from 0.1 to 0.9
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.best_threshold = best_threshold
        return best_threshold


def evaluate_model(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate model on test set with multiple metrics

    Parameters:
    -----------
    model : lgb.Booster
        Trained LightGBM model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    threshold : float
        Classification threshold for binary predictions

    Returns:
    --------
    dict : Dictionary of evaluation metrics including confusion matrix
    """
    from sklearn.metrics import confusion_matrix

    # Predict probabilities
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)

    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    metrics = {
        'auroc': float(roc_auc_score(y_test, y_pred_proba)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'threshold': threshold,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }

    return metrics


def display_confusion_matrix(
    cm_dict: Dict[str, int],
    console
) -> None:
    """
    Display confusion matrix using Rich Table

    Parameters:
    -----------
    cm_dict : dict
        Dictionary with keys: tn, fp, fn, tp
    console : Console
        Rich console object
    """
    from rich.table import Table

    tn, fp, fn, tp = cm_dict['tn'], cm_dict['fp'], cm_dict['fn'], cm_dict['tp']
    total = tn + fp + fn + tp

    table = Table(title="Confusion Matrix", show_header=True, header_style="bold cyan")
    table.add_column("", style="cyan", width=12)
    table.add_column("Predicted: No", justify="right", width=18)
    table.add_column("Predicted: Yes", justify="right", width=18)
    table.add_column("Total", justify="right", width=12, style="bold")

    table.add_row(
        "Actual: No",
        f"{tn:,} (TN)",
        f"[red]{fp:,} (FP)[/red]",
        f"{tn+fp:,}"
    )
    table.add_row(
        "Actual: Yes",
        f"[red]{fn:,} (FN)[/red]",
        f"[green]{tp:,} (TP)[/green]",
        f"{fn+tp:,}"
    )
    table.add_row(
        "Total",
        f"{tn+fn:,}",
        f"{fp+tp:,}",
        f"{total:,}",
        style="bold"
    )

    console.print("\n")
    console.print(table)

    # Add legend
    console.print("\nWhere:")
    console.print("  [green]TP (True Positives)[/green]:  Correctly identified readmissions")
    console.print("  [red]FP (False Positives)[/red]: False alarms (predicted readmission, didn't happen)")
    console.print("  TN (True Negatives):   Correctly identified non-readmissions")
    console.print("  [red]FN (False Negatives)[/red]: Missed readmissions (predicted no readmission, but happened)")


def display_clinical_performance(
    cm_dict: Dict[str, int],
    console
) -> None:
    """
    Display clinical performance summary using Rich Panel

    Parameters:
    -----------
    cm_dict : dict
        Dictionary with keys: tn, fp, fn, tp
    console : Console
        Rich console object
    """
    from rich.panel import Panel

    tn, fp, fn, tp = cm_dict['tn'], cm_dict['fp'], cm_dict['fn'], cm_dict['tp']
    total = tn + fp + fn + tp
    actual_positives = tp + fn
    flagged = tp + fp

    # Calculate rates
    catch_rate = 100 * tp / actual_positives if actual_positives > 0 else 0
    miss_rate = 100 * fn / actual_positives if actual_positives > 0 else 0
    hit_rate = 100 * tp / flagged if flagged > 0 else 0
    false_alarm_rate = 100 * fp / flagged if flagged > 0 else 0
    nns = flagged / tp if tp > 0 else float('inf')

    # Calculate additional clinical metrics
    specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = 100 * tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = hit_rate  # Same as precision

    content = f"""[bold]Patient Cohort:[/bold]
  Total Patients:              {total:,}
  Actual Readmissions:         {actual_positives:,} ({100*actual_positives/total:.1f}%)
  Non-Readmissions:            {tn+fp:,} ({100*(tn+fp)/total:.1f}%)

[bold green]Detection Performance:[/bold green]
  Readmissions CAUGHT:         {tp:,} ({catch_rate:.1f}%) ✓
  Readmissions MISSED:         {fn:,} ({miss_rate:.1f}%) ✗

[bold yellow]Intervention Load:[/bold yellow]
  Patients Flagged:            {flagged:,} ({100*flagged/total:.1f}%)
    └─ True Positives:         {tp:,} ({hit_rate:.1f}% hit rate)
    └─ False Alarms:           {fp:,} ({false_alarm_rate:.1f}%)

[bold]Clinical Efficiency:[/bold]
  Number Needed to Screen:     {nns:.1f}
  (Intervene on {nns:.0f} patients to catch 1 readmission)

[bold]Additional Metrics:[/bold]
  Sensitivity (Recall):        {catch_rate:.1f}%
  Specificity:                 {specificity:.1f}%
  PPV (Precision):             {ppv:.1f}%
  NPV:                         {npv:.1f}%
"""

    panel = Panel(content, title="Clinical Impact Summary", border_style="blue", expand=False)
    console.print("\n")
    console.print(panel)


def display_transfer_gap_comparison(
    cv_auroc: float,
    test_auroc: float,
    test_recall: float,
    console
) -> None:
    """
    Display performance comparison showing transfer gap

    Parameters:
    -----------
    cv_auroc : float
        Cross-validation AUROC (on synthetic/training data)
    test_auroc : float
        Test AUROC (on real data)
    test_recall : float
        Test recall percentage
    console : Console
        Rich console object
    """
    from rich.table import Table
    from rich.panel import Panel

    transfer_gap = cv_auroc - test_auroc

    # Determine gap severity
    if transfer_gap < 0.05:
        gap_status = "[green]Excellent[/green]"
        gap_icon = "✓"
    elif transfer_gap < 0.15:
        gap_status = "[yellow]Acceptable[/yellow]"
        gap_icon = "⚠️"
    elif transfer_gap < 0.25:
        gap_status = "[orange1]Concerning[/orange1]"
        gap_icon = "⚠️"
    else:
        gap_status = "[red]Poor Transfer[/red]"
        gap_icon = "❌"

    table = Table(show_header=True, header_style="bold cyan", title="Performance Comparison: Synthetic vs Real Data")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Training/CV", justify="right", width=15)
    table.add_column("Test (Real)", justify="right", width=15)
    table.add_column("Difference", justify="right", width=15)

    # AUROC row
    diff_auroc = test_auroc - cv_auroc
    table.add_row(
        "AUROC",
        f"{cv_auroc:.4f}",
        f"{test_auroc:.4f}",
        f"[red]{diff_auroc:+.4f}[/red]" if diff_auroc < 0 else f"[green]{diff_auroc:+.4f}[/green]"
    )

    # Recall row (only test available)
    table.add_row(
        "Recall",
        "N/A",
        f"{test_recall:.1f}%",
        "-"
    )

    console.print("\n")
    console.print(table)

    # Transfer gap panel
    gap_content = f"""[bold]Transfer Gap:[/bold] {transfer_gap:.4f} (CV AUROC - Test AUROC)

[bold]Status:[/bold] {gap_status} {gap_icon}

[bold]Interpretation:[/bold]
"""

    if transfer_gap < 0.05:
        gap_content += "  Model patterns transfer excellently to real data.\n  Predictions are reliable and deployment-ready."
    elif transfer_gap < 0.15:
        gap_content += "  Model shows acceptable transfer to real data.\n  Some synthetic-specific patterns, but generally reliable."
    elif transfer_gap < 0.25:
        gap_content += "  Concerning transfer gap. Model may have learned\n  synthetic-specific patterns. Use with caution."
    else:
        gap_content += "  [bold red]Poor transfer![/bold red] Model learned unrealistic patterns\n  from synthetic data. Not recommended for deployment.\n  High CV performance is misleading."

    gap_panel = Panel(gap_content, title="Transfer Gap Analysis", border_style="yellow", expand=False)
    console.print(gap_panel)


def save_model_and_metrics(
    model: lgb.Booster,
    metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    output_dir: Path,
    label_encoders: Optional[Dict[str, Any]] = None,
    prefix: str = "lgbm_readmission"
) -> Tuple[Path, Path]:
    """
    Save trained model and metrics to disk

    Parameters:
    -----------
    model : lgb.Booster
        Trained LightGBM model
    metrics : dict
        Evaluation metrics
    best_params : dict
        Best hyperparameters found during tuning
    output_dir : Path
        Output directory
    label_encoders : dict, optional
        Dictionary of label encoders for categorical features
    prefix : str
        Prefix for output files

    Returns:
    --------
    tuple : (model_path, metrics_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model along with label encoders
    model_path = output_dir / f"{prefix}_{timestamp}.pkl"
    model_package = {
        'model': model,
        'label_encoders': label_encoders
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)

    # Prepare metrics output
    # Determine encoding type and features
    if label_encoders is not None:
        # Check if it's RDTDatasetEncoder or dict of LabelEncoders
        if hasattr(label_encoders, 'sdtypes'):
            # RDT encoder
            encoding_type = 'RDT'
            encoded_features = list(label_encoders.sdtypes.keys())
        else:
            # Simple LabelEncoder dict
            encoding_type = 'LabelEncoder'
            encoded_features = list(label_encoders.keys()) if isinstance(label_encoders, dict) else []
    else:
        encoding_type = 'None'
        encoded_features = []

    metrics_output = {
        'timestamp': timestamp,
        'test_metrics': metrics,
        'best_hyperparameters': best_params,
        'model_info': {
            'num_trees': model.num_trees(),
            'best_iteration': model.best_iteration
        },
        'preprocessing': {
            'encoding_type': encoding_type,
            'encoded_features': encoded_features
        }
    }

    # Save metrics
    metrics_path = output_dir / f"{prefix}_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)

    return model_path, metrics_path


def train_readmission_model(
    train_file: Path,
    test_file: Path,
    target_column: str = "IS_READMISSION_30D",
    n_trials: int = 100,
    n_folds: int = 5,
    timeout: Optional[int] = None,
    random_state: int = 42,
    output_dir: Path = Path("experiments/models/downstream"),
    val_split: float = 0.2,
    encoding_config: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Main function to train LGBM readmission prediction model

    Parameters:
    -----------
    train_file : Path
        Path to training data CSV
    test_file : Path
        Path to test data CSV
    target_column : str
        Name of the target column (default: "IS_READMISSION_30D")
    n_trials : int
        Number of Bayesian optimization trials
    n_folds : int
        Number of cross-validation folds
    timeout : int, optional
        Maximum time in seconds for optimization
    random_state : int
        Random seed
    output_dir : Path
        Output directory for models and metrics
    val_split : float
        Fraction of training data to use for validation in final model
    encoding_config : Path, optional
        Path to RDT encoding config YAML (same format as SDG pipeline)
        If provided, uses RDT encoding. If None, uses simple LabelEncoder.

    Returns:
    --------
    dict : Results summary including paths and metrics
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

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

    # Encode categorical columns
    rdt_encoder = None
    label_encoders = {}

    if encoding_config is not None:
        # Use RDT encoding (consistent with SDG pipeline)
        console.print(f"\n[bold cyan]Using RDT encoding from config...[/bold cyan]")
        console.print(f"  Config: {encoding_config}")

        from sdpype.encoding import load_encoding_config, RDTDatasetEncoder

        # Load encoding config
        enc_config = load_encoding_config(encoding_config)

        # Filter out target column from encoding config (it's not a feature)
        if target_column in enc_config['sdtypes']:
            enc_config['sdtypes'] = {k: v for k, v in enc_config['sdtypes'].items() if k != target_column}
        if target_column in enc_config['transformers']:
            enc_config['transformers'] = {k: v for k, v in enc_config['transformers'].items() if k != target_column}

        # Create and fit encoder on training data
        rdt_encoder = RDTDatasetEncoder(enc_config)
        rdt_encoder.fit(X_train)

        # Transform both datasets
        X_train = rdt_encoder.transform(X_train)
        X_test = rdt_encoder.transform(X_test)

        console.print(f"  ✓ Encoded {X_train.shape[1]} features using RDT transformers")

    else:
        # Fall back to simple LabelEncoder for categorical columns
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

        if categorical_cols:
            console.print(f"\n[bold cyan]Encoding categorical features (LabelEncoder)...[/bold cyan]")

            for col in categorical_cols:
                le = LabelEncoder()
                # Fit on training data
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                # Transform test data (handle unseen categories)
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                label_encoders[col] = le
                console.print(f"  ✓ {col}: {len(le.classes_)} categories")

    # Display data info
    console.print(f"\n[bold cyan]Data Summary:[/bold cyan]")
    console.print(f"  Training samples: {len(X_train):,}")
    console.print(f"  Test samples: {len(X_test):,}")
    console.print(f"  Features: {X_train.shape[1]}")
    console.print(f"  Target: {target_column}")
    console.print(f"  Class distribution (train): {dict(y_train.value_counts())}")

    # Initialize tuner
    console.print(f"\n[bold cyan]Starting Bayesian Optimization[/bold cyan]")
    console.print(f"  Trials: {n_trials}")
    console.print(f"  Folds: {n_folds}")
    console.print(f"  Timeout: {timeout if timeout else 'None'}")

    tuner = LGBMBayesianTuner(
        X_train=X_train,
        y_train=y_train,
        n_folds=n_folds,
        n_trials=n_trials,
        random_state=random_state
    )

    # Run tuning with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running hyperparameter optimization...", total=None)
        best_params = tuner.tune(timeout=timeout)

    console.print(f"\n[bold green]✓ Optimization completed![/bold green]")
    console.print(f"  Best CV AUROC: {tuner.best_score:.4f}")

    console.print(f"\n[bold cyan]Best Hyperparameters:[/bold cyan]")
    for param, value in best_params.items():
        console.print(f"  {param}: {value}")

    # Train final model with validation split
    console.print(f"\n[bold cyan]Training final model...[/bold cyan]")

    # Split training data for final model validation
    from sklearn.model_selection import train_test_split
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train,
        test_size=val_split,
        random_state=random_state,
        stratify=y_train
    )

    final_model = tuner.train_final_model(
        X_train_final, y_train_final,
        X_val_final, y_val_final
    )

    console.print(f"[bold green]✓ Model trained![/bold green]")
    console.print(f"  Best iteration: {final_model.best_iteration}")
    console.print(f"  Total trees: {final_model.num_trees()}")

    # Learn optimal threshold on validation data
    console.print(f"\n[bold cyan]Finding optimal threshold...[/bold cyan]")
    optimal_threshold = tuner.find_optimal_threshold(
        final_model, X_val_final, y_val_final, metric='f1'
    )
    console.print(f"[bold green]✓ Optimal threshold found: {optimal_threshold:.3f}[/bold green]")
    console.print(f"  (Optimized for F1 score on validation set)")

    # Evaluate on test set
    console.print(f"\n[bold cyan]Evaluating on test set...[/bold cyan]")
    test_metrics = evaluate_model(final_model, X_test, y_test, threshold=optimal_threshold)

    console.print(f"\n[bold green]Test Set Performance:[/bold green]")
    console.print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    console.print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    console.print(f"  Precision: {test_metrics['precision']:.4f}")
    console.print(f"  Recall:    {test_metrics['recall']:.4f}")
    console.print(f"  F1 Score:  {test_metrics['f1_score']:.4f}")

    # Display confusion matrix
    display_confusion_matrix(test_metrics['confusion_matrix'], console)

    # Display clinical performance summary
    display_clinical_performance(test_metrics['confusion_matrix'], console)

    # Display transfer gap comparison (shows how well synthetic patterns transfer to real data)
    display_transfer_gap_comparison(
        cv_auroc=tuner.best_score,
        test_auroc=test_metrics['auroc'],
        test_recall=test_metrics['recall'] * 100,
        console=console
    )

    # Save model and metrics
    console.print(f"\n[bold cyan]Saving model and metrics...[/bold cyan]")

    # Package encoder (either RDT or simple label encoders)
    encoder_package = rdt_encoder if rdt_encoder is not None else label_encoders

    # Add optimal threshold to best params for saving
    best_params_with_threshold = best_params.copy()
    best_params_with_threshold['optimal_threshold'] = float(optimal_threshold)

    model_path, metrics_path = save_model_and_metrics(
        model=final_model,
        metrics=test_metrics,
        best_params=best_params_with_threshold,
        output_dir=output_dir,
        label_encoders=encoder_package,
        prefix="lgbm_readmission"
    )

    console.print(f"[bold green]✓ Saved successfully![/bold green]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Metrics: {metrics_path}")

    # Return results
    return {
        'model_path': str(model_path),
        'metrics_path': str(metrics_path),
        'test_metrics': test_metrics,
        'best_params': best_params,
        'cv_score': tuner.best_score
    }
