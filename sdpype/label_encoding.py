"""
Simple label encoding for Synthpop CART compatibility.

This module provides a lightweight label encoder specifically for Synthpop CART,
which requires numeric data but has sklearn version incompatibilities with its
native DataProcessor.

The encoder:
- Uses sklearn LabelEncoder (modern, compatible)
- Converts categorical columns to integers
- Stores encoders for reversible transformation
- Separate from RDT encoding (which is used for metrics)
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class SimpleLabelEncoder:
    """
    Simple label encoder for Synthpop CART.

    Converts categorical columns to integers using sklearn LabelEncoder,
    avoiding the sklearn compatibility issues in Synthpop's DataProcessor.

    Usage:
        # Training
        encoder = SimpleLabelEncoder(metadata)
        encoded_data = encoder.fit_transform(training_data)
        encoder.save("encoder.pkl")

        # Generation
        encoder = SimpleLabelEncoder.load("encoder.pkl")
        decoded_data = encoder.inverse_transform(generated_data)
    """

    def __init__(self, metadata: Dict[str, str]):
        """
        Initialize encoder with metadata.

        Args:
            metadata: Dict mapping column names to data types
                     (e.g., {'col1': 'categorical', 'col2': 'numerical'})
        """
        self.metadata = metadata
        self.encoders: Dict[str, LabelEncoder] = {}
        self.categorical_columns = [
            col for col, dtype in metadata.items()
            if dtype in ['categorical', 'boolean']
        ]
        self._is_fitted = False

        logger.info(f"Initialized SimpleLabelEncoder")
        logger.info(f"  Categorical columns: {len(self.categorical_columns)}")
        logger.info(f"  Columns: {self.categorical_columns}")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoders and transform data to integers.

        Args:
            data: Original DataFrame with categorical columns

        Returns:
            DataFrame with categorical columns encoded as integers
        """
        logger.info("Fitting and transforming data...")
        encoded_data = data.copy()

        for col in self.categorical_columns:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found in data, skipping")
                continue

            logger.info(f"  Encoding {col}...")
            encoder = LabelEncoder()
            # Convert to string to handle mixed types
            encoded_data[col] = encoder.fit_transform(data[col].astype(str))
            self.encoders[col] = encoder

            logger.info(f"    → {len(encoder.classes_)} unique values encoded")

        self._is_fitted = True
        logger.info(f"✓ Encoding complete: {len(self.encoders)} columns encoded")

        return encoded_data

    def inverse_transform(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """
        Decode integer-encoded data back to original categories.

        Args:
            encoded_data: DataFrame with integer-encoded categorical columns

        Returns:
            DataFrame with original categorical values
        """
        if not self._is_fitted:
            raise ValueError("Encoder not fitted. Call fit_transform first or load from file.")

        logger.info("Decoding data back to original format...")
        decoded_data = encoded_data.copy()

        for col, encoder in self.encoders.items():
            if col not in encoded_data.columns:
                logger.warning(f"Column '{col}' not found in encoded data, skipping")
                continue

            logger.info(f"  Decoding {col}...")

            # Convert to float, round, and clip to valid range
            # This handles cases where CART generates fractional values
            values = (
                encoded_data[col]
                .astype(float)  # Ensure numeric
                .round()        # Round to nearest integer
                .clip(0, len(encoder.classes_) - 1)  # Clip to valid range
                .astype(int)    # Convert to int for inverse_transform
            )

            decoded_data[col] = encoder.inverse_transform(values)
            logger.info(f"    → Decoded {len(values)} values")

        logger.info(f"✓ Decoding complete: {len(self.encoders)} columns decoded")

        return decoded_data

    def save(self, filepath: Path):
        """
        Save encoder to pickle file.

        Args:
            filepath: Path to save encoder
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'metadata': self.metadata,
            'encoders': self.encoders,
            'categorical_columns': self.categorical_columns,
            'is_fitted': self._is_fitted,
            'version': '1.0'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"✓ Saved SimpleLabelEncoder to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'SimpleLabelEncoder':
        """
        Load encoder from pickle file.

        Args:
            filepath: Path to saved encoder

        Returns:
            Loaded SimpleLabelEncoder instance
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Create instance
        instance = cls(save_data['metadata'])
        instance.encoders = save_data['encoders']
        instance.categorical_columns = save_data['categorical_columns']
        instance._is_fitted = save_data['is_fitted']

        logger.info(f"✓ Loaded SimpleLabelEncoder from: {filepath}")
        logger.info(f"  Categorical columns: {len(instance.categorical_columns)}")

        return instance

    def get_encoded_metadata(self) -> Dict[str, str]:
        """
        Get metadata for encoded data (all columns marked as 'numerical').

        This is needed for Synthpop CART which requires numerical metadata
        to avoid sklearn classification errors.

        Returns:
            Dict with all columns marked as 'numerical'
        """
        encoded_metadata = {}
        for col, dtype in self.metadata.items():
            # Mark categorical columns as numerical (they're now integers)
            if col in self.categorical_columns:
                encoded_metadata[col] = 'numerical'
            else:
                # Keep original type for truly numerical columns
                encoded_metadata[col] = dtype

        return encoded_metadata
