#!/usr/bin/env python3
"""
Standalone test script for experimenting with Synthpop CART preprocessing.

This script tests different preprocessing approaches to find what works
with Synthpop CART given the sklearn version incompatibility.

Usage:
    python test_synthpop_cart.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from synthpop.method import CARTMethod
from synthpop import DataProcessor, MissingDataHandler

# Configuration
DATA_FILE = "experiments/data/raw/Covid19Cases.csv"
N_SAMPLES = 100
RANDOM_STATE = 42


def load_sample_data(n_rows=1000):
    """Load a sample of the data for quick testing."""
    print(f"📥 Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, nrows=n_rows)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n📊 Data types:")
    print(df.dtypes)
    print(f"\n📋 Sample data:")
    print(df.head())
    return df


def get_metadata(df):
    """Get metadata automatically."""
    print("\n🔍 Detecting metadata...")
    metadata = MissingDataHandler.get_column_dtypes(df)
    print(f"   Metadata: {metadata}")
    return metadata


def test_raw_data():
    """Test 1: Raw data without preprocessing (we know this fails)."""
    print("\n" + "="*70)
    print("TEST 1: Raw data without preprocessing")
    print("="*70)

    df = load_sample_data()
    metadata = get_metadata(df)

    try:
        cart = CARTMethod(metadata, random_state=RANDOM_STATE)
        print("✓ CARTMethod created")

        print("⏳ Fitting on raw data...")
        cart.fit(df)
        print("✓ Fit successful!")

        print(f"⏳ Generating {N_SAMPLES} samples...")
        synthetic = cart.sample(N_SAMPLES)
        print(f"✓ Generated {len(synthetic)} samples with {len(synthetic.columns)} columns")
        print(f"   Columns: {list(synthetic.columns)}")

        return True, synthetic
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        return False, None


def test_dataprocessor_preprocessing():
    """Test 2: Synthpop's DataProcessor (we know this has sklearn issue)."""
    print("\n" + "="*70)
    print("TEST 2: Synthpop DataProcessor preprocessing")
    print("="*70)

    df = load_sample_data()
    metadata = get_metadata(df)

    try:
        print("⏳ Creating DataProcessor...")
        processor = DataProcessor(metadata)
        print("✓ DataProcessor created")

        print("⏳ Preprocessing data...")
        processed_data = processor.preprocess(df)
        print("✓ Preprocessing successful!")
        print(f"   Processed shape: {processed_data.shape}")
        print(f"   Processed dtypes:\n{processed_data.dtypes}")
        print(f"\n   Sample processed data:")
        print(processed_data.head())

        print("\n⏳ Creating CARTMethod...")
        cart = CARTMethod(metadata, random_state=RANDOM_STATE)
        print("✓ CARTMethod created")

        print("⏳ Fitting on preprocessed data...")
        cart.fit(processed_data)
        print("✓ Fit successful!")

        print(f"⏳ Generating {N_SAMPLES} samples...")
        synthetic_processed = cart.sample(N_SAMPLES)
        print(f"✓ Generated {len(synthetic_processed)} samples")
        print(f"   Shape: {synthetic_processed.shape}")

        print("\n⏳ Postprocessing to original format...")
        synthetic = processor.postprocess(synthetic_processed)
        print(f"✓ Postprocessing successful!")
        print(f"   Final shape: {synthetic.shape}")
        print(f"   Final columns: {list(synthetic.columns)}")

        return True, synthetic
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_manual_encoding():
    """Test 3: Manual label encoding for categorical columns."""
    print("\n" + "="*70)
    print("TEST 3: Manual label encoding (sklearn-safe)")
    print("="*70)

    df = load_sample_data()
    metadata = get_metadata(df)

    try:
        from sklearn.preprocessing import LabelEncoder

        print("⏳ Manually encoding categorical columns...")
        df_encoded = df.copy()
        encoders = {}

        for col, dtype in metadata.items():
            if dtype in ['categorical', 'boolean']:
                print(f"   Encoding {col} ({dtype})...")
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            elif dtype == 'datetime':
                print(f"   Converting {col} to timestamp...")
                df_encoded[col] = pd.to_datetime(df[col]).astype(np.int64) / 10**9

        print(f"\n✓ Encoded shape: {df_encoded.shape}")
        print(f"   Encoded dtypes:\n{df_encoded.dtypes}")
        print(f"\n   Sample encoded data:")
        print(df_encoded.head())

        # Update metadata to numerical
        encoded_metadata = {col: 'numerical' for col in df_encoded.columns}

        print("\n⏳ Creating CARTMethod with numerical metadata...")
        cart = CARTMethod(encoded_metadata, random_state=RANDOM_STATE)
        print("✓ CARTMethod created")

        print("⏳ Fitting on encoded data...")
        cart.fit(df_encoded)
        print("✓ Fit successful!")

        print(f"⏳ Generating {N_SAMPLES} samples...")
        synthetic_encoded = cart.sample(N_SAMPLES)
        print(f"✓ Generated {len(synthetic_encoded)} samples")
        print(f"   Shape: {synthetic_encoded.shape}")
        print(f"   Columns: {list(synthetic_encoded.columns)}")

        # Decode back
        print("\n⏳ Decoding back to original format...")
        synthetic = synthetic_encoded.copy()
        for col, encoder in encoders.items():
            print(f"   Decoding {col}...")
            # Round to integers and clip to valid range
            values = synthetic_encoded[col].round().clip(0, len(encoder.classes_) - 1).astype(int)
            synthetic[col] = encoder.inverse_transform(values)

        print(f"✓ Decoding successful!")
        print(f"   Final columns: {list(synthetic.columns)}")
        print(f"\n   Sample synthetic data:")
        print(synthetic.head())

        return True, synthetic
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_pandas_getdummies():
    """Test 4: Use pandas get_dummies for encoding."""
    print("\n" + "="*70)
    print("TEST 4: Pandas get_dummies encoding")
    print("="*70)

    df = load_sample_data()
    metadata = get_metadata(df)

    try:
        print("⏳ Encoding with pd.get_dummies...")
        df_encoded = df.copy()

        # Identify categorical columns
        cat_cols = [col for col, dtype in metadata.items() if dtype in ['categorical', 'boolean']]
        print(f"   Categorical columns: {cat_cols}")

        # Convert datetime
        for col, dtype in metadata.items():
            if dtype == 'datetime':
                print(f"   Converting {col} to timestamp...")
                df_encoded[col] = pd.to_datetime(df[col]).astype(np.int64) / 10**9

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

        print(f"\n✓ Encoded shape: {df_encoded.shape}")
        print(f"   Encoded columns ({len(df_encoded.columns)}): {list(df_encoded.columns)[:10]}...")
        print(f"\n   Sample encoded data:")
        print(df_encoded.head())

        # All numerical now
        encoded_metadata = {col: 'numerical' for col in df_encoded.columns}

        print("\n⏳ Creating CARTMethod...")
        cart = CARTMethod(encoded_metadata, random_state=RANDOM_STATE)
        print("✓ CARTMethod created")

        print("⏳ Fitting on encoded data...")
        cart.fit(df_encoded)
        print("✓ Fit successful!")

        print(f"⏳ Generating {N_SAMPLES} samples...")
        synthetic = cart.sample(N_SAMPLES)
        print(f"✓ Generated {len(synthetic)} samples")
        print(f"   Shape: {synthetic.shape}")
        print(f"   Columns ({len(synthetic.columns)}): {list(synthetic.columns)[:10]}...")

        return True, synthetic
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all tests."""
    print("🧪 Synthpop CART Preprocessing Experiments")
    print("=" * 70)
    print(f"Data file: {DATA_FILE}")
    print(f"Sample size: {N_SAMPLES}")
    print(f"Random state: {RANDOM_STATE}")

    results = {}

    # Run all tests
    results['raw_data'] = test_raw_data()
    results['dataprocessor'] = test_dataprocessor_preprocessing()
    results['manual_encoding'] = test_manual_encoding()
    results['pandas_dummies'] = test_pandas_getdummies()

    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)

    for test_name, (success, synthetic) in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        if success and synthetic is not None:
            print(f"                      Generated {len(synthetic)} rows × {len(synthetic.columns)} columns")

    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)

    if results['dataprocessor'][0]:
        print("✓ Synthpop DataProcessor works! Use the official workflow.")
    elif results['manual_encoding'][0]:
        print("✓ Manual label encoding works! Use sklearn LabelEncoder.")
    elif results['pandas_dummies'][0]:
        print("✓ Pandas get_dummies works! Use one-hot encoding.")
    else:
        print("❌ No preprocessing approach worked.")
        print("   Synthpop CART may be incompatible with this dataset/sklearn version.")


if __name__ == "__main__":
    main()
