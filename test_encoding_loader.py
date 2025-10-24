"""
Quick test script to verify encoding config loading works.
"""

import logging
from pathlib import Path
from sdpype.encoding import load_encoding_config, TRANSFORMER_REGISTRY

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_load_config():
    """Test loading the example COVID config"""
    config_path = Path('experiments/configs/encoding/example_covid.yaml')

    print(f"\n{'='*70}")
    print(f"Testing encoding config loader")
    print(f"{'='*70}\n")

    print(f"Config file: {config_path}")
    print(f"Exists: {config_path.exists()}\n")

    # Load config
    config = load_encoding_config(config_path)

    # Display results
    print(f"\n{'='*70}")
    print(f"LOADED CONFIG")
    print(f"{'='*70}\n")

    print(f"SDTypes ({len(config['sdtypes'])} columns):")
    for col_name, sdtype in config['sdtypes'].items():
        print(f"  • {col_name:30s} → {sdtype}")

    print(f"\nTransformers ({len(config['transformers'])} configured):")
    for col_name, transformer in config['transformers'].items():
        transformer_type = type(transformer).__name__
        print(f"  • {col_name:30s} → {transformer_type}")
        # Show params if it's OrderedUniformEncoder (has order param)
        if hasattr(transformer, 'order') and transformer.order is not None:
            print(f"      order: {list(transformer.order)}")

    print(f"\n{'='*70}")
    print(f"Available Transformer Types in Registry:")
    print(f"{'='*70}")
    for name in TRANSFORMER_REGISTRY.keys():
        print(f"  • {name}")

    print(f"\n{'='*70}")
    print(f"✓ Config loaded successfully!")
    print(f"{'='*70}\n")

    return config

if __name__ == '__main__':
    test_load_config()
