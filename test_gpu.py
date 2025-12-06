#!/usr/bin/env python3
"""Test GPU compatibility with PyTorch."""

import torch
import sys

print("=" * 60)
print("PyTorch GPU Compatibility Test")
print("=" * 60)

# Basic PyTorch info
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(i)}")

        # Get memory info
        props = torch.cuda.get_device_properties(i)
        print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Multi-processor count: {props.multi_processor_count}")

    # Test tensor operations
    print("\n" + "=" * 60)
    print("Testing CUDA tensor operations...")
    print("=" * 60)

    try:
        # Simple test
        print("\n1. Creating tensor on CPU...")
        x = torch.randn(100, 100)
        print("   ✓ Success")

        print("2. Moving tensor to GPU...")
        x_gpu = x.cuda()
        print("   ✓ Success")

        print("3. Performing matrix multiplication on GPU...")
        y_gpu = torch.matmul(x_gpu, x_gpu)
        print("   ✓ Success")

        print("4. Moving result back to CPU...")
        y_cpu = y_gpu.cpu()
        print("   ✓ Success")

        print("\n✅ All basic GPU operations successful!")

    except RuntimeError as e:
        print(f"\n❌ Error during GPU operations: {e}")
        print("\nThis error suggests the GPU architecture is not supported by this PyTorch build.")
        sys.exit(1)

else:
    print("\n❌ CUDA is not available")
    sys.exit(1)

print("\n" + "=" * 60)
