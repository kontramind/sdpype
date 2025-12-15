#!/bin/bash
# Quick test script for post-training metrics computation

set -e

echo "========================================"
echo "Post-Training Metrics Computation Test"
echo "========================================"
echo ""

# Check if folder exists
FOLDER="./mimic_iii_baseline_dseed233_sdv_gaussiancopula_mseed24157817"
if [ ! -d "$FOLDER" ]; then
    echo "ERROR: Experiment folder not found: $FOLDER"
    echo "Please update this script with the correct path to your experiment folder"
    exit 1
fi

echo "Testing on folder: $FOLDER"
echo ""

# Test 1: Compute statistical metrics for generation 0
echo "Test 1: Computing statistical metrics for generation 0..."
python compute_metrics_cli.py \
    --folder "$FOLDER" \
    --generation 0 \
    --metrics statistical

echo ""
echo "✓ Test 1 complete"
echo ""

# Test 2: Compute detection metrics for generation 0
echo "Test 2: Computing detection metrics for generation 0..."
python compute_metrics_cli.py \
    --folder "$FOLDER" \
    --generation 0 \
    --metrics detection

echo ""
echo "✓ Test 2 complete"
echo ""

# Test 3: Compute hallucination metrics for generation 0
echo "Test 3: Computing hallucination metrics for generation 0..."
python compute_metrics_cli.py \
    --folder "$FOLDER" \
    --generation 0 \
    --metrics hallucination

echo ""
echo "✓ Test 3 complete"
echo ""

# Show results
echo "Results saved in:"
ls -lh "$FOLDER/metrics/"*_gen_0_*.json | head -5

echo ""
echo "========================================"
echo "✓ All tests passed!"
echo "========================================"
