#!/bin/bash
set -euo pipefail

# FIX #6: Get script directory and navigate to code/ (relative path fix)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""
# List of experiments
experiments=(
    "baseline:events_baseline_1M.npz"
    "distinct:events_distinct_1M.npz"
    "cadence_05:events_cadence_05_1M.npz"
    "cadence_40:events_cadence_40_1M.npz"
    "error_low:events_error_low_1M.npz"
    "error_high:events_error_high_1M.npz"
)

for exp in "${experiments[@]}"; do
    IFS=: read -r name data <<< "$exp"
    
    echo ""
    echo "="*60
    echo "EVALUATING: $name"
    echo "="*60
    
    # Find latest results directory
    # Note: Using the name from the loop to match the result directory structure
    RESULTS_DIR=$(ls -td ../results/${name}_* 2>/dev/null | head -1)
    
    if [ -z "$RESULTS_DIR" ]; then
        echo "⚠️  No results found for $name"
        continue
    fi
    
    # The expected model path should be in the directory found by the glob
    if [ ! -f "$RESULTS_DIR/best_model.pt" ]; then
        echo "⚠️  No trained model found for $name at $RESULTS_DIR/best_model.pt"
        continue
    fi
    
    echo "Results dir: $RESULTS_DIR"
    echo "Data: ../data/raw/$data"
    
    # Evaluate
    python evaluate.py \
        --model "$RESULTS_DIR/best_model.pt" \
        --data "../data/raw/$data" \
        --output_dir "$RESULTS_DIR/evaluation" \
        --batch_size 128 \
        --early_detection
    
    echo "✓ $name evaluation complete"
    
    # Show metrics (checking for the fixed evaluation_summary.json)
    if [ -f "$RESULTS_DIR/evaluation/evaluation_summary.json" ]; then
        echo ""
        echo "Metrics (evaluation_summary.json):"
        cat "$RESULTS_DIR/evaluation/evaluation_summary.json" | python -m json.tool | grep -E 'accuracy|roc_auc|tn|tp' | sed 's/"metrics": {//; s/"tn":/    "tn":/'
    else
        echo "⚠️  evaluation_summary.json not found to display metrics."
    fi
done

echo ""
echo "="*60
echo "ALL EVALUATIONS COMPLETE"
echo "="*60