#!/bin/bash

cd ~/Thesis/code

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
    RESULTS_DIR=$(ls -td ../results/${name}_* 2>/dev/null | head -1)
    
    if [ -z "$RESULTS_DIR" ]; then
        echo "⚠️  No results found for $name"
        continue
    fi
    
    if [ ! -f "$RESULTS_DIR/best_model.pt" ]; then
        echo "⚠️  No trained model found for $name"
        continue
    fi
    
    echo "Results dir: $RESULTS_DIR"
    echo "Data: ../data/raw/$data"
    
    # Evaluate
    python evaluate.py \
        --model "$RESULTS_DIR/best_model.pt" \
        --data "../data/raw/$data" \
        --scaler "$RESULTS_DIR/scaler.pkl" \
        --output_dir "$RESULTS_DIR/evaluation" \
        --batch_size 128 \
        --early_detection
    
    echo "✓ $name evaluation complete"
    
    # Show metrics
    if [ -f "$RESULTS_DIR/evaluation/metrics.json" ]; then
        echo ""
        echo "Metrics:"
        cat "$RESULTS_DIR/evaluation/metrics.json" | python -m json.tool
    fi
done

echo ""
echo "="*60
echo "ALL EVALUATIONS COMPLETE"
echo "="*60
