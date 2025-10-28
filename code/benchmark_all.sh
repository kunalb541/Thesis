#!/bin/bash
set -euo pipefail

cd ~/Thesis/code

# List of experiments (name:datafile)
experiments=(
    "baseline:events_baseline_1M.npz"
    "distinct:events_distinct_1M.npz"
    "cadence_05:events_cadence_05_1M.npz"
    "cadence_40:events_cadence_40_1M.npz"
    "error_low:events_error_low_1M.npz"
    "error_high:events_error_high_1M.npz"
)

echo ""
echo "="*60
echo "BENCHMARKING REAL-TIME CAPABILITY"
echo "="*60

for exp in "${experiments[@]}"; do
    IFS=: read -r name data <<< "$exp"

    echo ""
    echo "-"*60
    echo "EXPERIMENT: $name"
    echo "Data: ../data/raw/$data"
    echo "-"*60

    # Find ALL result directories for this experiment (newest first)
    mapfile -t RUN_DIRS < <(ls -td ../results/${name}_* 2>/dev/null || true)

    if [ ${#RUN_DIRS[@]} -eq 0 ]; then
        echo "⚠️  No results found for $name"
        continue
    fi

    for RESULTS_DIR in "${RUN_DIRS[@]}"; do
        echo ""
        echo "→ Run: $RESULTS_DIR"

        MODEL_PATH="$RESULTS_DIR/best_model.pt"
        if [ ! -f "$MODEL_PATH" ]; then
            echo "   ⚠️  No trained model found at $MODEL_PATH — skipping"
            continue
        fi

        OUT_DIR="$RESULTS_DIR/benchmark"
        mkdir -p "$OUT_DIR"

        # Run benchmark
        python benchmark_realtime.py \
            --model "$MODEL_PATH" \
            --data "../data/raw/$data" \
            --output_dir "$OUT_DIR"

        echo "   ✓ Benchmark complete"

        # Pretty-print results if present
        if [ -f "$OUT_DIR/benchmark_results.json" ]; then
            echo "   Results:"
            cat "$OUT_DIR/benchmark_results.json" | python -m json.tool
        else
            echo "   ⚠️  benchmark_results.json not found in $OUT_DIR"
        fi
    done
done

echo ""
echo "="*60
echo "ALL BENCHMARKS COMPLETE"
echo "="*60
