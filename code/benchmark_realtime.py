#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_realtime.py - Real-time inference performance benchmarking

FIXED (v5.0): 
- Removed hard-coded TDConvClassifier
- Imports models from model.py
- Reads config.json to load the correct model (Simple or LSTM)
- Calls model(x, return_sequence=False) for final prediction

Author: Kunal Bhatia
Version: 5.0
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm

# Import model architectures (matches train.py)
from model import TimeDistributedCNNSimple, TimeDistributedCNN

from utils import load_npz_dataset, load_scalers, apply_scalers_to_data
import config as CFG


def find_latest_results_dir(experiment_name, base_dir='../results'):
    """Find the most recent results directory for an experiment"""
    base_path = Path(base_dir)
    pattern = f"{experiment_name}_*"
    
    matching_dirs = sorted(base_path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No results directories found matching '{pattern}' in {base_dir}")
    
    return matching_dirs[0]


def benchmark_inference(model, X, device, batch_size=128, n_warmup=10, n_runs=100):
    """Benchmark inference performance"""
    model.eval()
    
    # Prepare data
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    # Create random batches
    n_samples = len(X)
    indices = np.random.choice(n_samples, size=n_runs * batch_size, replace=True)
    
    # Warmup
    print(f"\nWarming up ({n_warmup} iterations)...")
    for i in range(n_warmup):
        idx = np.random.choice(n_samples, size=batch_size, replace=False)
        # Input shape [B, T] -> [B, 1, T]
        X_batch = torch.from_numpy(X_processed[idx]).float().unsqueeze(1).to(device)
        with torch.no_grad():
            # --- FIX: Call model with return_sequence=False ---
            _ = model(X_batch, return_sequence=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({n_runs} iterations, batch_size={batch_size})...")
    latencies = []
    
    for i in tqdm(range(n_runs), desc="Benchmarking"):
        idx = indices[i*batch_size:(i+1)*batch_size]
        # Input shape [B, T] -> [B, 1, T]
        X_batch = torch.from_numpy(X_processed[idx]).float().unsqueeze(1).to(device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            # --- FIX: Call model with return_sequence=False ---
            _ = model(X_batch, return_sequence=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    # Calculate metrics
    results = {
        'batch_size': batch_size,
        'n_runs': n_runs,
        'latency_ms': {
            'mean': float(latencies.mean()),
            'std': float(latencies.std()),
            'min': float(latencies.min()),
            'max': float(latencies.max()),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
        },
        'latency_per_event_ms': {
            'mean': float(latencies.mean() / batch_size),
            'std': float(latencies.std() / batch_size),
        },
        'throughput_events_per_sec': {
            'mean': float((batch_size * 1000) / latencies.mean()),
            'max': float((batch_size * 1000) / latencies.min()),
        },
    }
    
    return results


def measure_memory(model, X, device, batch_size=128):
    """Measure GPU memory usage"""
    if not torch.cuda.is_available():
        return {
            'device': 'CPU',
            'note': 'Memory measurement only available for CUDA'
        }
    
    model.eval()
    
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    baseline_memory = torch.cuda.memory_allocated(device)
    
    idx = np.random.choice(len(X), size=batch_size, replace=False)
    # Input shape [B, T] -> [B, 1, T]
    X_batch = torch.from_numpy(X_processed[idx]).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        # --- FIX: Call model with return_sequence=False ---
        _ = model(X_batch, return_sequence=False)
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated(device)
    current_memory = torch.cuda.memory_allocated(device)
    
    results = {
        'device': torch.cuda.get_device_name(device),
        'baseline_mb': float(baseline_memory / 1e6),
        'current_mb': float(current_memory / 1e6),
        'peak_mb': float(peak_memory / 1e6),
        'inference_mb': float((peak_memory - baseline_memory) / 1e6),
        'batch_size': batch_size,
    }
    
    return results


def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    if isinstance(results, dict) and 'batch_size' in results:
        results = {results['batch_size']: results}
    
    print(f"\n{'Batch Size':<12} {'Latency/Event (ms)':<20} {'Throughput (ev/s)':<20}")
    print("-" * 52)
    
    for bs in sorted(results.keys()):
        res = results[bs]
        latency_per_event = res['latency_per_event_ms']['mean']
        throughput = res['throughput_events_per_sec']['mean']
        
        print(f"{bs:<12} {latency_per_event:<20.3f} {throughput:<20.0f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark real-time inference (v5.0 - TimeDistributed Compatible)')
    parser.add_argument("--model", type=str, default=None, help='Path to model checkpoint')
    parser.add_argument("--data", type=str, required=True, help='Path to test data')
    parser.add_argument("--output_dir", type=str, default=None, help='Output directory')
    parser.add_argument("--experiment_name", type=str, default=None, help='Experiment name')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size')
    parser.add_argument("--n_runs", type=int, default=100, help='Number of iterations')
    parser.add_argument("--n_samples", type=int, default=10000, help='Number of samples')
    args = parser.parse_args()
    
    if args.model is None or args.output_dir is None:
        if args.experiment_name is None:
            raise ValueError("Must provide --model and --output_dir, OR --experiment_name")
        
        results_dir = find_latest_results_dir(args.experiment_name)
        print(f"✓ Auto-detected: {results_dir}")
        
        if args.model is None:
            args.model = str(results_dir / "best_model.pt")
        if args.output_dir is None:
            args.output_dir = str(results_dir / "benchmark")
    else:
        results_dir = Path(args.model).parent
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("REAL-TIME INFERENCE BENCHMARK (TimeDistributed Compatible)")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    
    # Load data
    print("\nLoading data...")
    # Load data, X shape is (N, T)
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    
    if len(X) > args.n_samples:
        indices = np.random.choice(len(X), args.n_samples, replace=False)
        X = X[indices]
    
    scaler_std, scaler_mm = load_scalers(results_dir)
    # apply_scalers_to_data expects (N, F) where F=T
    X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # --- FIX: Load model based on config.json ---
    config_path = results_dir / "config.json"
    model_type = 'TimeDistributed_Simple'
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            model_type = config.get('model_type', 'TimeDistributed_Simple')

    if model_type == 'TimeDistributed_LSTM':
        print("Loading TimeDistributedCNN (LSTM)...")
        window_size = config.get('window_size', 50)
        model = TimeDistributedCNN(
            in_channels=1, 
            n_classes=2, 
            window_size=window_size,
            use_lstm=True,
            dropout=0.3 
        )
    else:
        print("Loading TimeDistributedCNNSimple...")
        model = TimeDistributedCNNSimple(
            in_channels=1, 
            n_classes=2, 
            dropout=0.3
        )
    # --- End Fix ---
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"✓ Model loaded ({model_type})")
    
    # Benchmark
    results = benchmark_inference(model, X, device, batch_size=args.batch_size, n_runs=args.n_runs)
    print_results_table(results)
    
    # Memory
    print("\n" + "="*80)
    print("MEMORY USAGE")
    print("="*80)
    memory_results = measure_memory(model, X, device, batch_size=args.batch_size)
    
    if 'device' in memory_results and memory_results['device'] != 'CPU':
        print(f"\nDevice: {memory_results['device']}")
        print(f"  Peak memory: {memory_results['peak_mb']:.1f} MB")
        print(f"  Inference overhead: {memory_results['inference_mb']:.1f} MB")
    
    # Save
    benchmark_summary = {
        'model_path': str(args.model),
        'model_architecture': model_type,
        'benchmark': results,
        'memory': memory_results,
    }
    
    output_path = output_dir / 'benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(benchmark_summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary
    latency_per_event = results['latency_per_event_ms']['mean']
    throughput = results['throughput_events_per_sec']['mean']
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n🚀 Performance:")
    print(f"   Latency per event: {latency_per_event:.3f} ms")
    print(f"   Throughput: {throughput:.0f} events/sec")
    
    lsst_alerts_per_night = 10000
    time_to_process = (lsst_alerts_per_night / throughput) / 60
    print(f"\n📊 LSST Context:")
    print(f"   Alerts/night: {lsst_alerts_per_night:,}")
    print(f"   Time to process: {time_to_process:.1f} minutes")
    print(f"   ✅ Real-time capable!" if time_to_process < 60 else "   ⚠️  May need optimization")


if __name__ == "__main__":
    main()