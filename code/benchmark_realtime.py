#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_realtime.py - Real-time inference performance benchmarking

Measures:
- Inference latency (ms per event)
- Throughput (events/second)
- GPU memory usage
- Batch processing efficiency

Author: Kunal Bhatia
Version: 1.0
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm

from model import TimeDistributedCNN
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
    """
    Benchmark inference performance
    
    Args:
        model: Trained model
        X: Test data
        device: torch device
        batch_size: Batch size for inference
        n_warmup: Number of warmup iterations
        n_runs: Number of benchmark iterations
    
    Returns:
        dict: Benchmark results
    """
    model.eval()
    
    # Prepare data
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    # Create random batches for benchmarking
    n_samples = len(X)
    indices = np.random.choice(n_samples, size=n_runs * batch_size, replace=True)
    
    # Warmup
    print(f"\nWarming up ({n_warmup} iterations)...")
    for i in range(n_warmup):
        idx = np.random.choice(n_samples, size=batch_size, replace=False)
        X_batch = torch.from_numpy(X_processed[idx]).float().unsqueeze(1).to(device)
        with torch.no_grad():
            _ = model(X_batch)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({n_runs} iterations, batch_size={batch_size})...")
    latencies = []
    
    for i in tqdm(range(n_runs), desc="Benchmarking"):
        idx = indices[i*batch_size:(i+1)*batch_size]
        X_batch = torch.from_numpy(X_processed[idx]).float().unsqueeze(1).to(device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(X_batch)
        
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
            'min': float(latencies.min() / batch_size),
            'max': float(latencies.max() / batch_size),
        },
        'throughput_events_per_sec': {
            'mean': float((batch_size * 1000) / latencies.mean()),
            'max': float((batch_size * 1000) / latencies.min()),
        },
    }
    
    return results


def measure_memory(model, X, device, batch_size=128):
    """
    Measure GPU memory usage
    
    Args:
        model: Trained model
        X: Test data
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        dict: Memory usage statistics
    """
    if not torch.cuda.is_available():
        return {
            'device': 'CPU',
            'note': 'Memory measurement only available for CUDA'
        }
    
    model.eval()
    
    # Prepare data
    X_processed = X.copy()
    X_processed[X_processed == CFG.PAD_VALUE] = 0.0
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Measure baseline memory
    baseline_memory = torch.cuda.memory_allocated(device)
    
    # Run inference
    idx = np.random.choice(len(X), size=batch_size, replace=False)
    X_batch = torch.from_numpy(X_processed[idx]).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        _ = model(X_batch)
    
    torch.cuda.synchronize()
    
    # Get memory stats
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


def test_batch_sizes(model, X, device, batch_sizes=[1, 8, 16, 32, 64, 128, 256, 512]):
    """
    Test performance across different batch sizes
    
    Args:
        model: Trained model
        X: Test data
        device: torch device
        batch_sizes: List of batch sizes to test
    
    Returns:
        dict: Results for each batch size
    """
    results = {}
    
    print("\n" + "="*80)
    print("BATCH SIZE SCALING TEST")
    print("="*80)
    
    for bs in batch_sizes:
        if bs > len(X):
            print(f"\nSkipping batch_size={bs} (larger than dataset size {len(X)})")
            continue
        
        print(f"\nTesting batch_size={bs}...")
        
        try:
            bench_results = benchmark_inference(
                model, X, device, 
                batch_size=bs, 
                n_warmup=5, 
                n_runs=50
            )
            
            results[bs] = bench_results
            
            # Print summary
            latency_per_event = bench_results['latency_per_event_ms']['mean']
            throughput = bench_results['throughput_events_per_sec']['mean']
            
            print(f"  Latency per event: {latency_per_event:.3f} ms")
            print(f"  Throughput: {throughput:.0f} events/sec")
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  ✗ OOM - batch_size={bs} too large")
                torch.cuda.empty_cache()
            else:
                raise
    
    return results


def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    if isinstance(results, dict) and 'batch_size' in results:
        # Single batch size result
        results = {results['batch_size']: results}
    
    # Header
    print(f"\n{'Batch Size':<12} {'Latency/Event (ms)':<20} {'Throughput (ev/s)':<20} {'Total Latency (ms)':<20}")
    print("-" * 80)
    
    # Sort by batch size
    for bs in sorted(results.keys()):
        res = results[bs]
        latency_per_event = res['latency_per_event_ms']['mean']
        throughput = res['throughput_events_per_sec']['mean']
        total_latency = res['latency_ms']['mean']
        
        print(f"{bs:<12} {latency_per_event:<20.3f} {throughput:<20.0f} {total_latency:<20.3f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark real-time inference performance')
    parser.add_argument("--model", type=str, default=None, help='Path to model checkpoint (auto-detect if not provided)')
    parser.add_argument("--data", type=str, required=True, help='Path to test data')
    parser.add_argument("--output_dir", type=str, default=None, help='Output directory (auto-detect if not provided)')
    parser.add_argument("--experiment_name", type=str, default=None, help='Experiment name (for auto-detect)')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size for benchmark')
    parser.add_argument("--n_runs", type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument("--test_batch_scaling", action='store_true', help='Test multiple batch sizes')
    parser.add_argument("--n_samples", type=int, default=10000, help='Number of samples to use for benchmark')
    args = parser.parse_args()
    
    # Auto-detect model and output_dir if not provided
    if args.model is None or args.output_dir is None:
        if args.experiment_name is None:
            raise ValueError("Must provide either --model and --output_dir, OR --experiment_name for auto-detection")
        
        results_dir = find_latest_results_dir(args.experiment_name)
        print(f"✓ Auto-detected results directory: {results_dir}")
        
        if args.model is None:
            args.model = str(results_dir / "best_model.pt")
        if args.output_dir is None:
            args.output_dir = str(results_dir / "benchmark")
    else:
        results_dir = Path(args.model).parent
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("REAL-TIME INFERENCE BENCHMARK")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    
    # =========================================================================
    # Load data with saved scalers
    # =========================================================================
    print("\n" + "="*80)
    print("LOADING DATA AND MODEL")
    print("="*80)
    
    # Load RAW data
    print("\n1. Loading RAW data (normalize=False)...")
    X, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True, normalize=False)
    L = X.shape[1]
    print(f"✓ Raw data loaded: {X.shape}")
    
    # Subsample for faster benchmarking
    if len(X) > args.n_samples:
        print(f"   Subsampling {args.n_samples} events for benchmark...")
        indices = np.random.choice(len(X), args.n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Load saved scalers
    print("\n2. Loading scalers from training...")
    scaler_std, scaler_mm = load_scalers(results_dir)
    print(f"✓ Loaded scalers from {results_dir}")
    
    # Apply scalers
    print("\n3. Applying saved scalers to data...")
    X = apply_scalers_to_data(X, scaler_std, scaler_mm, pad_value=CFG.PAD_VALUE)
    print(f"✓ Applied same normalization as training")
    # =========================================================================
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n4. Loading model...")
    print(f"   Device: {device}")
    
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2)
    
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # =========================================================================
    # Run benchmarks
    # =========================================================================
    
    if args.test_batch_scaling:
        # Test multiple batch sizes
        batch_sizes = [1, 8, 16, 32, 64, 128, 256]
        if args.batch_size not in batch_sizes:
            batch_sizes.append(args.batch_size)
            batch_sizes.sort()
        
        results = test_batch_sizes(model, X, device, batch_sizes=batch_sizes)
        
        # Print summary
        print_results_table(results)
        
    else:
        # Single batch size benchmark
        print("\n" + "="*80)
        print(f"BENCHMARKING (batch_size={args.batch_size})")
        print("="*80)
        
        results = benchmark_inference(
            model, X, device,
            batch_size=args.batch_size,
            n_warmup=10,
            n_runs=args.n_runs
        )
        
        print_results_table(results)
    
    # =========================================================================
    # Memory measurement
    # =========================================================================
    print("\n" + "="*80)
    print("MEMORY USAGE")
    print("="*80)
    
    memory_results = measure_memory(model, X, device, batch_size=args.batch_size)
    
    if 'device' in memory_results:
        if memory_results['device'] == 'CPU':
            print(f"\nDevice: CPU (memory measurement not available)")
        else:
            print(f"\nDevice: {memory_results['device']}")
            print(f"  Baseline memory: {memory_results['baseline_mb']:.1f} MB")
            print(f"  Peak memory: {memory_results['peak_mb']:.1f} MB")
            print(f"  Inference overhead: {memory_results['inference_mb']:.1f} MB")
            print(f"  Batch size: {memory_results['batch_size']}")
    
    # =========================================================================
    # Save results
    # =========================================================================
    benchmark_summary = {
        'model_path': str(args.model),
        'data_path': str(args.data),
        'device': str(device),
        'n_samples': len(X),
        'sequence_length': L,
        'n_parameters': n_params,
        'benchmark': results if not args.test_batch_scaling else {
            'batch_size': args.batch_size,
            'results': results.get(args.batch_size, {})
        },
        'batch_scaling': results if args.test_batch_scaling else None,
        'memory': memory_results,
        'config': {
            'batch_size': args.batch_size,
            'n_runs': args.n_runs,
        }
    }
    
    output_path = output_dir / 'benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(benchmark_summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    
    if not args.test_batch_scaling:
        latency_per_event = results['latency_per_event_ms']['mean']
        throughput = results['throughput_events_per_sec']['mean']
        
        print(f"\n🚀 Performance Summary (batch_size={args.batch_size}):")
        print(f"   Latency per event: {latency_per_event:.3f} ms")
        print(f"   Throughput: {throughput:.0f} events/sec")
        
        # Context
        lsst_alerts_per_night = 10000
        time_to_process = (lsst_alerts_per_night / throughput) / 60  # minutes
        
        print(f"\n📊 Survey Context:")
        print(f"   LSST alerts/night: {lsst_alerts_per_night:,}")
        print(f"   Time to process: {time_to_process:.1f} minutes")
        
        if time_to_process < 60:
            print(f"   ✅ Real-time capable for LSST operations!")
        else:
            print(f"   ⚠️  May need optimization for real-time LSST operations")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()