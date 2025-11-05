#!/usr/bin/env python3
"""
Real-Time Performance Benchmark

Tests inference speed and throughput to validate real-time capability
for LSST/Roman alert stream processing.

Target: <1 ms per event, 10,000+ events/sec

Author: Kunal Bhatia
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from pathlib import Path
import json
import pickle

from model import TransformerClassifier
from utils import load_npz_dataset, load_scalers, apply_scalers_to_data


def benchmark_inference(model, X, device, n_runs=5, batch_sizes=[1, 8, 32, 128, 512]):
    """
    Benchmark inference speed across different batch sizes
    
    Returns:
        results: Dict with timing statistics
    """
    model.eval()
    results = {}
    
    print("\n" + "="*80)
    print("INFERENCE SPEED BENCHMARK")
    print("="*80)
    print(f"Device: {device}")
    print(f"Data shape: {X.shape}")
    print(f"Number of runs per batch size: {n_runs}")
    
    for batch_size in batch_sizes:
        if batch_size > len(X):
            continue
            
        print(f"\n{'='*40}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*40}")
        
        # Prepare batch
        X_batch = X[:batch_size]
        X_tensor = torch.from_numpy(X_batch).float().to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(X_tensor, return_sequence=False)
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            
            with torch.no_grad():
                logits, _ = model(X_tensor, return_sequence=False)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        # Statistics
        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()
        min_time = times.min()
        
        time_per_event = mean_time / batch_size
        events_per_sec = batch_size / mean_time
        
        print(f"  Total time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Min time: {min_time*1000:.2f} ms")
        print(f"  Time per event: {time_per_event*1000:.3f} ms")
        print(f"  Throughput: {events_per_sec:.1f} events/sec")
        
        # Check real-time capability
        if time_per_event < 0.001:  # < 1 ms
            status = "✅ REAL-TIME CAPABLE"
        elif time_per_event < 0.01:  # < 10 ms
            status = "✅ NEAR REAL-TIME"
        else:
            status = "⚠️  TOO SLOW FOR REAL-TIME"
        
        print(f"  Status: {status}")
        
        results[batch_size] = {
            'mean_time_ms': float(mean_time * 1000),
            'std_time_ms': float(std_time * 1000),
            'min_time_ms': float(min_time * 1000),
            'time_per_event_ms': float(time_per_event * 1000),
            'throughput_per_sec': float(events_per_sec),
            'real_time_capable': time_per_event < 0.001
        }
    
    return results


def benchmark_memory(model, X, device, batch_sizes=[1, 8, 32, 128, 512]):
    """
    Benchmark memory usage
    """
    if device.type != 'cuda':
        print("\n⚠️  Memory benchmark requires CUDA")
        return {}
    
    print("\n" + "="*80)
    print("MEMORY USAGE BENCHMARK")
    print("="*80)
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(X):
            continue
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # Measure
        X_batch = torch.from_numpy(X[:batch_size]).float().to(device)
        
        with torch.no_grad():
            _ = model(X_batch, return_sequence=False)
        
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        
        memory_per_event = peak_memory / batch_size * 1024  # MB
        
        print(f"  Batch {batch_size:3d}: {peak_memory:.3f} GB total, "
              f"{memory_per_event:.2f} MB per event")
        
        results[batch_size] = {
            'peak_memory_gb': float(peak_memory),
            'memory_per_event_mb': float(memory_per_event)
        }
    
    return results


def estimate_survey_throughput(throughput_per_sec, survey='LSST'):
    """
    Estimate if system can handle survey alert rates
    """
    print("\n" + "="*80)
    print(f"SURVEY CAPABILITY ASSESSMENT - {survey}")
    print("="*80)
    
    if survey == 'LSST':
        alerts_per_night = 10_000
        alerts_per_hour = alerts_per_night / 10  # Assume 10-hour night
        alerts_per_sec = alerts_per_hour / 3600
        
        processing_capacity = throughput_per_sec
        capacity_ratio = processing_capacity / alerts_per_sec
        
        print(f"  LSST Alert Rate: ~{alerts_per_night:,} alerts/night")
        print(f"  Peak rate: ~{alerts_per_sec:.2f} alerts/sec")
        print(f"  Your throughput: {throughput_per_sec:.1f} events/sec")
        print(f"  Capacity ratio: {capacity_ratio:.1f}x")
        
        if capacity_ratio > 100:
            print(f"  ✅ EXCELLENT: Can process {capacity_ratio:.0f}x LSST rate!")
        elif capacity_ratio > 10:
            print(f"  ✅ GOOD: Can handle LSST + {capacity_ratio:.0f}x overhead")
        elif capacity_ratio > 1:
            print(f"  ✅ ADEQUATE: Can keep up with LSST")
        else:
            print(f"  ❌ INSUFFICIENT: Only {capacity_ratio:.1%} of required speed")
    
    elif survey == 'Roman':
        alerts_per_year = 20_000
        alerts_per_day = alerts_per_year / 365
        alerts_per_sec = alerts_per_day / 86400
        
        processing_capacity = throughput_per_sec
        capacity_ratio = processing_capacity / alerts_per_sec
        
        print(f"  Roman Alert Rate: ~{alerts_per_year:,} alerts/year")
        print(f"  Average rate: ~{alerts_per_sec:.4f} alerts/sec")
        print(f"  Your throughput: {throughput_per_sec:.1f} events/sec")
        print(f"  Capacity ratio: {capacity_ratio:.0f}x")
        print(f"  ✅ Roman rate is easily manageable")


def compare_to_traditional_fitting():
    """
    Compare to traditional PSPL fitting speeds
    """
    print("\n" + "="*80)
    print("COMPARISON TO TRADITIONAL FITTING")
    print("="*80)
    
    # Typical PSPL fitting times from literature
    fitting_time_per_event = 300  # seconds (5 minutes)
    fitting_events_per_sec = 1 / fitting_time_per_event
    
    print(f"  Traditional PSPL Fitting:")
    print(f"    Time per event: ~{fitting_time_per_event} seconds")
    print(f"    Throughput: ~{fitting_events_per_sec:.4f} events/sec")
    print(f"    Time for 10,000 events: ~{fitting_time_per_event * 10000 / 3600:.1f} hours")
    
    return fitting_events_per_sec


def main():
    parser = argparse.ArgumentParser(description="Benchmark real-time performance")
    parser.add_argument("--experiment_name", required=True, help="Experiment to benchmark")
    parser.add_argument("--data", required=True, help="Test data path")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to test")
    parser.add_argument("--n_runs", type=int, default=5, help="Runs per batch size")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("REAL-TIME PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Device: {device}")
    
    # Find latest experiment
    results_dir = Path("../results")
    matching = sorted(results_dir.glob(f"{args.experiment_name}_*"))
    if not matching:
        raise ValueError(f"No experiments found: {args.experiment_name}")
    
    exp_dir = matching[-1]
    print(f"Experiment: {exp_dir}")
    
    # Load config
    with open(exp_dir / "config.json") as f:
        config = json.load(f)
    
    # Load model
    print("\nLoading model...")
    model = TransformerClassifier(
        in_channels=1,
        n_classes=2,
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 256),
        downsample_factor=config.get('downsample_factor', 3),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    checkpoint = torch.load(exp_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    X_raw, y, timestamps, meta = load_npz_dataset(args.data, apply_perm=True)
    
    if X_raw.ndim == 2:
        X_raw = X_raw[:, None, :]
    
    # Normalize
    scaler_std, scaler_mm = load_scalers(exp_dir)
    X = apply_scalers_to_data(X_raw, scaler_std, scaler_mm, pad_value=-1.0)
    
    # Subsample
    X = X[:args.n_samples]
    y = y[:args.n_samples]
    
    print(f"Data shape: {X.shape}")
    
    # Benchmark inference
    batch_sizes = [1, 8, 32, 128, 512]
    inference_results = benchmark_inference(model, X, device, args.n_runs, batch_sizes)
    
    # Benchmark memory
    if device.type == 'cuda':
        memory_results = benchmark_memory(model, X, device, batch_sizes)
    else:
        memory_results = {}
    
    # Find optimal batch size
    optimal_batch = max(inference_results.keys(), 
                       key=lambda k: inference_results[k]['throughput_per_sec'])
    optimal_throughput = inference_results[optimal_batch]['throughput_per_sec']
    
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION")
    print("="*80)
    print(f"  Optimal batch size: {optimal_batch}")
    print(f"  Max throughput: {optimal_throughput:.1f} events/sec")
    print(f"  Time per event: {inference_results[optimal_batch]['time_per_event_ms']:.3f} ms")
    
    # Survey assessments
    estimate_survey_throughput(optimal_throughput, 'LSST')
    estimate_survey_throughput(optimal_throughput, 'Roman')
    
    # Compare to traditional
    fitting_speed = compare_to_traditional_fitting()
    speedup = optimal_throughput / fitting_speed
    
    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS")
    print("="*80)
    print(f"  ML speedup: ~{speedup:.0f}x faster than traditional fitting")
    print(f"  Processing 10,000 events:")
    print(f"    Traditional: ~{10000 * 300 / 3600:.1f} hours")
    print(f"    This system: ~{10000 / optimal_throughput / 60:.1f} minutes")
    
    # Save results
    benchmark_results = {
        'device': str(device),
        'model_config': config,
        'test_samples': args.n_samples,
        'inference_results': inference_results,
        'memory_results': memory_results,
        'optimal_batch_size': int(optimal_batch),
        'optimal_throughput': float(optimal_throughput),
        'speedup_vs_fitting': float(speedup),
        'real_time_capable': inference_results[optimal_batch]['time_per_event_ms'] < 1.0
    }
    
    output_path = exp_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\n✅ Benchmark results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
    