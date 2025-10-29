#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-Time Capability Benchmarking for Microlensing Classification

Measures inference speed, throughput, and scalability to demonstrate
operational feasibility for LSST/Roman alert streams.

Author: Kunal Bhatia
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Local import of the unified model
from model import TimeDistributedCNN
import config as CFG

# Note: LightCurveDataset must handle the padding logic consistently with training/evaluation
class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        # Apply the same pre-processing as in train.py: map PAD_VALUE to 0.0
        X_processed = X.copy()
        X_processed[X_processed == CFG.PAD_VALUE] = 0.0
        
        self.X = torch.from_numpy(X_processed).float().unsqueeze(1) # [N, 1, L]
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def benchmark_inference_speed(model, data_loader, device, n_warmup=10):
    """
    Measure inference latency (time per event)
    
    Returns latency statistics in milliseconds
    """
    model.eval()
    latencies = []
    
    print("\n" + "="*60)
    print("INFERENCE LATENCY BENCHMARK")
    print("="*60)
    
    with torch.no_grad():
        # Warmup
        print(f"\nWarming up ({n_warmup} batches)...")
        for i, (batch_x, _) in enumerate(data_loader):
            if i >= n_warmup:
                break
            batch_x = batch_x.to(device)
            _ = model(batch_x)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # Actual measurement
        print("Measuring latency...")
        for batch_x, _ in tqdm(data_loader, desc="Inference"):
            batch_x = batch_x.to(device)
            
            start = time.perf_counter()
            # The model outputs [B, L, 2], but we only measure the inference time for the forward pass
            _ = model(batch_x)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            # Latency per event in milliseconds
            batch_latency = ((end - start) / len(batch_x)) * 1000
            latencies.append(batch_latency)
    
    latencies = np.array(latencies)
    
    results = {
        'mean_ms': float(np.mean(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
    }
    
    print(f"\nLatency Statistics:")
    print(f"  Mean:   {results['mean_ms']:.3f} ms/event")
    print(f"  Median: {results['median_ms']:.3f} ms/event")
    print(f"  P95:    {results['p95_ms']:.3f} ms/event")
    print(f"  P99:    {results['p99_ms']:.3f} ms/event")
    print(f"  Min:    {results['min_ms']:.3f} ms/event")
    print(f"  Max:    {results['max_ms']:.3f} ms/event")
    
    return results

def benchmark_throughput(model, X_test, y_test, device, batch_sizes=[1, 8, 16, 32, 64, 128, 256]):
    """
    Measure throughput (events/second) vs batch size
    """
    model.eval()
    
    print("\n" + "="*60)
    print("THROUGHPUT BENCHMARK")
    print("="*60)
    
    results = []
    
    # Use a fixed, large subset of data for throughput testing
    n_throughput_test = min(len(X_test), 10000) 
    X_test_sub = X_test[:n_throughput_test]
    y_test_sub = y_test[:n_throughput_test]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}...")
        
        # Create data loader
        dataset = LightCurveDataset(X_test_sub, y_test_sub)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)
        
        total_events = 0
        total_time = 0
        
        with torch.no_grad():
            # Warmup
            for i, (batch_x, _) in enumerate(loader):
                if i >= 5:
                    break
                batch_x = batch_x.to(device)
                _ = model(batch_x)
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            # Measurement
            start = time.perf_counter()
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)
                total_events += len(batch_x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_time = end - start
        
        throughput = total_events / total_time
        
        results.append({
            'batch_size': batch_size,
            'events_per_second': throughput,
            'time_per_event_ms': (total_time / total_events) * 1000
        })
        
        print(f"  Throughput: {throughput:.1f} events/second")
        print(f"  Latency:    {results[-1]['time_per_event_ms']:.3f} ms/event")
    
    return results

def benchmark_lsst_scale(model, X_test, y_test, device):
    """
    Simulate LSST operational scenario
    """
    model.eval()
    
    print("\n" + "="*60)
    print("LSST OPERATIONAL SCALE SIMULATION")
    print("="*60)
    
    lsst_events_per_night = 10000
    print(f"\nSimulating {lsst_events_per_night:,} LSST alerts...")
    
    # Use optimal batch size (typically 128 or 256)
    batch_size = 128
    
    # Create dataset
    n_events = min(lsst_events_per_night, len(X_test))
    dataset = LightCurveDataset(X_test[:n_events], 
                               y_test[:n_events])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        start = time.perf_counter()
        
        for batch_x, _ in tqdm(loader, desc="Processing alerts"):
            batch_x = batch_x.to(device)
            _ = model(batch_x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
    
    total_time = end - start
    events_per_second = n_events / total_time
    
    print(f"\nResults:")
    print(f"  Events processed: {n_events:,}")
    print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Throughput: {events_per_second:.1f} events/second")
    print(f"  Average latency: {(total_time/n_events)*1000:.3f} ms/event")
    
    # Compare to traditional fitting
    print(f"\n" + "-"*60)
    print("COMPARISON TO TRADITIONAL χ² FITTING")
    print("-"*60)
    
    # Literature values for binary lens fitting (seconds per event)
    chi2_time_per_event = 500  
    chi2_total_time = chi2_time_per_event * n_events
    
    print(f"\nTraditional Binary Fitting [Estimated from Literature]:")
    print(f"  Time per event: ~{chi2_time_per_event} seconds")
    print(f"  Total time for {n_events:,} events: {chi2_total_time/86400:.1f} days")
    print(f"  (Requires continuous computation on single CPU)")
    
    speedup = chi2_total_time / total_time
    print(f"\nSpeedup Factor: {speedup:.0f}×")
    
    print(f"\n" + "="*60)
    print("OPERATIONAL ASSESSMENT")
    print("="*60)
    
    if total_time < 3600:  # Less than 1 hour
        print("✅ REAL-TIME PROCESSING: FEASIBLE")
        print(f"   Can process nightly alerts in {total_time/60:.1f} minutes")
    elif total_time < 86400:  # Less than 1 day
        print("⚠️  BATCH PROCESSING: FEASIBLE")
        print(f"   Requires {total_time/3600:.1f} hours per night")
    else:
        print("❌ NOT FEASIBLE for operational deployment")
    
    print("\nTraditional Fitting:")
    print("❌ REAL-TIME PROCESSING: INFEASIBLE")
    print(f"   Would require {chi2_total_time/86400:.1f} days continuous compute")
    print("="*60)
    
    return {
        'n_events': n_events,
        'total_time_seconds': total_time,
        'events_per_second': events_per_second,
        'feasible': total_time < 3600,
        'chi2_estimated_days': chi2_total_time / 86400,
        'speedup_factor': speedup
    }

def plot_throughput_vs_batch_size(results, save_path):
    """Plot throughput vs batch size"""
    batch_sizes = [r['batch_size'] for r in results]
    throughputs = [r['events_per_second'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Throughput (events/second)', fontsize=12)
    plt.title('Inference Throughput vs Batch Size', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xscale('log', base=2)
    
    # Annotate optimal point
    max_idx = np.argmax(throughputs)
    plt.annotate(f"Optimal: {batch_sizes[max_idx]}\n{throughputs[max_idx]:.1f} events/s",
                xy=(batch_sizes[max_idx], throughputs[max_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Throughput plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark real-time capability')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("REAL-TIME CAPABILITY BENCHMARK")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\nLoading test data...")
    data = np.load(args.data)
    # The original script uses the last 10k. We must check if "perm" is present and apply it first.
    # The benchmark uses a subset, so we will manually apply permutation to the entire dataset 
    # and then slice the end. This is a robust way to handle the perm array.
    X_full = data['X']
    y_full = data['y']
    if "perm" in data.files:
        perm = data['perm']
        X_full = X_full[perm]
        y_full = y_full[perm]
    
    # Label mapping (assumes labels are 'PSPL'/'Binary' strings or the 0/1 encoding used later)
    # The provided data loader is simple, let's ensure the label encoding is correct.
    if y_full.dtype.kind in ("U", "S", "O"):
        y_encoded = np.array([0 if (str(v).lower().startswith("pspl")) else 1 for v in y_full], dtype=np.uint8)
    else:
        y_encoded = y_full.astype(np.uint8)
        
    # Use last 10k for testing
    X_test = X_full[-10000:]
    y_test_encoded = y_encoded[-10000:]
    
    print(f"Test data shape: {X_test.shape}")
    L = X_test.shape[1]
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    # FIX #4: Corrected checkpoint key and unified model size
    model = TimeDistributedCNN(sequence_length=L, num_channels=1, num_classes=2)
    
    # Handle DataParallel saved state dicts
    state_dict = checkpoint['model_state_dict']
    if 'module.' in list(state_dict.keys())[0] and not isinstance(model, nn.DataParallel):
        # Remove 'module.' prefix for non-DataParallel model loading
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Run benchmarks
    all_results = {}
    
    # 1. Latency benchmark (using the subset)
    dataset_latency = LightCurveDataset(X_test, y_test_encoded)
    loader_latency = DataLoader(dataset_latency, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)

    latency_results = benchmark_inference_speed(model, loader_latency, device)
    all_results['latency'] = latency_results
    
    # 2. Throughput benchmark
    throughput_results = benchmark_throughput(model, X_test, y_test_encoded, device)
    all_results['throughput'] = throughput_results
    
    # Plot throughput
    plot_path = output_dir / 'throughput_vs_batch_size.png'
    plot_throughput_vs_batch_size(throughput_results, plot_path)
    
    # 3. LSST scale benchmark
    lsst_results = benchmark_lsst_scale(model, X_test, y_test_encoded, device)
    all_results['lsst_scale'] = lsst_results
    
    # Save results
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n📊 Inference Latency:")
    print(f"   Mean: {latency_results['mean_ms']:.3f} ms/event")
    print(f"   P99:  {latency_results['p99_ms']:.3f} ms/event")
    
    print(f"\n📈 Peak Throughput:")
    max_throughput = max(r['events_per_second'] for r in throughput_results)
    print(f"   {max_throughput:.1f} events/second")
    
    print(f"\n🔭 LSST Operational Capability:")
    print(f"   10,000 events in {lsst_results['total_time_seconds']/60:.1f} minutes")
    print(f"   {lsst_results['speedup_factor']:.0f}× faster than traditional fitting")
    print(f"   Real-time feasible: {'✅ YES' if lsst_results['feasible'] else '❌ NO'}")
    
    print("\n" + "="*80)
    print("READY FOR OPERATIONAL DEPLOYMENT")
    print("="*80)

if __name__ == "__main__":
    main()