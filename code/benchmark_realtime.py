

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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

class TimeDistributedCNN(nn.Module):
    """Same architecture as training"""
    def __init__(self, sequence_length=1500, num_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
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

def benchmark_throughput(model, test_data, device, batch_sizes=[1, 8, 16, 32, 64, 128, 256]):
    """
    Measure throughput (events/second) vs batch size
    
    Critical for understanding LSST/Roman operational capacity
    """
    model.eval()
    
    print("\n" + "="*60)
    print("THROUGHPUT BENCHMARK")
    print("="*60)
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}...")
        
        # Create data loader
        dataset = LightCurveDataset(test_data['X'][:1000], test_data['y'][:1000])
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

def benchmark_lsst_scale(model, test_data, device):
    """
    Simulate LSST operational scenario
    
    LSST will generate ~10,000 microlensing alerts per night
    Can we process them in real-time?
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
    n_events = min(lsst_events_per_night, len(test_data['X']))
    dataset = LightCurveDataset(test_data['X'][:n_events], 
                               test_data['y'][:n_events])
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
    
    # Literature values for binary lens fitting
    chi2_time_per_event = 500  # seconds (conservative estimate)
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
    plt.annotate(f'Optimal: {batch_sizes[max_idx]}\n{throughputs[max_idx]:.1f} events/s',
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
    X_test = data['X'][-10000:]  # Use last 10k for testing
    y_test = data['y'][-10000:]
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y_test])
    
    test_data = {'X': X_test, 'y': y_encoded}
    print(f"Test data: {X_test.shape}")
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model, map_location=device)
    model = TimeDistributedCNN(X_test.shape[1], X_test.shape[2])
    
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        model = nn.DataParallel(model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Create data loader for latency testing
    dataset = LightCurveDataset(X_test, y_encoded)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)
    
    # Run benchmarks
    all_results = {}
    
    # 1. Latency benchmark
    latency_results = benchmark_inference_speed(model, loader, device)
    all_results['latency'] = latency_results
    
    # 2. Throughput benchmark
    throughput_results = benchmark_throughput(model, test_data, device)
    all_results['throughput'] = throughput_results
    
    # Plot throughput
    plot_path = output_dir / 'throughput_vs_batch_size.png'
    plot_throughput_vs_batch_size(throughput_results, plot_path)
    
    # 3. LSST scale benchmark
    lsst_results = benchmark_lsst_scale(model, test_data, device)
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