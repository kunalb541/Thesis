#!/usr/bin/env python3
"""
Real-Time Streaming Inference for Binary Microlensing Detection

Production-ready pipeline for processing telescope alert streams.
Maintains circular buffer and triggers alerts on detection.

Author: Kunal Bhatia
Version: 6.0
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import deque
import threading
import queue

from streaming_transformer import StreamingTransformer
from normalization import CausticPreservingNormalizer
import config as CFG


class StreamingPipeline:
    """
    Real-time inference pipeline for microlensing detection.
    
    Features:
    - Circular buffer for streaming observations
    - Per-observation inference with <1ms latency
    - Confidence tracking over time
    - Alert triggering system
    - Thread-safe for production deployment
    """
    
    def __init__(
        self,
        model_path: str,
        normalizer_path: str,
        buffer_size: int = CFG.BUFFER_SIZE,
        confidence_threshold: float = CFG.CONFIDENCE_THRESHOLD,
        caustic_threshold: float = CFG.CAUSTIC_THRESHOLD,
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            normalizer_path: Path to fitted normalizer
            buffer_size: Size of circular buffer
            confidence_threshold: Threshold for binary detection
            caustic_threshold: Threshold for caustic alert
            device: Compute device
        """
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.caustic_threshold = caustic_threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = StreamingTransformer().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle DDP checkpoint
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix from DDP
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load normalizer
        print(f"Loading normalizer from {normalizer_path}...")
        self.normalizer = CausticPreservingNormalizer()
        self.normalizer.load(normalizer_path)
        
        # Initialize buffers
        self.reset()
        
        print(f"Pipeline initialized on {self.device}")
    
    def reset(self):
        """Reset all buffers and tracking variables"""
        self.observation_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.confidence_history = []
        self.predictions_history = []
        self.caustic_history = []
        
        self.detection_made = False
        self.detection_timestep = None
        self.detection_class = None
        self.detection_confidence = None
        
        self.alert_queue = queue.Queue()
    
    @torch.no_grad()
    def process_observation(
        self,
        magnitude: float,
        timestamp: float,
        error: Optional[float] = None
    ) -> Dict:
        """
        Process single observation in real-time.
        
        Args:
            magnitude: Observed magnitude
            timestamp: Observation time
            error: Photometric error (optional)
            
        Returns:
            Dict with current predictions and alerts
        """
        start_time = time.perf_counter()
        
        # Add to buffer
        self.observation_buffer.append(magnitude)
        self.time_buffer.append(timestamp)
        
        # Convert to flux (simple approximation)
        baseline = 20.0  # Assumed baseline
        flux = 10.0 ** (-(magnitude - baseline) / 2.5)
        
        # Prepare input array
        n_obs = len(self.observation_buffer)
        X = np.full(self.buffer_size, CFG.PAD_VALUE, dtype=np.float32)
        X[:n_obs] = list(self.observation_buffer)
        
        # Convert flux array
        flux_array = np.full(self.buffer_size, CFG.PAD_VALUE, dtype=np.float32)
        for i, mag in enumerate(self.observation_buffer):
            flux_array[i] = 10.0 ** (-(mag - baseline) / 2.5)
        
        # Normalize
        X_norm = self.normalizer.transform(flux_array.reshape(1, -1))
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)
        
        # Run inference
        outputs = self.model.streaming_forward(X_tensor, n_obs - 1)
        
        # Get predictions
        binary_probs = torch.softmax(outputs['binary'], dim=-1).cpu().numpy()[0]
        binary_conf = binary_probs.max()
        binary_class = binary_probs.argmax()
        
        # Track confidence
        self.confidence_history.append(binary_conf)
        self.predictions_history.append(binary_class)
        
        # Get additional outputs if available
        anomaly_score = None
        caustic_prob = None
        
        if outputs.get('anomaly') is not None:
            anomaly_score = outputs['anomaly'].cpu().numpy()[0, 0]
        
        if outputs.get('caustic') is not None:
            caustic_prob = outputs['caustic'].cpu().numpy()[0, 0]
            self.caustic_history.append(caustic_prob)
        
        # Check for detection
        alerts = []
        
        if not self.detection_made and binary_conf >= self.confidence_threshold:
            self.detection_made = True
            self.detection_timestep = n_obs
            self.detection_class = binary_class
            self.detection_confidence = binary_conf
            
            if binary_class == 1:  # Binary detected
                alert = {
                    'type': 'BINARY_DETECTION',
                    'timestamp': timestamp,
                    'timestep': n_obs,
                    'confidence': float(binary_conf),
                    'message': f'Binary microlensing event detected at t={timestamp:.2f}'
                }
                alerts.append(alert)
                self.alert_queue.put(alert)
        
        # Check for caustic
        if caustic_prob is not None and caustic_prob >= self.caustic_threshold:
            alert = {
                'type': 'CAUSTIC_CROSSING',
                'timestamp': timestamp,
                'timestep': n_obs,
                'probability': float(caustic_prob),
                'message': f'Potential caustic crossing at t={timestamp:.2f}'
            }
            alerts.append(alert)
            self.alert_queue.put(alert)
        
        # Compute latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Prepare response
        response = {
            'timestep': n_obs,
            'timestamp': timestamp,
            'prediction': {
                'class': 'Binary' if binary_class == 1 else 'PSPL',
                'confidence': float(binary_conf),
                'probabilities': {
                    'PSPL': float(binary_probs[0]),
                    'Binary': float(binary_probs[1])
                }
            },
            'anomaly_score': float(anomaly_score) if anomaly_score is not None else None,
            'caustic_probability': float(caustic_prob) if caustic_prob is not None else None,
            'alerts': alerts,
            'latency_ms': latency_ms,
            'detection_status': {
                'detected': self.detection_made,
                'at_timestep': self.detection_timestep,
                'class': 'Binary' if self.detection_class == 1 else 'PSPL' if self.detection_class == 0 else None,
                'confidence': float(self.detection_confidence) if self.detection_confidence else None
            }
        }
        
        return response
    
    def get_summary(self) -> Dict:
        """Get summary of current event processing"""
        n_obs = len(self.observation_buffer)
        
        summary = {
            'observations_processed': n_obs,
            'buffer_utilization': n_obs / self.buffer_size,
            'detection_made': self.detection_made,
            'current_prediction': {
                'class': 'Binary' if self.predictions_history[-1] == 1 else 'PSPL' if self.predictions_history else 'Unknown',
                'confidence': float(self.confidence_history[-1]) if self.confidence_history else 0.0
            },
            'confidence_trajectory': {
                'mean': np.mean(self.confidence_history) if self.confidence_history else 0.0,
                'max': np.max(self.confidence_history) if self.confidence_history else 0.0,
                'current': float(self.confidence_history[-1]) if self.confidence_history else 0.0
            }
        }
        
        if self.detection_made:
            summary['detection_efficiency'] = self.detection_timestep / n_obs if n_obs > 0 else 0.0
        
        return summary


class AlertHandler(threading.Thread):
    """Background thread for handling alerts"""
    
    def __init__(self, alert_queue: queue.Queue, output_file: Optional[str] = None):
        super().__init__(daemon=True)
        self.alert_queue = alert_queue
        self.output_file = output_file
        self.running = True
    
    def run(self):
        """Process alerts in background"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                self.handle_alert(alert)
            except queue.Empty:
                continue
    
    def handle_alert(self, alert: Dict):
        """Handle individual alert"""
        # Log to console
        print(f"\n🚨 ALERT: {alert['type']}")
        print(f"   {alert['message']}")
        
        # Log to file if specified
        if self.output_file:
            with open(self.output_file, 'a') as f:
                json.dump(alert, f)
                f.write('\n')
    
    def stop(self):
        """Stop alert handler"""
        self.running = False


def benchmark_latency(
    pipeline: StreamingPipeline,
    n_observations: int = 1000
) -> Dict:
    """Benchmark pipeline latency"""
    
    print(f"\nBenchmarking latency ({n_observations} observations)...")
    
    latencies = []
    timestamps = np.linspace(0, 100, n_observations)
    magnitudes = 20.0 + np.random.normal(0, 0.1, n_observations)
    
    # Add a caustic spike
    spike_pos = n_observations // 2
    magnitudes[spike_pos-5:spike_pos+5] -= 3.0  # Brighter = lower magnitude
    
    pipeline.reset()
    
    for i, (t, mag) in enumerate(zip(timestamps, magnitudes)):
        result = pipeline.process_observation(mag, t)
        latencies.append(result['latency_ms'])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_observations} observations")
    
    latencies = np.array(latencies)
    
    stats = {
        'mean_latency_ms': latencies.mean(),
        'median_latency_ms': np.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'max_latency_ms': latencies.max(),
        'throughput_per_sec': 1000 / latencies.mean()
    }
    
    print("\nLatency Statistics:")
    print(f"  Mean: {stats['mean_latency_ms']:.3f} ms")
    print(f"  P95: {stats['p95_latency_ms']:.3f} ms")
    print(f"  P99: {stats['p99_latency_ms']:.3f} ms")
    print(f"  Throughput: {stats['throughput_per_sec']:.0f} obs/sec")
    
    if stats['p99_latency_ms'] < CFG.MAX_LATENCY_MS:
        print(f"  ✅ Meets real-time requirement (<{CFG.MAX_LATENCY_MS} ms)")
    else:
        print(f"  ⚠️  Exceeds target latency ({CFG.MAX_LATENCY_MS} ms)")
    
    return stats


def main():
    """Demo streaming inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Streaming inference demo')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--normalizer', required=True, help='Path to normalizer')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--simulate', action='store_true', help='Simulate stream')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StreamingPipeline(
        model_path=args.model,
        normalizer_path=args.normalizer
    )
    
    if args.benchmark:
        # Run benchmark
        stats = benchmark_latency(pipeline, n_observations=1000)
        
        # Save results
        with open('benchmark_results.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    if args.simulate:
        # Simulate real-time stream
        print("\nSimulating real-time observation stream...")
        print("Press Ctrl+C to stop\n")
        
        # Start alert handler
        alert_handler = AlertHandler(pipeline.alert_queue, 'alerts.log')
        alert_handler.start()
        
        try:
            # Generate synthetic observations
            t = 0
            while True:
                # Simulate observation arrival
                mag = 20.0 + np.random.normal(0, 0.1)
                
                # Add occasional spike
                if np.random.rand() < 0.01:
                    mag -= 2.0  # Caustic-like spike
                
                result = pipeline.process_observation(mag, t)
                
                # Display current status
                print(f"\rT={t:6.2f} | {result['prediction']['class']:6s} "
                      f"({result['prediction']['confidence']:.2%}) | "
                      f"Latency: {result['latency_ms']:.2f}ms", end='')
                
                t += 0.1
                time.sleep(0.05)  # Simulate observation cadence
                
        except KeyboardInterrupt:
            print("\n\nStopping simulation...")
            alert_handler.stop()
            
            # Print summary
            summary = pipeline.get_summary()
            print("\nFinal Summary:")
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
