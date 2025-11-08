#!/usr/bin/env python3
"""
Detailed Binary Event Visualization (CUDA-ENABLED)
===================================================

Shows individual high-confidence binary events with:
- Light curve with features highlighted
- Confidence evolution (as if observing in real-time)
- Model's "reasoning" via attention (if available)
- Comparison to ground truth

Perfect for thesis figures!

Usage:
    python visualize_binary_events_detailed_cuda.py \
        --model_path ../results/critical_20gpu_working_20251108_135150/best_model.pt \
        --data_path ../data/raw/test.npz \
        --event_id 42 \
        --output_dir ./figures/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import json
import sys
from tqdm import tqdm

sys.path.insert(0, '/pfs/data6/home/hd/hd_hd/hd_vm305/Thesis/code')

from transformer import SimpleStableTransformer
from scipy.signal import find_peaks

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300


def plot_binary_event_detailed(model, light_curve, true_label, event_id,
                               meta=None, output_path='event_detail.png', device='cuda'):
    """
    Create detailed visualization of a single binary event
    """
    # Get prediction
    model.eval()
    with torch.no_grad():
        x = torch.tensor([light_curve], dtype=torch.float32).to(device)
        output = model(x, return_all_timesteps=False)
        logits = output['binary']
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    binary_prob = probs[1]
    pred_label = 'Binary' if binary_prob > 0.5 else 'Single'
    true_label_str = 'Binary' if true_label == 1 else 'Single'
    is_correct = (binary_prob > 0.5) == (true_label == 1)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== MAIN LIGHT CURVE =====
    ax1 = fig.add_subplot(gs[0:2, :])
    
    # Plot light curve
    time_points = np.arange(len(light_curve))
    ax1.plot(time_points, light_curve, 'b-', linewidth=2, alpha=0.7, label='Light Curve')
    
    # Find and mark peaks
    peaks, properties = find_peaks(light_curve, height=1.0, distance=50, prominence=0.5)
    if len(peaks) > 0:
        ax1.scatter(peaks, light_curve[peaks], color='red', s=200, 
                   zorder=5, marker='v', label=f'Detected Peaks ({len(peaks)})',
                   edgecolors='darkred', linewidth=2)
        
        # Annotate peaks
        for i, peak in enumerate(peaks):
            ax1.annotate(f'Peak {i+1}\nt={peak}', 
                        xy=(peak, light_curve[peak]),
                        xytext=(peak, light_curve[peak] + 1.5),
                        fontsize=9, ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                      color='red', lw=2))
    
    # Baseline
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Baseline')
    
    # Anomaly regions (if any)
    anomaly_threshold = 2.0
    anomalies = np.where(np.abs(light_curve) > anomaly_threshold)[0]
    if len(anomalies) > 0:
        ax1.fill_between(time_points, -5, 5, 
                        where=(np.abs(light_curve) > anomaly_threshold),
                        alpha=0.2, color='orange', label='High Flux Region')
    
    # Labels and styling
    ax1.set_xlabel('Time Point (arbitrary units)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Flux', fontsize=12, fontweight='bold')
    
    # Title with prediction
    title_color = 'green' if is_correct else 'red'
    title_symbol = '✓' if is_correct else '✗'
    ax1.set_title(
        f'{title_symbol} Event {event_id}: {true_label_str} Event\n'
        f'Model Prediction: {pred_label} (Confidence: {binary_prob:.1%})',
        fontsize=14, fontweight='bold', color=title_color
    )
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add metadata box if available
    if meta is not None:
        info_text = []
        if 'u0' in meta:
            info_text.append(f"Impact Parameter: u₀ = {meta['u0']:.3f}")
        if 't_E' in meta:
            info_text.append(f"Einstein Time: t_E = {meta['t_E']:.1f} days")
        if 'separation' in meta and meta['separation'] is not None:
            info_text.append(f"Binary Separation: d = {meta['separation']:.3f}")
        if 'mass_ratio' in meta and meta['mass_ratio'] is not None:
            info_text.append(f"Mass Ratio: q = {meta['mass_ratio']:.3f}")
        
        if info_text:
            ax1.text(0.02, 0.98, '\n'.join(info_text),
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, family='monospace')
    
    # ===== PREDICTION CONFIDENCE =====
    ax2 = fig.add_subplot(gs[2, 0])
    
    categories = ['Single', 'Binary']
    colors = ['skyblue', 'coral']
    bars = ax2.bar(categories, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Highlight predicted class
    predicted_idx = 1 if binary_prob > 0.5 else 0
    bars[predicted_idx].set_edgecolor('red')
    bars[predicted_idx].set_linewidth(3)
    
    ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.set_title('Model Output', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== FEATURE ANALYSIS =====
    ax3 = fig.add_subplot(gs[2, 1])
    
    # Compute basic features
    features = {
        'Peak Count': len(peaks),
        'Max Flux': float(light_curve.max()),
        'Baseline Std': float(np.std(light_curve[:200])),
        'Asymmetry': float(np.abs(np.mean(light_curve[:750]) - np.mean(light_curve[750:]))),
    }
    
    # Create horizontal bar chart
    feature_names = list(features.keys())
    feature_values = list(features.values())
    
    y_pos = np.arange(len(feature_names))
    bars = ax3.barh(y_pos, feature_values, color='mediumpurple', alpha=0.7, edgecolor='black')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(feature_names)
    ax3.set_xlabel('Value', fontsize=11, fontweight='bold')
    ax3.set_title('Extracted Features', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, feature_values)):
        ax3.text(val + 0.05 * max(feature_values), i, f'{val:.2f}',
                va='center', fontweight='bold', fontsize=9)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed visualization: {output_path}")
    plt.close()


def plot_confidence_evolution(model, light_curve, true_label, output_path='confidence_evolution.png', device='cuda'):
    """
    Show how confidence evolves as more data is observed
    Simulates real-time classification
    """
    model.eval()
    
    # Sample different amounts of data (10%, 20%, ..., 100%)
    fractions = np.linspace(0.1, 1.0, 10)
    confidences = []
    binary_probs = []
    
    # Batch process for speed
    partial_curves = []
    for frac in fractions:
        n_points = int(1500 * frac)
        partial_curve = np.zeros(1500)
        partial_curve[:n_points] = light_curve[:n_points]
        partial_curves.append(partial_curve)
    
    # Get all predictions at once
    with torch.no_grad():
        x = torch.tensor(partial_curves, dtype=torch.float32).to(device)
        output = model(x, return_all_timesteps=False)
        logits = output['binary']
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    binary_probs = probs[:, 1].tolist()
    confidences = probs.max(axis=1).tolist()
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # ===== BINARY PROBABILITY EVOLUTION =====
    ax = axes[0]
    
    days = fractions * 1500  # Assuming time points are days
    ax.plot(days, binary_probs, 'o-', linewidth=3, markersize=8,
           color='darkblue', label='Binary Probability')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
              label='Decision Threshold')
    ax.axhline(y=0.8, color='orange', linestyle=':', linewidth=2,
              label='High Confidence Threshold')
    
    # Shade regions
    ax.fill_between(days, 0.8, 1.0, alpha=0.2, color='green', label='High Conf Binary')
    ax.fill_between(days, 0.5, 0.8, alpha=0.2, color='yellow', label='Moderate Conf')
    ax.fill_between(days, 0.0, 0.5, alpha=0.2, color='lightblue', label='Predict Single')
    
    true_label_str = 'Binary' if true_label == 1 else 'Single'
    ax.set_ylabel('Binary Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Real-Time Classification Evolution (True: {true_label_str})',
                fontsize=13, fontweight='bold')
    ax.legend(loc='right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # Annotate key points
    # When did it first confidently predict binary?
    high_conf_binary = np.where(np.array(binary_probs) > 0.8)[0]
    if len(high_conf_binary) > 0:
        first_conf = high_conf_binary[0]
        ax.annotate(f'First high-confidence\nbinary prediction\n({fractions[first_conf]*100:.0f}% data)',
                   xy=(days[first_conf], binary_probs[first_conf]),
                   xytext=(days[first_conf] + 200, 0.9),
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # ===== CONFIDENCE EVOLUTION =====
    ax = axes[1]
    
    ax.plot(days, confidences, 's-', linewidth=3, markersize=8,
           color='purple', label='Overall Confidence')
    ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2,
              label='80% Threshold')
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2,
              label='90% Threshold')
    
    ax.set_xlabel('Time Points Observed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Evolution Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confidence evolution: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--event_id', type=int, default=None,
                       help='Specific event ID to visualize (default: find high-conf binary)')
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--min_confidence', type=float, default=0.9)
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Detect architecture from checkpoint
    if 'pos_embed' in checkpoint['model_state_dict']:
        d_model = checkpoint['model_state_dict']['pos_embed'].shape[2]
    else:
        d_model = checkpoint['model_state_dict']['input_proj.0.weight'].shape[0]
    
    num_layers = sum(1 for k in checkpoint['model_state_dict'].keys() 
                    if k.startswith('blocks.') and '.norm1.weight' in k)
    
    # Detect nhead
    possible_nheads = [2, 4, 8, 16]
    nhead = 4  # default
    for nh in possible_nheads:
        if d_model % nh == 0:
            nhead = nh
            break
    
    print(f"Detected: d_model={d_model}, num_layers={num_layers}, nhead={nhead}")
    
    model = SimpleStableTransformer(
        n_points=1500,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=d_model * 4,
        dropout=0.2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Load data
    print("\nLoading data...")
    data = np.load(args.data_path)
    X = data['X'].squeeze()
    y = data['y']
    
    meta_list = None
    if 'meta_json' in data:
        try:
            meta_json = data['meta_json']
            if isinstance(meta_json, np.ndarray):
                if meta_json.ndim == 0:
                    meta_json = np.array([meta_json.item()])
                meta_list = [json.loads(m) if isinstance(m, (str, bytes)) else m 
                            for m in meta_json]
            else:
                meta_list = [json.loads(meta_json)]
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            meta_list = None
    
    # Find high-confidence binary event if not specified
    if args.event_id is None:
        print(f"\nFinding high-confidence binary event (≥{args.min_confidence*100:.0f}%)...")
        
        # Get predictions for all events (batched for speed)
        predictions = []
        confidences = []
        
        batch_size = 256 if device.type == 'cuda' else 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X), batch_size), desc="Predicting"):
                batch_end = min(i + batch_size, len(X))
                x_batch = torch.tensor(X[i:batch_end], dtype=torch.float32).to(device)
                
                output = model(x_batch, return_all_timesteps=False)
                logits = output['binary']
                probs = torch.softmax(logits, dim=1)
                
                pred = probs.argmax(dim=1).cpu().numpy()
                conf = probs.max(dim=1).values.cpu().numpy()
                
                predictions.extend(pred)
                confidences.extend(conf)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Find high-confidence binary predictions that are correct
        high_conf_binary = np.where((predictions == 1) & 
                                    (confidences >= args.min_confidence) &
                                    (y == 1))[0]
        
        if len(high_conf_binary) == 0:
            print("No high-confidence binary events found, using any binary event")
            high_conf_binary = np.where(y == 1)[0]
        
        event_id = high_conf_binary[0]
        print(f"Selected event {event_id} (confidence: {confidences[event_id]:.1%})")
    else:
        event_id = args.event_id
    
    # Get event data
    light_curve = X[event_id]
    true_label = y[event_id]
    meta = meta_list[event_id] if meta_list is not None else None
    
    # Generate visualizations
    print(f"\nGenerating visualizations for event {event_id}...")
    
    print("1. Detailed event visualization...")
    plot_binary_event_detailed(
        model, light_curve, true_label, event_id, meta,
        output_path=output_dir / f'event_{event_id}_detailed.png',
        device=device
    )
    
    print("2. Confidence evolution (real-time simulation)...")
    plot_confidence_evolution(
        model, light_curve, true_label,
        output_path=output_dir / f'event_{event_id}_confidence_evolution.png',
        device=device
    )
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
