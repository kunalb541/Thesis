#!/usr/bin/env python3
"""
Transformer Visualization v16.0.1
=================================

FIXED: Dynamic array sizing based on actual data shape
ENHANCED: Ultra-high resolution evolution plots (100 points)

Author: Kunal Bhatia
Version: 16.0.1
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import argparse
from tqdm import tqdm
import sys

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class StableNormalizer:
    """Robust normalizer (for pickle compatibility)"""
    
    def __init__(self, pad_value=-1.0):
        self.pad_value = pad_value
        self.mean = 0.0
        self.std = 1.0
    
    def fit(self, X):
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            valid_values = X[valid_mask]
            self.mean = np.median(valid_values)
            self.std = np.median(np.abs(valid_values - self.mean))
            
            if self.std < 1e-8:
                self.std = 1.0
            
            self.mean = np.clip(self.mean, -100, 100)
            self.std = np.clip(self.std, 0.01, 100)
        
        return self
    
    def transform(self, X):
        X_norm = X.copy()
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            X_norm[valid_mask] = (X[valid_mask] - self.mean) / self.std
            X_norm[valid_mask] = np.clip(X_norm[valid_mask], -10, 10)
        
        return np.nan_to_num(X_norm, nan=0.0, posinf=10.0, neginf=-10.0)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def load_model_and_data(experiment_name, data_path):
    """Load trained model and test data"""
    # Find experiment directory
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f'{experiment_name}_*'))
    
    if not exp_dirs:
        print(f"No experiment found: {experiment_name}")
        return None, None, None, None, None
    
    exp_dir = exp_dirs[-1]
    print(f"Loading experiment: {exp_dir.name}")
    
    # Load model
    sys.path.insert(0, str(Path(__file__).parent))
    from transformer import MicrolensingTransformer
    
    model_path = exp_dir / 'best_model.pt'
    config_path = exp_dir / 'config.json'
    normalizer_path = exp_dir / 'normalizer.pkl'
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Create model
    model = MicrolensingTransformer(
        n_points=1500,  # Will be overridden
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 4),
        dim_feedforward=config.get('d_model', 128) * 4,
        dropout=config.get('dropout', 0.1),
        pad_value=-1.0,
        causal_attention=not config.get('no_causal_attention', False)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    
    # Load normalizer
    with open(normalizer_path, 'rb') as f:
        normalizer = pickle.load(f)
    
    # Load data
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    timestamps = data['timestamps'] if 'timestamps' in data else np.linspace(-100, 100, X.shape[1])
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    n_points = X.shape[1]
    X_norm = normalizer.transform(X)
    
    print(f"✅ Data loaded ({len(X)} events, n_points={n_points})")
    
    return model, X_norm, y, timestamps, n_points


def visualize_attention_patterns(model, X_norm, y, timestamps, event_idx, output_dir):
    """Visualize attention patterns across all layers"""
    
    light_curve = X_norm[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    # Get model outputs with attention
    x_tensor = torch.from_numpy(light_curve).unsqueeze(0).float()
    
    with torch.no_grad():
        outputs = model(x_tensor, return_all=True, return_attention=True)
        attention_weights = outputs['attention_weights']  # List of [B, H, T, T]
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
    
    # Find valid observations
    valid_mask = light_curve != -1.0
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    # Create figure
    num_layers = len(attention_weights)
    fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 10))
    
    for layer_idx, attn in enumerate(attention_weights):
        attn_mat = attn[0].mean(dim=0).cpu().numpy()  # Average over heads [T, T]
        
        # Top row: Full attention matrix
        ax = axes[0, layer_idx]
        
        # Only show valid-to-valid attention
        attn_valid = attn_mat[valid_indices][:, valid_indices]
        
        im = ax.imshow(attn_valid, cmap='hot', aspect='auto', 
                      vmin=0, vmax=attn_valid.max())
        ax.set_title(f'Layer {layer_idx+1}\nAttention Matrix', fontweight='bold')
        ax.set_xlabel('Source Position')
        ax.set_ylabel('Target Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add causal boundary if enabled
        if model.causal_attention:
            ax.plot([0, n_valid], [0, n_valid], 'b--', linewidth=2, 
                   label='Causal Boundary', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)
        
        # Bottom row: Average attention per timestep
        ax = axes[1, layer_idx]
        
        # Average attention received by each position
        avg_attn_received = attn_mat[:, valid_indices].mean(axis=0)
        
        ax.plot(timestamps[valid_indices], avg_attn_received, 'o-', 
               linewidth=2, markersize=4, color='darkblue')
        ax.set_xlabel('Time (days)', fontweight='bold')
        ax.set_ylabel('Avg Attention Received', fontweight='bold')
        ax.set_title(f'Layer {layer_idx+1}\nTemporal Attention', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, 
                  alpha=0.5, label='t=0')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(f'Attention Patterns - True: {class_names[true_label]}, '
                f'Pred: {class_names[probs.argmax()]} ({probs.max()*100:.1f}%)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'attention_patterns_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path.name}")
    plt.close()


def visualize_caustic_features(model, X_norm, y, timestamps, event_idx, output_dir):
    """Visualize SimpleCausticDetector feature extraction"""
    
    light_curve = X_norm[event_idx]
    light_curve_raw = X_norm[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    # Get embeddings and caustic features
    x_tensor = torch.from_numpy(light_curve).unsqueeze(0).float()
    
    with torch.no_grad():
        # Get intermediate embeddings
        padding_mask = model.create_padding_mask(x_tensor)
        
        # Input embedding
        if x_tensor.dim() == 2:
            x_tensor_3d = x_tensor.unsqueeze(-1)
        else:
            x_tensor_3d = x_tensor
        
        x_clean = x_tensor_3d.clone()
        x_clean[padding_mask.unsqueeze(-1)] = 0.0
        
        x_embed = model.input_embed(x_clean)
        
        # Add positional encoding
        pos_encoding = model.pos_encoding(x_embed, padding_mask)
        x_embed = x_embed + pos_encoding
        
        # Through transformer
        for layer in model.layers:
            x_embed, _ = layer(x_embed, padding_mask)
        
        x_embed = model.norm(x_embed)
        
        # Extract caustic features
        caustic_features = model.caustic_detector(x_embed, padding_mask)
        
        # Get predictions
        outputs = model(x_tensor, return_all=True)
        probs = F.softmax(outputs['logits'], dim=-1).squeeze(0).numpy()
        caustic_prob = torch.sigmoid(outputs['caustic']).item()
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    valid_mask = light_curve != -1.0
    times = timestamps[valid_mask]
    values = light_curve_raw[valid_mask]
    
    # 1. Light curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(times, values, c='darkblue', s=15, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Time (days)', fontweight='bold')
    ax1.set_ylabel('Normalized Flux', fontweight='bold')
    ax1.set_title(f'Light Curve - True: {class_names[true_label]}, '
                 f'Pred: {class_names[probs.argmax()]} ({probs.max()*100:.1f}%)',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # 2. Embedding evolution (take max over d_model dimension)
    ax2 = fig.add_subplot(gs[1, :])
    embed_strength = x_embed[0].max(dim=-1)[0].cpu().numpy()
    embed_strength_valid = embed_strength[valid_mask]
    
    ax2.plot(times, embed_strength_valid, 'o-', linewidth=2, markersize=4, 
            color='purple', label='Max Embedding Strength')
    ax2.set_xlabel('Time (days)', fontweight='bold')
    ax2.set_ylabel('Embedding Strength', fontweight='bold')
    ax2.set_title('Transformer Embeddings (max over d_model)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Caustic feature values
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Manually compute the 4 features to show them
    valid_expand = (~padding_mask).unsqueeze(-1).float()
    x_pooled = (x_embed * valid_expand).sum(dim=1) / (valid_expand.sum(dim=1) + 1e-8)
    
    # Feature 1: Peak strength
    x_masked = x_embed.masked_fill(padding_mask.unsqueeze(-1), -65000.0)
    max_strength = x_masked.max(dim=1)[0].max(dim=-1)[0].item()
    
    # Feature 2: Variance
    x_var = ((x_embed - x_pooled.unsqueeze(1))**2 * valid_expand).sum(dim=1) / (valid_expand.sum(dim=1) + 1e-8)
    variance = x_var.max(dim=-1)[0].item()
    
    # Feature 3: Peak count (approximate)
    threshold = x_pooled.max(dim=-1, keepdim=True)[0] * 0.7
    high_act = (x_embed > threshold.unsqueeze(1)).float()
    peak_count = (high_act * valid_expand).sum(dim=1).max(dim=-1)[0].item()
    
    # Feature 4: Asymmetry
    T = x_embed.shape[1]
    mid = T // 2
    early = (x_embed[:, :mid] * valid_expand[:, :mid]).sum(dim=1).max(dim=-1)[0].item()
    late = (x_embed[:, mid:] * valid_expand[:, mid:]).sum(dim=1).max(dim=-1)[0].item()
    asymmetry = abs(early - late)
    
    features = ['Peak\nStrength', 'Variance\n(Spikiness)', 'Peak\nCount', 'Asymmetry']
    values_feat = [max_strength, variance, peak_count, asymmetry]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    bars = ax3.bar(features, values_feat, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Feature Value', fontweight='bold')
    ax3.set_title('SimpleCausticDetector Features', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, values_feat):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Class probabilities
    ax4 = fig.add_subplot(gs[2, 1])
    
    class_colors = ['gray', 'darkred', 'darkblue']
    bars = ax4.bar(class_names, probs, color=class_colors, alpha=0.7, 
                  edgecolor='black', linewidth=2)
    ax4.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_ylabel('Probability', fontweight='bold')
    ax4.set_title('Classification Output', fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Highlight predicted class
    bars[probs.argmax()].set_edgecolor('gold')
    bars[probs.argmax()].set_linewidth(4)
    
    # Add values
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Caustic detection
    ax5 = fig.add_subplot(gs[3, 0])
    
    ax5.bar(['Not Binary', 'Binary'], [1-caustic_prob, caustic_prob],
           color=['lightgray', 'darkblue'], alpha=0.7, edgecolor='black', linewidth=2)
    ax5.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_ylabel('Probability', fontweight='bold')
    ax5.set_title('Caustic Detection (Morphology-Based)', fontweight='bold')
    ax5.set_ylim([0, 1.05])
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.text(0, 1-caustic_prob, f'{(1-caustic_prob)*100:.1f}%', 
            ha='center', va='bottom', fontweight='bold')
    ax5.text(1, caustic_prob, f'{caustic_prob*100:.1f}%', 
            ha='center', va='bottom', fontweight='bold')
    
    # 6. Feature interpretation
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    
    interpretation = f"""
    Feature Interpretation:
    
    Peak Strength: {max_strength:.2f}
    → {'High' if max_strength > 2 else 'Low'} caustic crossing strength
    
    Variance: {variance:.2f}
    → {'Spiky' if variance > 1 else 'Smooth'} light curve
    
    Peak Count: {peak_count:.0f}
    → {'Multiple' if peak_count > 100 else 'Single'} peaks detected
    
    Asymmetry: {asymmetry:.2f}
    → {'Asymmetric' if asymmetry > 1 else 'Symmetric'} evolution
    
    Binary Indicators:
    {'✓' if max_strength > 2 else '✗'} High peak strength
    {'✓' if variance > 1 else '✗'} High variance
    {'✓' if peak_count > 100 else '✗'} Multiple peaks
    {'✓' if asymmetry > 1 else '✗'} Asymmetric
    
    Caustic Probability: {caustic_prob*100:.1f}%
    """
    
    ax6.text(0.1, 0.5, interpretation, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle(f'SimpleCausticDetector Analysis - Event {event_idx}',
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'caustic_features_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path.name}")
    plt.close()


def visualize_classification_evolution(model, X_norm, y, timestamps, event_idx, output_dir, n_points):
    """ULTRA-HIGH-RES: Show how classification evolves (100 points) - v16.0.1 FIXED"""
    
    light_curve = X_norm[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    # ULTRA-HIGH RESOLUTION: 100 fractions
    fractions = np.linspace(0.1, 1.0, 100)
    
    flat_probs, pspl_probs, binary_probs = [], [], []
    caustic_probs = []
    confidences = []
    
    with torch.no_grad():
        for frac in fractions:
            n_pts = int(n_points * frac)
            partial_curve = np.full(n_points, -1.0, dtype=np.float32)
            partial_curve[:n_pts] = light_curve[:n_pts]
            
            x = torch.from_numpy(partial_curve).unsqueeze(0).float()
            outputs = model(x, return_all=True)
            
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            
            flat_probs.append(probs[0])
            pspl_probs.append(probs[1])
            binary_probs.append(probs[2])
            caustic_probs.append(torch.sigmoid(outputs['caustic']).item())
            confidences.append(probs.max())
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    completeness = [f*100 for f in fractions]
    
    # Top: Light curve
    ax = axes[0]
    valid_mask = light_curve != -1.0
    times = timestamps[valid_mask]
    values = light_curve[valid_mask]
    
    ax.scatter(times, values, c='darkblue', s=15, alpha=0.7, 
              edgecolors='black', linewidth=0.5)
    ax.set_ylabel('Normalized Flux', fontweight='bold')
    ax.set_title(f'ULTRA-HIGH-RES Classification Evolution (100 Points) - True: {class_names[true_label]}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Middle: Class probabilities
    ax = axes[1]
    
    ax.plot(completeness, flat_probs, '-', linewidth=1.5,
           color='gray', label='Flat', alpha=0.8)
    ax.plot(completeness, pspl_probs, '-', linewidth=1.5,
           color='darkred', label='PSPL', alpha=0.8)
    ax.plot(completeness, binary_probs, '-', linewidth=1.5,
           color='darkblue', label='Binary', alpha=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_ylabel('Class Probability', fontweight='bold')
    ax.legend(loc='right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # Bottom: Caustic detection + confidence
    ax = axes[2]
    
    ax.plot(completeness, caustic_probs, '-', linewidth=1.5,
           color='purple', label='Caustic Probability')
    ax.plot(completeness, confidences, '-', linewidth=1.5,
           color='orange', label='Confidence', alpha=0.7)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
              label='High Confidence')
    
    ax.set_xlabel('Observation Completeness (%)', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    output_path = output_dir / f'ultrahighres_classification_evolution_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path.name}")
    plt.close()


def compare_binary_vs_pspl(model, X_norm, y, timestamps, output_dir):
    """Compare attention and features for Binary vs PSPL events"""
    
    # Find good examples
    binary_idx = np.where(y == 2)[0][0]
    pspl_idx = np.where(y == 1)[0][0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for row, (idx, event_type) in enumerate([(binary_idx, 'Binary'), (pspl_idx, 'PSPL')]):
        light_curve = X_norm[idx]
        x_tensor = torch.from_numpy(light_curve).unsqueeze(0).float()
        
        with torch.no_grad():
            outputs = model(x_tensor, return_all=True, return_attention=True)
            attention_weights = outputs['attention_weights']
            probs = F.softmax(outputs['logits'], dim=-1).squeeze(0).numpy()
            caustic_prob = torch.sigmoid(outputs['caustic']).item()
        
        valid_mask = light_curve != -1.0
        times = timestamps[valid_mask]
        values = light_curve[valid_mask]
        
        # Column 1: Light curve
        ax = axes[row, 0]
        color = 'darkblue' if event_type == 'Binary' else 'darkred'
        ax.scatter(times, values, c=color, s=15, alpha=0.7,
                  edgecolors='black', linewidth=0.5)
        ax.set_ylabel('Normalized Flux', fontweight='bold')
        ax.set_title(f'{event_type} Event\nCaustic Prob: {caustic_prob*100:.1f}%',
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Column 2: Attention (last layer, averaged over heads)
        ax = axes[row, 1]
        attn = attention_weights[-1][0].mean(dim=0).cpu().numpy()
        valid_indices = np.where(valid_mask)[0]
        attn_valid = attn[valid_indices][:, valid_indices]
        
        im = ax.imshow(attn_valid, cmap='hot', aspect='auto')
        ax.set_title('Final Layer Attention', fontweight='bold')
        ax.set_xlabel('Source Position')
        ax.set_ylabel('Target Position')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Column 3: Class probabilities
        ax = axes[row, 2]
        class_names = ['Flat', 'PSPL', 'Binary']
        colors_bar = ['gray', 'darkred', 'darkblue']
        
        bars = ax.bar(class_names, probs, color=colors_bar, alpha=0.7,
                     edgecolor='black', linewidth=2)
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel('Probability', fontweight='bold')
        ax.set_title('Classification', fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2., prob,
                   f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Binary vs PSPL Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'binary_vs_pspl_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize transformer internals')
    parser.add_argument('--experiment_name', required=True, help='Experiment to visualize')
    parser.add_argument('--data', required=True, help='Test dataset path')
    parser.add_argument('--output_dir', default='../results/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--event_indices', nargs='+', type=int, default=None,
                       help='Specific event indices to visualize (default: auto-select)')
    parser.add_argument('--n_examples', type=int, default=3,
                       help='Number of examples per class if auto-selecting')
    
    args = parser.parse_args()
    
    # Load model and data
    model, X_norm, y, timestamps, n_points = load_model_and_data(args.experiment_name, args.data)
    
    if model is None:
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("TRANSFORMER VISUALIZATION v16.0.1")
    print(f"{'='*70}\n")
    
    # Select events to visualize
    if args.event_indices:
        event_indices = args.event_indices
    else:
        # Auto-select good examples from each class
        event_indices = []
        for class_id in range(3):
            class_mask = y == class_id
            class_indices = np.where(class_mask)[0]
            selected = np.random.choice(class_indices, 
                                       min(args.n_examples, len(class_indices)),
                                       replace=False)
            event_indices.extend(selected)
    
    print(f"Visualizing {len(event_indices)} events...\n")
    
    # Generate visualizations
    for i, event_idx in enumerate(event_indices):
        print(f"\nEvent {i+1}/{len(event_indices)} (index {event_idx}):")
        print(f"  True class: {['Flat', 'PSPL', 'Binary'][y[event_idx]]}")
        
        # 1. Attention patterns
        print("  → Attention patterns...")
        visualize_attention_patterns(model, X_norm, y, timestamps, event_idx, output_dir)
        
        # 2. Caustic features
        print("  → Caustic features...")
        visualize_caustic_features(model, X_norm, y, timestamps, event_idx, output_dir)
        
        # 3. Classification evolution (ULTRA-HIGH-RES: 100 points)
        print("  → Classification evolution (ULTRA-HIGH-RES)...")
        visualize_classification_evolution(model, X_norm, y, timestamps, event_idx, output_dir, n_points)
    
    # 4. Binary vs PSPL comparison
    print("\n  → Binary vs PSPL comparison...")
    compare_binary_vs_pspl(model, X_norm, y, timestamps, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
