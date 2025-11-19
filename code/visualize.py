#!/usr/bin/env python3
"""
Transformer Visualization

Comprehensive visualization suite for MicrolensingTransformer v1.0

Features:
- Attention pattern visualization (with hooks)
- Classification evolution analysis
- Temporal encoding analysis
- Confidence progression
- Comparative analysis (Binary vs PSPL)

Author: Kunal Bhatia
Version: 1.0.0
Date: November 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys
from collections import defaultdict

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def create_delta_t_from_timestamps(timestamps):
    """Create delta_t array from timestamps."""
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    
    return delta_t


def compute_lengths_from_flux(flux, pad_value=-1.0):
    """Compute valid sequence lengths from padded flux."""
    return np.sum(flux != pad_value, axis=1).astype(np.int64)


class AttentionCapture:
    """Hook-based attention weight capture for visualization"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self):
        """Register forward hooks to capture attention weights"""
        self.attention_weights = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Attention module returns (output, cache)
                # We need to capture the attention weights from inside
                # For now, we'll capture from the output if possible
                pass
            return hook
        
        # Register hooks on each transformer layer's attention module
        for i, layer in enumerate(self.model.layers):
            hook = layer.attention.register_forward_hook(
                self._create_attention_hook(i)
            )
            self.hooks.append(hook)
    
    def _create_attention_hook(self, layer_idx):
        """Create a hook that captures attention weights"""
        def hook(module, input, output):
            # This is tricky - we need to capture attention weights
            # from inside the attention computation
            # For now, we'll compute them manually after forward pass
            pass
        return hook
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_weights(self, flux, delta_t, lengths):
        """
        Get attention weights by manually computing them
        
        Since the new architecture doesn't expose attention weights directly,
        we'll compute them by accessing the attention module's internals.
        """
        self.attention_weights = []
        
        with torch.no_grad():
            # Embed inputs
            x = self.model.flux_embedding(flux.unsqueeze(-1))
            temporal = self.model.temporal_encoding(delta_t)
            x = x + temporal
            
            # Create padding mask
            if lengths is not None:
                padding_mask = self.model.create_padding_mask_from_lengths(lengths, flux.size(1))
            else:
                padding_mask = None
            
            # Go through each layer and capture attention
            for layer in self.model.layers:
                # Get attention module
                attn_module = layer.attention
                
                # Compute Q, K, V
                x_normed = layer.norm1(x)
                B, N, D = x_normed.shape
                
                qkv = attn_module.qkv_proj(x_normed)
                qkv = qkv.reshape(B, N, 3, attn_module.n_heads, attn_module.d_head)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) * attn_module.scale
                
                # Apply causal mask
                causal_mask = attn_module._create_causal_window_mask(N, N, x.device)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Apply padding mask
                if padding_mask is not None:
                    key_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~key_mask, float('-inf'))
                
                # Softmax
                attn_weights = F.softmax(scores, dim=-1)
                
                # Store
                self.attention_weights.append(attn_weights)
                
                # Apply attention and continue forward pass
                attn_out, _ = layer.attention(x_normed, padding_mask=padding_mask, 
                                             return_cache=False)
                x = x + attn_out
                x = x + layer.ffn(layer.norm2(x))
        
        return self.attention_weights


def load_model_and_data(experiment_name, data_path, device='cuda'):
    """Load trained model and test data"""
    # Find experiment directory
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f'{experiment_name}_*'))
    
    if not exp_dirs:
        print(f"❌ No experiment found: {experiment_name}")
        return None, None, None, None, None, None
    
    exp_dir = exp_dirs[-1]
    print(f"Loading experiment: {exp_dir.name}")
    
    # Load model
    sys.path.insert(0, str(Path(__file__).parent))
    from transformer import MicrolensingTransformer, ModelConfig, count_parameters
    
    model_path = exp_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = exp_dir / 'final_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found in: {exp_dir}")
        return None, None, None, None, None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config_path = exp_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            
            config = ModelConfig(
                d_model=config_dict.get('d_model', 128),
                n_heads=config_dict.get('n_heads', 8),
                n_layers=config_dict.get('n_layers', 4),
                dropout=config_dict.get('dropout', 0.1),
                attention_window=config_dict.get('attention_window', 64),
                train_final_only=True,
                use_adaptive_normalization=True
            )
        else:
            print(f"❌ Config not found")
            return None, None, None, None, None, None
    
    # Create model
    model = MicrolensingTransformer(config)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # Load temperature
    if 'calibration' in checkpoint and 'temperature' in checkpoint['calibration']:
        model.set_temperature(checkpoint['calibration']['temperature'])
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded ({count_parameters(model):,} params)")
    
    # Load data
    data = np.load(data_path)
    
    # Load flux
    if 'flux' in data:
        flux = data['flux']
    elif 'X' in data:
        flux = data['X']
    else:
        print("❌ No flux data found")
        return None, None, None, None, None, None
    
    if flux.ndim == 3:
        flux = flux.squeeze(1)
    
    # Load labels
    if 'labels' in data:
        y = data['labels']
    elif 'y' in data:
        y = data['y']
    else:
        print("❌ No labels found")
        return None, None, None, None, None, None
    
    # Load or create delta_t
    if 'delta_t' in data:
        delta_t = data['delta_t']
        if delta_t.ndim == 3:
            delta_t = delta_t.squeeze(1)
    elif 'timestamps' in data:
        timestamps = data['timestamps']
        if timestamps.ndim == 1:
            timestamps = np.tile(timestamps, (len(flux), 1))
        delta_t = create_delta_t_from_timestamps(timestamps)
    else:
        # Create uniform delta_t
        n_points = flux.shape[1]
        dt = 200.0 / n_points
        delta_t = np.full_like(flux, dt)
        delta_t[:, 0] = 0.0
    
    # Get timestamps
    if 'timestamps' in data:
        timestamps = data['timestamps']
        if timestamps.ndim == 1:
            timestamps = np.tile(timestamps, (len(flux), 1))
    else:
        timestamps = np.linspace(-100, 100, flux.shape[1])
        timestamps = np.tile(timestamps, (len(flux), 1))
    
    n_points = flux.shape[1]
    
    print(f"✅ Data loaded ({len(flux)} events, n_points={n_points})")
    
    return model, flux, delta_t, y, timestamps, n_points


def visualize_attention_patterns(model, flux, delta_t, y, timestamps, event_idx, 
                                 output_dir, device='cuda'):
    """Visualize attention patterns across all layers"""
    
    event_flux = flux[event_idx]
    event_delta = delta_t[event_idx]
    event_times = timestamps[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    # Prepare input
    flux_tensor = torch.tensor(event_flux, dtype=torch.float32).unsqueeze(0).to(device)
    delta_tensor = torch.tensor(event_delta, dtype=torch.float32).unsqueeze(0).to(device)
    length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
    length_tensor = torch.tensor([length], dtype=torch.long).to(device)
    
    # Get attention weights
    capture = AttentionCapture(model)
    attention_weights = capture.get_attention_weights(flux_tensor, delta_tensor, length_tensor)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(flux_tensor, delta_tensor, length_tensor, return_all_timesteps=False)
        probs = outputs['probs'].cpu().numpy()[0]
    
    # Find valid observations
    valid_mask = (event_flux != -1.0) & np.isfinite(event_flux)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    # Create figure
    num_layers = len(attention_weights)
    fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 10))
    
    if num_layers == 1:
        axes = axes.reshape(2, 1)
    
    for layer_idx, attn in enumerate(attention_weights):
        # Average over heads and batch
        attn_mat = attn[0].mean(dim=0).cpu().numpy()  # [T, T]
        
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
        
        # Add causal boundary
        ax.plot([0, n_valid], [0, n_valid], 'b--', linewidth=2, 
               label='Causal Boundary', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
        
        # Bottom row: Average attention per timestep
        ax = axes[1, layer_idx]
        
        # Average attention received by each position
        avg_attn_received = attn_mat[:, valid_indices].mean(axis=0)
        
        ax.plot(event_times[valid_indices], avg_attn_received, 'o-', 
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
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def visualize_temporal_encoding(model, flux, delta_t, y, timestamps, event_idx,
                                output_dir, device='cuda'):
    """Visualize temporal encoding analysis"""
    
    event_flux = flux[event_idx]
    event_delta = delta_t[event_idx]
    event_times = timestamps[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    valid_mask = (event_flux != -1.0) & np.isfinite(event_flux)
    valid_times = event_times[valid_mask]
    valid_flux = event_flux[valid_mask]
    valid_delta = event_delta[valid_mask]
    
    # Get temporal encoding
    delta_tensor = torch.tensor(valid_delta, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        temporal_enc = model.temporal_encoding(delta_tensor)
        temporal_enc = temporal_enc.cpu().numpy()[0]  # [N, D]
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Light curve
    ax1 = fig.add_subplot(gs[0, :])
    
    # Convert to magnitudes
    baseline = 20.0
    magnitudes = baseline - 2.5 * np.log10(np.maximum(valid_flux, 1e-10))
    
    color = ['gray', 'darkred', 'darkblue'][true_label]
    ax1.scatter(valid_times, magnitudes, c=color, s=15, alpha=0.7,
               edgecolors='black', linewidth=0.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('Time (days)', fontweight='bold')
    ax1.set_ylabel('Magnitude', fontweight='bold')
    ax1.set_title(f'Light Curve - {class_names[true_label]} Event',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # 2. Time deltas
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(valid_times, valid_delta, 'o-', linewidth=2, markersize=4, color='purple')
    ax2.set_xlabel('Time (days)', fontweight='bold')
    ax2.set_ylabel('Δt (days)', fontweight='bold')
    ax2.set_title('Observation Time Deltas', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=valid_delta.mean(), color='orange', linestyle='--', 
               linewidth=2, alpha=0.5, label=f'Mean: {valid_delta.mean():.3f}d')
    ax2.legend()
    
    # 3. Temporal encoding magnitude
    ax3 = fig.add_subplot(gs[1, 1])
    encoding_norm = np.linalg.norm(temporal_enc, axis=1)
    ax3.plot(valid_times, encoding_norm, 'o-', linewidth=2, markersize=4, 
            color='darkgreen')
    ax3.set_xlabel('Time (days)', fontweight='bold')
    ax3.set_ylabel('Encoding Magnitude', fontweight='bold')
    ax3.set_title('Temporal Encoding Strength', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Temporal encoding heatmap
    ax4 = fig.add_subplot(gs[2, :])
    
    # Show first 64 dimensions
    n_dims = min(64, temporal_enc.shape[1])
    im = ax4.imshow(temporal_enc[:, :n_dims].T, cmap='RdBu_r', aspect='auto',
                   vmin=-temporal_enc[:, :n_dims].max(), 
                   vmax=temporal_enc[:, :n_dims].max())
    ax4.set_xlabel('Time Step', fontweight='bold')
    ax4.set_ylabel('Encoding Dimension', fontweight='bold')
    ax4.set_title(f'Temporal Encoding (first {n_dims} dimensions)', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Activation')
    
    plt.suptitle(f'Temporal Encoding Analysis - Event {event_idx}',
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'temporal_encoding_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def visualize_classification_evolution(model, flux, delta_t, y, timestamps, event_idx,
                                       output_dir, n_points, device='cuda'):
    """Ultra-high-res classification evolution (100 points)"""
    
    event_flux = flux[event_idx]
    event_delta = delta_t[event_idx]
    event_times = timestamps[event_idx]
    true_label = y[event_idx]
    full_length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    # Ultra-high resolution: 100 fractions
    fractions = np.linspace(0.05, 1.0, 100)
    
    flat_probs, pspl_probs, binary_probs = [], [], []
    confidences = []
    
    with torch.no_grad():
        for frac in fractions:
            n_pts = max(1, int(full_length * frac))
            
            # Create partial observation
            partial_flux = torch.tensor(event_flux[:n_pts], dtype=torch.float32).unsqueeze(0).to(device)
            partial_delta = torch.tensor(event_delta[:n_pts], dtype=torch.float32).unsqueeze(0).to(device)
            partial_length = torch.tensor([n_pts], dtype=torch.long).to(device)
            
            outputs = model(partial_flux, partial_delta, partial_length, 
                          return_all_timesteps=False)
            probs = outputs['probs'].cpu().numpy()[0]
            
            flat_probs.append(probs[0])
            pspl_probs.append(probs[1])
            binary_probs.append(probs[2])
            confidences.append(probs.max())
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.2, 1], hspace=0.3)
    
    # Top: Light curve
    ax1 = fig.add_subplot(gs[0])
    valid_mask = (event_flux != -1.0) & np.isfinite(event_flux)
    times = event_times[valid_mask]
    fluxes = event_flux[valid_mask]
    
    # Convert to magnitudes
    baseline = 20.0
    magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
    
    color = ['gray', 'darkred', 'darkblue'][true_label]
    ax1.scatter(times, magnitudes, c=color, s=15, alpha=0.7,
               edgecolors='black', linewidth=0.5)
    ax1.invert_yaxis()
    ax1.set_ylabel('Magnitude', fontsize=13, fontweight='bold')
    ax1.set_title(f'ULTRA-HIGH-RES Evolution (100 Points) - True: {class_names[true_label]}',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Middle: Class probabilities
    ax2 = fig.add_subplot(gs[1])
    completeness = [f*100 for f in fractions]
    
    ax2.plot(completeness, flat_probs, '-', linewidth=1.5,
            color='gray', label='Flat', alpha=0.8)
    ax2.plot(completeness, pspl_probs, '-', linewidth=1.5,
            color='darkred', label='PSPL', alpha=0.8)
    ax2.plot(completeness, binary_probs, '-', linewidth=1.5,
            color='darkblue', label='Binary', alpha=0.8)
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, 
               label='50% Threshold')
    ax2.axhline(y=0.8, color='orange', linestyle=':', linewidth=1.5, 
               alpha=0.7, label='High Conf')
    
    ax2.set_ylabel('Class Probability', fontsize=13, fontweight='bold')
    ax2.legend(loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.05, 1.05])
    
    # Bottom: Confidence
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(completeness, confidences, '-', linewidth=1.5,
            color='purple', label='Confidence')
    ax3.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80%')
    ax3.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90%')
    ax3.set_xlabel('Observation Completeness (%)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Prediction Confidence', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.3, 1.05])
    
    plt.suptitle(f'ULTRA-HIGH-RES Classification Evolution (100 Points)',
                fontsize=15, fontweight='bold')
    
    output_path = output_dir / f'ultrahighres_evolution_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def visualize_embedding_space(model, flux, delta_t, y, output_dir, 
                              n_samples=500, device='cuda'):
    """Visualize hidden state embedding space using PCA"""
    from sklearn.decomposition import PCA
    
    print("  Computing embeddings...")
    
    # Sample events
    n_per_class = n_samples // 3
    indices = []
    for c in range(3):
        class_indices = np.where(y == c)[0]
        selected = np.random.choice(class_indices, 
                                   min(n_per_class, len(class_indices)),
                                   replace=False)
        indices.extend(selected)
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="    Extracting embeddings"):
            event_flux = torch.tensor(flux[idx], dtype=torch.float32).unsqueeze(0).to(device)
            event_delta = torch.tensor(delta_t[idx], dtype=torch.float32).unsqueeze(0).to(device)
            length = compute_lengths_from_flux(flux[idx][np.newaxis, :])[0]
            length_tensor = torch.tensor([length], dtype=torch.long).to(device)
            
            # Get final hidden state
            x = model.flux_embedding(event_flux.unsqueeze(-1))
            temporal = model.temporal_encoding(event_delta)
            x = x + temporal
            
            padding_mask = model.create_padding_mask_from_lengths(length_tensor, flux.shape[1])
            
            for layer in model.layers:
                x, _ = layer(x, padding_mask=padding_mask, return_cache=False)
            
            x = model.norm(x)
            
            # Get final timestep
            final_hidden = x[0, length-1].cpu().numpy()
            embeddings.append(final_hidden)
            labels.append(y[idx])
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print("  Reducing dimensionality with PCA...")
    
    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    class_names = ['Flat', 'PSPL', 'Binary']
    colors = ['gray', 'darkred', 'darkblue']
    markers = ['o', '^', 's']
    
    for c, name, color, marker in zip(range(3), class_names, colors, markers):
        mask = labels == c
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=color, label=name, alpha=0.6, s=50, marker=marker,
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Final Hidden State Embedding Space (PCA)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'embedding_space_pca.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def compare_binary_vs_pspl(model, flux, delta_t, y, timestamps, output_dir, device='cuda'):
    """Compare attention and features for Binary vs PSPL events"""
    
    # Find good examples
    binary_indices = np.where(y == 2)[0]
    pspl_indices = np.where(y == 1)[0]
    
    if len(binary_indices) == 0 or len(pspl_indices) == 0:
        print("  ⚠️  Skipping comparison (missing class)")
        return
    
    binary_idx = binary_indices[0]
    pspl_idx = pspl_indices[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for row, (idx, event_type) in enumerate([(binary_idx, 'Binary'), (pspl_idx, 'PSPL')]):
        event_flux = flux[idx]
        event_delta = delta_t[idx]
        event_times = timestamps[idx]
        
        # Prepare input
        flux_tensor = torch.tensor(event_flux, dtype=torch.float32).unsqueeze(0).to(device)
        delta_tensor = torch.tensor(event_delta, dtype=torch.float32).unsqueeze(0).to(device)
        length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
        length_tensor = torch.tensor([length], dtype=torch.long).to(device)
        
        # Get attention weights
        capture = AttentionCapture(model)
        attention_weights = capture.get_attention_weights(flux_tensor, delta_tensor, length_tensor)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(flux_tensor, delta_tensor, length_tensor, 
                          return_all_timesteps=False)
            probs = outputs['probs'].cpu().numpy()[0]
        
        valid_mask = (event_flux != -1.0) & np.isfinite(event_flux)
        times = event_times[valid_mask]
        fluxes = event_flux[valid_mask]
        
        # Convert to magnitudes
        baseline = 20.0
        magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
        
        # Column 1: Light curve
        ax = axes[row, 0]
        color = 'darkblue' if event_type == 'Binary' else 'darkred'
        ax.scatter(times, magnitudes, c=color, s=15, alpha=0.7,
                  edgecolors='black', linewidth=0.5)
        ax.invert_yaxis()
        ax.set_ylabel('Magnitude', fontweight='bold')
        ax.set_title(f'{event_type} Event',
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
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def visualize_confidence_evolution(model, flux, delta_t, y, output_dir, 
                                   n_events=50, device='cuda'):
    """Visualize how confidence evolves with observation completeness"""
    
    print("  Computing confidence evolution...")
    
    # Sample events from each class
    n_per_class = n_events // 3
    indices = []
    for c in range(3):
        class_indices = np.where(y == c)[0]
        selected = np.random.choice(class_indices, 
                                   min(n_per_class, len(class_indices)),
                                   replace=False)
        indices.extend(selected)
    
    fractions = np.linspace(0.1, 1.0, 20)
    
    all_confidences = {c: [] for c in range(3)}
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="    Processing events"):
            event_flux = flux[idx]
            event_delta = delta_t[idx]
            event_label = y[idx]
            full_length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
            
            confidences = []
            
            for frac in fractions:
                n_pts = max(1, int(full_length * frac))
                
                partial_flux = torch.tensor(event_flux[:n_pts], dtype=torch.float32).unsqueeze(0).to(device)
                partial_delta = torch.tensor(event_delta[:n_pts], dtype=torch.float32).unsqueeze(0).to(device)
                partial_length = torch.tensor([n_pts], dtype=torch.long).to(device)
                
                outputs = model(partial_flux, partial_delta, partial_length, 
                              return_all_timesteps=False)
                conf = outputs['confidence'].cpu().numpy()[0]
                confidences.append(conf)
            
            all_confidences[event_label].append(confidences)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    class_names = ['Flat', 'PSPL', 'Binary']
    colors = ['gray', 'darkred', 'darkblue']
    
    completeness = [f*100 for f in fractions]
    
    for c, name, color in zip(range(3), class_names, colors):
        if len(all_confidences[c]) > 0:
            confs = np.array(all_confidences[c])
            mean_conf = confs.mean(axis=0)
            std_conf = confs.std(axis=0)
            
            ax.plot(completeness, mean_conf, '-', linewidth=2.5, 
                   color=color, label=name)
            ax.fill_between(completeness, mean_conf - std_conf, mean_conf + std_conf,
                           color=color, alpha=0.2)
    
    ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, 
              alpha=0.5, label='High Confidence (80%)')
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, 
              alpha=0.5, label='Very High (90%)')
    
    ax.set_xlabel('Observation Completeness (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Classification Confidence', fontsize=13, fontweight='bold')
    ax.set_title('Confidence Evolution by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 1.05])
    ax.set_xlim([0, 105])
    
    plt.tight_layout()
    
    output_path = output_dir / 'confidence_evolution_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Transformer v1.0 internals',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--experiment_name', required=True, 
                       help='Experiment to visualize')
    parser.add_argument('--data', required=True, 
                       help='Test dataset path')
    parser.add_argument('--output_dir', default='../results/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--event_indices', nargs='+', type=int, default=None,
                       help='Specific event indices to visualize (default: auto-select)')
    parser.add_argument('--n_examples', type=int, default=2,
                       help='Number of examples per class if auto-selecting')
    parser.add_argument('--no_attention', action='store_true',
                       help='Skip attention visualization (faster)')
    parser.add_argument('--no_embedding', action='store_true',
                       help='Skip embedding space visualization')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Force CPU')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.no_cuda else 'cuda'
    
    # Load model and data
    result = load_model_and_data(args.experiment_name, args.data, device=device)
    if result[0] is None:
        return
    
    model, flux, delta_t, y, timestamps, n_points = result
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("TRANSFORMER VISUALIZATION v1.0")
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
            if len(class_indices) > 0:
                selected = np.random.choice(class_indices, 
                                           min(args.n_examples, len(class_indices)),
                                           replace=False)
                event_indices.extend(selected)
    
    print(f"Visualizing {len(event_indices)} events...\n")
    
    # Generate visualizations for individual events
    for i, event_idx in enumerate(event_indices):
        print(f"\n{'─'*70}")
        print(f"Event {i+1}/{len(event_indices)} (index {event_idx}):")
        print(f"  True class: {['Flat', 'PSPL', 'Binary'][y[event_idx]]}")
        print(f"{'─'*70}")
        
        # 1. Attention patterns
        if not args.no_attention:
            print("  → Attention patterns...")
            try:
                visualize_attention_patterns(model, flux, delta_t, y, timestamps, 
                                            event_idx, output_dir, device)
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
        
        # 2. Temporal encoding
        print("  → Temporal encoding...")
        try:
            visualize_temporal_encoding(model, flux, delta_t, y, timestamps,
                                       event_idx, output_dir, device)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
        
        # 3. Classification evolution
        print("  → Classification evolution (ULTRA-HIGH-RES)...")
        try:
            visualize_classification_evolution(model, flux, delta_t, y, timestamps,
                                              event_idx, output_dir, n_points, device)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
    
    # 4. Global visualizations
    print(f"\n{'─'*70}")
    print("Global Visualizations")
    print(f"{'─'*70}")
    
    # Binary vs PSPL comparison
    print("  → Binary vs PSPL comparison...")
    try:
        compare_binary_vs_pspl(model, flux, delta_t, y, timestamps, output_dir, device)
    except Exception as e:
        print(f"    ⚠️  Error: {e}")
    
    # Embedding space
    if not args.no_embedding:
        print("  → Embedding space (PCA)...")
        try:
            visualize_embedding_space(model, flux, delta_t, y, output_dir, 
                                     n_samples=500, device=device)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
    
    # Confidence evolution
    print("  → Confidence evolution...")
    try:
        visualize_confidence_evolution(model, flux, delta_t, y, output_dir,
                                      n_events=50, device=device)
    except Exception as e:
        print(f"    ⚠️  Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()