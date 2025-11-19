#!/usr/bin/env python3
"""
Transformer Visualization Suite

Comprehensive visualization tools for MicrolensingTransformer v1.0

Features:
- Attention pattern analysis across transformer layers
- Classification probability evolution over observation completeness
- Temporal encoding representation
- Confidence progression analysis
- Comparative analysis (Binary vs PSPL discrimination)
- Embedding space visualization (PCA)

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
    """
    Compute time intervals from timestamps.
    
    Args:
        timestamps: [N] or [B, N] array of timestamps
        
    Returns:
        delta_t: Same shape as input, delta_t[i] = timestamps[i] - timestamps[i-1]
    """
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    
    return delta_t


def compute_lengths_from_flux(flux, pad_value=-1.0):
    """
    Compute valid sequence lengths from padded flux arrays.
    
    Args:
        flux: [B, N] array with padding
        pad_value: Value used for padding
        
    Returns:
        lengths: [B] array of valid lengths
    """
    return np.sum(flux != pad_value, axis=1).astype(np.int64)


class AttentionCapture:
    """
    Hook-based attention weight extraction for visualization.
    
    Captures attention matrices from all transformer layers by directly
    computing Q, K, V and attention scores during forward pass.
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self):
        """Register forward hooks to capture attention weights."""
        self.attention_weights = []
        
        # Register hooks on each transformer layer's attention module
        for i, layer in enumerate(self.model.layers):
            hook = layer.attention.register_forward_hook(
                self._create_attention_hook(i)
            )
            self.hooks.append(hook)
    
    def _create_attention_hook(self, layer_idx):
        """Create a hook that captures attention weights."""
        def hook(module, input, output):
            # Attention weights must be captured manually
            # since they're not returned by the attention module
            pass
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_weights(self, flux, delta_t, lengths):
        """
        Extract attention weights by manually computing attention.
        
        Since the architecture doesn't expose attention weights directly,
        we compute them by accessing the attention module's internal projections.
        
        Args:
            flux: [B, N] tensor
            delta_t: [B, N] tensor
            lengths: [B] tensor
            
        Returns:
            attention_weights: List of [B, H, N, N] tensors for each layer
        """
        self.attention_weights = []
        
        with torch.no_grad():
            # Embed inputs
            x = self.model.flux_embedding(flux.unsqueeze(-1))
            temporal = self.model.temporal_encoding(delta_t)
            x = x + temporal
            
            # Create padding mask
            if lengths is not None:
                padding_mask = self.model.create_padding_mask_from_lengths(
                    lengths, flux.size(1)
                )
            else:
                padding_mask = None
            
            # Process each layer and capture attention
            for layer in self.model.layers:
                # Get attention module
                attn_module = layer.attention
                
                # Compute Q, K, V projections
                x_normed = layer.norm1(x)
                B, N, D = x_normed.shape
                
                qkv = attn_module.qkv_proj(x_normed)
                qkv = qkv.reshape(B, N, 3, attn_module.n_heads, attn_module.d_head)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) * attn_module.scale
                
                # Apply causal mask
                causal_mask = attn_module._create_causal_window_mask(
                    N, N, x.device
                )
                scores = scores.masked_fill(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), 
                    float('-inf')
                )
                
                # Apply padding mask
                if padding_mask is not None:
                    key_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~key_mask, float('-inf'))
                
                # Compute attention probabilities
                attn_weights = F.softmax(scores, dim=-1)
                
                # Store for visualization
                self.attention_weights.append(attn_weights)
                
                # Continue forward pass
                attn_out, _ = layer.attention(
                    x_normed, 
                    padding_mask=padding_mask, 
                    return_cache=False
                )
                x = x + attn_out
                x = x + layer.ffn(layer.norm2(x))
        
        return self.attention_weights


def load_model_and_data(experiment_name, data_path, device='cuda'):
    """
    Load trained model and test data for visualization.
    
    Args:
        experiment_name: Name of experiment directory
        data_path: Path to .npz data file
        device: Device to load model on
        
    Returns:
        model, flux, delta_t, labels, timestamps, n_points
    """
    # Find experiment directory
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f'{experiment_name}_*'))
    
    if not exp_dirs:
        print(f"ERROR: No experiment found matching '{experiment_name}'")
        return None, None, None, None, None, None
    
    exp_dir = exp_dirs[-1]
    print(f"Loading experiment: {exp_dir.name}")
    
    # Import model classes
    sys.path.insert(0, str(Path(__file__).parent))
    from transformer import MicrolensingTransformer, ModelConfig, count_parameters
    
    # Find model checkpoint
    model_path = exp_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = exp_dir / 'final_model.pt'
    
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found in {exp_dir}")
        return None, None, None, None, None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Reconstruct model configuration
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
            print(f"ERROR: Configuration file not found")
            return None, None, None, None, None, None
    
    # Create model instance
    model = MicrolensingTransformer(config)
    
    # Load state dict (handle DDP wrapper)
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # Load temperature calibration if available
    if 'calibration' in checkpoint and 'temperature' in checkpoint['calibration']:
        model.set_temperature(checkpoint['calibration']['temperature'])
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {count_parameters(model):,} parameters")
    
    # Load test data
    data = np.load(data_path)
    
    # Extract flux
    if 'flux' in data:
        flux = data['flux']
    elif 'X' in data:
        flux = data['X']
    else:
        print("ERROR: No flux data found in dataset")
        return None, None, None, None, None, None
    
    if flux.ndim == 3:
        flux = flux.squeeze(1)
    
    # Extract labels
    if 'labels' in data:
        y = data['labels']
    elif 'y' in data:
        y = data['y']
    else:
        print("ERROR: No labels found in dataset")
        return None, None, None, None, None, None
    
    # Extract or reconstruct delta_t
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
        # Create uniform time intervals as fallback
        n_points = flux.shape[1]
        dt = 200.0 / n_points
        delta_t = np.full_like(flux, dt)
        delta_t[:, 0] = 0.0
    
    # Extract or reconstruct timestamps
    if 'timestamps' in data:
        timestamps = data['timestamps']
        if timestamps.ndim == 1:
            timestamps = np.tile(timestamps, (len(flux), 1))
    else:
        timestamps = np.linspace(-100, 100, flux.shape[1])
        timestamps = np.tile(timestamps, (len(flux), 1))
    
    n_points = flux.shape[1]
    
    print(f"Data loaded: {len(flux)} events, {n_points} timesteps per event")
    
    return model, flux, delta_t, y, timestamps, n_points


def visualize_attention_patterns(model, flux, delta_t, y, timestamps, event_idx, 
                                 output_dir, device='cuda'):
    """
    Visualize attention patterns across all transformer layers.
    
    Generates:
    - Attention matrices for each layer (causal structure visible)
    - Average attention received by each temporal position
    """
    
    event_flux = flux[event_idx]
    event_delta = delta_t[event_idx]
    event_times = timestamps[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    # Prepare tensors
    flux_tensor = torch.tensor(event_flux, dtype=torch.float32).unsqueeze(0).to(device)
    delta_tensor = torch.tensor(event_delta, dtype=torch.float32).unsqueeze(0).to(device)
    length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
    length_tensor = torch.tensor([length], dtype=torch.long).to(device)
    
    # Extract attention weights
    capture = AttentionCapture(model)
    attention_weights = capture.get_attention_weights(
        flux_tensor, delta_tensor, length_tensor
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(flux_tensor, delta_tensor, length_tensor, 
                       return_all_timesteps=False)
        probs = outputs['probs'].cpu().numpy()[0]
    
    # Identify valid observations
    valid_mask = (event_flux != -1.0) & np.isfinite(event_flux)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    # Create visualization
    num_layers = len(attention_weights)
    fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 10))
    
    if num_layers == 1:
        axes = axes.reshape(2, 1)
    
    for layer_idx, attn in enumerate(attention_weights):
        # Average over attention heads and batch dimension
        attn_mat = attn[0].mean(dim=0).cpu().numpy()  # [N, N]
        
        # Top row: Attention matrix heatmap
        ax = axes[0, layer_idx]
        
        # Extract valid-to-valid attention only
        attn_valid = attn_mat[valid_indices][:, valid_indices]
        
        im = ax.imshow(attn_valid, cmap='hot', aspect='auto', 
                      vmin=0, vmax=attn_valid.max())
        ax.set_title(f'Layer {layer_idx+1}\nAttention Matrix', fontweight='bold')
        ax.set_xlabel('Source Position')
        ax.set_ylabel('Target Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Overlay causal boundary
        ax.plot([0, n_valid], [0, n_valid], 'b--', linewidth=2, 
               label='Causal Boundary', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
        
        # Bottom row: Average attention across time
        ax = axes[1, layer_idx]
        
        # Average attention received by each position
        avg_attn_received = attn_mat[:, valid_indices].mean(axis=0)
        
        ax.plot(event_times[valid_indices], avg_attn_received, 'o-', 
               linewidth=2, markersize=4, color='darkblue')
        ax.set_xlabel('Time (days)', fontweight='bold')
        ax.set_ylabel('Average Attention Received', fontweight='bold')
        ax.set_title(f'Layer {layer_idx+1}\nTemporal Attention Distribution', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, 
                  alpha=0.5, label='t=0')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(
        f'Attention Patterns - True: {class_names[true_label]}, '
        f'Predicted: {class_names[probs.argmax()]} ({probs.max()*100:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    output_path = output_dir / f'attention_patterns_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def visualize_temporal_encoding(model, flux, delta_t, y, timestamps, event_idx,
                                output_dir, device='cuda'):
    """
    Visualize temporal encoding representation.
    
    Generates:
    - Light curve in magnitude space
    - Time interval distribution
    - Temporal encoding embedding (first 6 dimensions)
    - PCA projection of temporal encoding
    """
    
    event_flux = flux[event_idx]
    event_delta = delta_t[event_idx]
    event_times = timestamps[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    valid_mask = (event_flux != -1.0) & np.isfinite(event_flux)
    valid_times = event_times[valid_mask]
    valid_flux = event_flux[valid_mask]
    valid_delta = event_delta[valid_mask]
    
    # Extract temporal encoding
    delta_tensor = torch.tensor(valid_delta, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        temporal_enc = model.temporal_encoding(delta_tensor)
        temporal_enc = temporal_enc.cpu().numpy()[0]  # [N, D]
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Light curve in magnitude space
    ax1 = fig.add_subplot(gs[0, :])
    
    baseline = 20.0
    magnitudes = baseline - 2.5 * np.log10(np.maximum(valid_flux, 1e-10))
    
    ax1.plot(valid_times, magnitudes, 'o-', linewidth=1.5, markersize=3, color='black')
    ax1.set_xlabel('Time (days)', fontweight='bold')
    ax1.set_ylabel('Magnitude', fontweight='bold')
    ax1.set_title(f'Light Curve - {class_names[true_label]}', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Panel 2: Time interval histogram
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.hist(valid_delta[valid_delta > 0], bins=30, color='steelblue', 
            edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Δt (days)', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Observation Interval Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Cumulative time intervals
    ax3 = fig.add_subplot(gs[1, 1])
    
    cumulative_time = np.cumsum(valid_delta)
    ax3.plot(range(len(cumulative_time)), cumulative_time, 'o-', 
            linewidth=2, markersize=3, color='darkgreen')
    ax3.set_xlabel('Observation Index', fontweight='bold')
    ax3.set_ylabel('Cumulative Time (days)', fontweight='bold')
    ax3.set_title('Temporal Sampling Pattern', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Temporal encoding dimensions (first 6)
    ax4 = fig.add_subplot(gs[2, 0])
    
    for i in range(min(6, temporal_enc.shape[1])):
        ax4.plot(valid_times, temporal_enc[:, i], 
                label=f'Dim {i+1}', linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('Time (days)', fontweight='bold')
    ax4.set_ylabel('Encoding Value', fontweight='bold')
    ax4.set_title('Temporal Encoding (First 6 Dimensions)', fontweight='bold')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: PCA of temporal encoding
    ax5 = fig.add_subplot(gs[2, 1])
    
    from sklearn.decomposition import PCA
    
    if temporal_enc.shape[0] > 2:
        pca = PCA(n_components=2)
        temporal_pca = pca.fit_transform(temporal_enc)
        
        scatter = ax5.scatter(temporal_pca[:, 0], temporal_pca[:, 1], 
                            c=valid_times, cmap='viridis', s=50, 
                            edgecolor='black', linewidth=0.5)
        
        ax5.set_xlabel('PC1', fontweight='bold')
        ax5.set_ylabel('PC2', fontweight='bold')
        ax5.set_title('Temporal Encoding (PCA Projection)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Time (days)', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Insufficient observations for PCA', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.axis('off')
    
    plt.suptitle(f'Temporal Encoding Analysis - Event {event_idx}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'temporal_encoding_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def visualize_classification_evolution(model, flux, delta_t, y, timestamps, 
                                       event_idx, output_dir, n_points, device='cuda'):
    """
    Visualize how classification probabilities evolve with observation completeness.
    
    Uses high-resolution sampling (100 evaluation points) to capture fine-grained
    probability dynamics.
    """
    
    event_flux = flux[event_idx]
    event_delta = delta_t[event_idx]
    event_times = timestamps[event_idx]
    true_label = y[event_idx]
    class_names = ['Flat', 'PSPL', 'Binary']
    
    # Compute valid length
    full_length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
    
    # High-resolution evaluation (100 points)
    n_eval_points = 100
    eval_indices = np.linspace(1, full_length, n_eval_points, dtype=int)
    
    probs_history = []
    confidence_history = []
    
    print(f"    Computing classification evolution ({n_eval_points} points)...")
    
    with torch.no_grad():
        for n_obs in eval_indices:
            # Create partial observation
            partial_flux = torch.tensor(
                event_flux[:n_obs], dtype=torch.float32
            ).unsqueeze(0).to(device)
            partial_delta = torch.tensor(
                event_delta[:n_obs], dtype=torch.float32
            ).unsqueeze(0).to(device)
            partial_length = torch.tensor([n_obs], dtype=torch.long).to(device)
            
            # Get predictions
            outputs = model(partial_flux, partial_delta, partial_length,
                          return_all_timesteps=False)
            
            probs = outputs['probs'].cpu().numpy()[0]
            confidence = outputs['confidence'].cpu().numpy()[0]
            
            probs_history.append(probs)
            confidence_history.append(confidence)
    
    probs_history = np.array(probs_history)
    confidence_history = np.array(confidence_history)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Observation completeness axis
    completeness = (eval_indices / full_length) * 100
    
    # Panel 1: Class probabilities
    ax1 = axes[0]
    
    colors = ['gray', 'darkred', 'darkblue']
    for c, name, color in zip(range(3), class_names, colors):
        ax1.plot(completeness, probs_history[:, c], linewidth=2.5, 
                color=color, label=name, alpha=0.8)
    
    # Highlight true class
    ax1.axhline(y=1.0, color=colors[true_label], linestyle='--', 
               linewidth=1.5, alpha=0.3)
    
    ax1.set_xlabel('Observation Completeness (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Class Probability', fontweight='bold', fontsize=12)
    ax1.set_title('Classification Probability Evolution', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11, loc='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([0, 105])
    
    # Panel 2: Prediction confidence
    ax2 = axes[1]
    
    ax2.plot(completeness, confidence_history, linewidth=2.5, color='purple')
    ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='High Confidence (80%)')
    ax2.axhline(y=0.9, color='green', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='Very High (90%)')
    
    ax2.set_xlabel('Observation Completeness (%)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Prediction Confidence', fontweight='bold', fontsize=12)
    ax2.set_title('Confidence Evolution', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.2, 1.05])
    ax2.set_xlim([0, 105])
    
    # Panel 3: Light curve with observation markers
    ax3 = axes[2]
    
    valid_mask = (event_flux != -1.0) & np.isfinite(event_flux)
    valid_times = event_times[valid_mask]
    valid_flux = event_flux[valid_mask]
    
    baseline = 20.0
    magnitudes = baseline - 2.5 * np.log10(np.maximum(valid_flux, 1e-10))
    
    ax3.plot(valid_times, magnitudes, 'o-', linewidth=1.5, markersize=4, 
            color='black', alpha=0.7)
    ax3.set_xlabel('Time (days)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Magnitude', fontweight='bold', fontsize=12)
    ax3.set_title(f'Light Curve - True Class: {class_names[true_label]}', 
                 fontweight='bold', fontsize=13)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='t=0')
    ax3.legend(fontsize=10)
    
    plt.suptitle(f'Classification Evolution - Event {event_idx}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'classification_evolution_event{event_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def visualize_embedding_space(model, flux, delta_t, y, output_dir, 
                              n_samples=500, device='cuda'):
    """
    Visualize final layer embedding space using PCA.
    
    Projects high-dimensional embeddings to 2D for visualization of class clustering.
    """
    
    from sklearn.decomposition import PCA
    
    print(f"    Extracting embeddings from {n_samples} events...")
    
    # Sample events
    if n_samples < len(flux):
        indices = np.random.choice(len(flux), n_samples, replace=False)
    else:
        indices = np.arange(len(flux))
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="    Processing"):
            event_flux = flux[idx]
            event_delta = delta_t[idx]
            event_label = y[idx]
            
            length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
            
            flux_tensor = torch.tensor(
                event_flux[:length], dtype=torch.float32
            ).unsqueeze(0).to(device)
            delta_tensor = torch.tensor(
                event_delta[:length], dtype=torch.float32
            ).unsqueeze(0).to(device)
            length_tensor = torch.tensor([length], dtype=torch.long).to(device)
            
            # Extract embedding before classification head
            x = model.flux_embedding(flux_tensor.unsqueeze(-1))
            temporal = model.temporal_encoding(delta_tensor)
            x = x + temporal
            
            padding_mask = model.create_padding_mask_from_lengths(
                length_tensor, length
            )
            
            # Pass through transformer layers
            for layer in model.layers:
                attn_out, _ = layer.attention(layer.norm1(x), 
                                             padding_mask=padding_mask,
                                             return_cache=False)
                x = x + attn_out
                x = x + layer.ffn(layer.norm2(x))
            
            x = model.final_norm(x)
            
            # Global pooling
            if padding_mask is not None:
                x_masked = x.masked_fill(~padding_mask.unsqueeze(-1), 0)
                avg_pool = x_masked.sum(dim=1) / padding_mask.sum(dim=1, keepdim=True)
            else:
                avg_pool = x.mean(dim=1)
            
            max_pool, _ = x.max(dim=1)
            pooled = torch.cat([avg_pool, max_pool], dim=-1)
            
            embeddings.append(pooled.cpu().numpy()[0])
            labels.append(event_label)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # PCA projection
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))
    
    class_names = ['Flat', 'PSPL', 'Binary']
    colors = ['gray', 'darkred', 'darkblue']
    
    for c, name, color in zip(range(3), class_names, colors):
        mask = labels == c
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                  c=color, label=name, alpha=0.6, s=30, edgecolors='black', 
                  linewidths=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                 fontweight='bold', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                 fontweight='bold', fontsize=12)
    ax.set_title('Embedding Space Visualization (PCA)', 
                fontweight='bold', fontsize=14)
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'embedding_space_pca.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def compare_binary_vs_pspl(model, flux, delta_t, y, timestamps, output_dir, device='cuda'):
    """
    Comparative analysis of Binary vs PSPL discrimination.
    
    Samples representative events from Binary and PSPL classes and visualizes
    how their classification probabilities evolve differently.
    """
    
    print("    Selecting representative events...")
    
    # Sample events
    pspl_indices = np.where(y == 1)[0]
    binary_indices = np.where(y == 2)[0]
    
    n_per_class = 5
    selected_pspl = np.random.choice(pspl_indices, 
                                    min(n_per_class, len(pspl_indices)), 
                                    replace=False)
    selected_binary = np.random.choice(binary_indices, 
                                       min(n_per_class, len(binary_indices)), 
                                       replace=False)
    
    # Compute evolution for each event
    fractions = np.linspace(0.1, 1.0, 20)
    
    pspl_probs_binary = []
    binary_probs_binary = []
    
    with torch.no_grad():
        for idx in tqdm(list(selected_pspl) + list(selected_binary), 
                       desc="    Computing evolution"):
            event_flux = flux[idx]
            event_delta = delta_t[idx]
            full_length = compute_lengths_from_flux(event_flux[np.newaxis, :])[0]
            
            probs_binary_class = []
            
            for frac in fractions:
                n_pts = max(1, int(full_length * frac))
                
                partial_flux = torch.tensor(
                    event_flux[:n_pts], dtype=torch.float32
                ).unsqueeze(0).to(device)
                partial_delta = torch.tensor(
                    event_delta[:n_pts], dtype=torch.float32
                ).unsqueeze(0).to(device)
                partial_length = torch.tensor([n_pts], dtype=torch.long).to(device)
                
                outputs = model(partial_flux, partial_delta, partial_length,
                              return_all_timesteps=False)
                probs = outputs['probs'].cpu().numpy()[0]
                
                probs_binary_class.append(probs[2])  # Binary class probability
            
            if y[idx] == 1:
                pspl_probs_binary.append(probs_binary_class)
            else:
                binary_probs_binary.append(probs_binary_class)
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    completeness = [f*100 for f in fractions]
    
    # Panel 1: PSPL events (should stay low)
    ax1 = axes[0]
    
    pspl_probs_binary = np.array(pspl_probs_binary)
    for i in range(len(pspl_probs_binary)):
        ax1.plot(completeness, pspl_probs_binary[i], 'o-', 
                linewidth=1.5, alpha=0.6, color='darkred')
    
    mean_pspl = pspl_probs_binary.mean(axis=0)
    ax1.plot(completeness, mean_pspl, linewidth=3, color='black', 
            label='Mean', linestyle='--')
    
    ax1.set_xlabel('Observation Completeness (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Binary Class Probability', fontweight='bold', fontsize=12)
    ax1.set_title('True PSPL Events\n(Should Maintain Low Binary Probability)', 
                 fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([0, 105])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    
    # Panel 2: Binary events (should increase)
    ax2 = axes[1]
    
    binary_probs_binary = np.array(binary_probs_binary)
    for i in range(len(binary_probs_binary)):
        ax2.plot(completeness, binary_probs_binary[i], 'o-', 
                linewidth=1.5, alpha=0.6, color='darkblue')
    
    mean_binary = binary_probs_binary.mean(axis=0)
    ax2.plot(completeness, mean_binary, linewidth=3, color='black', 
            label='Mean', linestyle='--')
    
    ax2.set_xlabel('Observation Completeness (%)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Binary Class Probability', fontweight='bold', fontsize=12)
    ax2.set_title('True Binary Events\n(Should Increase Binary Probability)', 
                 fontweight='bold', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlim([0, 105])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    
    plt.suptitle('Binary vs PSPL Discrimination', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'binary_vs_pspl_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def visualize_confidence_evolution(model, flux, delta_t, y, output_dir, 
                                   n_events=50, device='cuda'):
    """
    Visualize confidence evolution across observation completeness by class.
    
    Shows how model confidence grows as more observations become available,
    stratified by true class label.
    """
    
    print("    Computing confidence evolution by class...")
    
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
                
                partial_flux = torch.tensor(
                    event_flux[:n_pts], dtype=torch.float32
                ).unsqueeze(0).to(device)
                partial_delta = torch.tensor(
                    event_delta[:n_pts], dtype=torch.float32
                ).unsqueeze(0).to(device)
                partial_length = torch.tensor([n_pts], dtype=torch.long).to(device)
                
                outputs = model(partial_flux, partial_delta, partial_length, 
                              return_all_timesteps=False)
                conf = outputs['confidence'].cpu().numpy()[0]
                confidences.append(conf)
            
            all_confidences[event_label].append(confidences)
    
    # Visualize
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
              alpha=0.5, label='Very High Confidence (90%)')
    
    ax.set_xlabel('Observation Completeness (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Classification Confidence', fontsize=13, fontweight='bold')
    ax.set_title('Confidence Evolution by True Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 1.05])
    ax.set_xlim([0, 105])
    
    plt.tight_layout()
    
    output_path = output_dir / 'confidence_evolution_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def main():
    """Main visualization workflow."""
    
    parser = argparse.ArgumentParser(
        description='Transformer v1.0 Visualization Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--experiment_name', required=True, 
                       help='Experiment directory name')
    parser.add_argument('--data', required=True, 
                       help='Test dataset path (.npz)')
    parser.add_argument('--output_dir', default='../results/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--event_indices', nargs='+', type=int, default=None,
                       help='Specific event indices to visualize')
    parser.add_argument('--n_examples', type=int, default=2,
                       help='Number of examples per class (if auto-selecting)')
    parser.add_argument('--no_attention', action='store_true',
                       help='Skip attention visualization (faster)')
    parser.add_argument('--no_embedding', action='store_true',
                       help='Skip embedding space visualization')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Force CPU execution')
    
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
    
    print("\n" + "="*70)
    print("TRANSFORMER VISUALIZATION SUITE v1.0")
    print("="*70 + "\n")
    
    # Select events to visualize
    if args.event_indices:
        event_indices = args.event_indices
    else:
        # Auto-select representative examples from each class
        event_indices = []
        for class_id in range(3):
            class_mask = y == class_id
            class_indices = np.where(class_mask)[0]
            if len(class_indices) > 0:
                selected = np.random.choice(class_indices, 
                                           min(args.n_examples, len(class_indices)),
                                           replace=False)
                event_indices.extend(selected)
    
    print(f"Visualizing {len(event_indices)} individual events\n")
    
    # Generate per-event visualizations
    for i, event_idx in enumerate(event_indices):
        print("\n" + "-"*70)
        print(f"Event {i+1}/{len(event_indices)} (index {event_idx})")
        print(f"  True class: {['Flat', 'PSPL', 'Binary'][y[event_idx]]}")
        print("-"*70)
        
        # Attention patterns
        if not args.no_attention:
            print("  Computing attention patterns...")
            try:
                visualize_attention_patterns(model, flux, delta_t, y, timestamps, 
                                            event_idx, output_dir, device)
            except Exception as e:
                print(f"  WARNING: Attention visualization failed: {e}")
        
        # Temporal encoding
        print("  Computing temporal encoding...")
        try:
            visualize_temporal_encoding(model, flux, delta_t, y, timestamps,
                                       event_idx, output_dir, device)
        except Exception as e:
            print(f"  WARNING: Temporal encoding visualization failed: {e}")
        
        # Classification evolution
        print("  Computing classification evolution (high resolution)...")
        try:
            visualize_classification_evolution(model, flux, delta_t, y, timestamps,
                                              event_idx, output_dir, n_points, device)
        except Exception as e:
            print(f"  WARNING: Evolution visualization failed: {e}")
    
    # Global visualizations
    print("\n" + "-"*70)
    print("Global Visualizations")
    print("-"*70)
    
    # Binary vs PSPL comparison
    print("  Computing Binary vs PSPL comparison...")
    try:
        compare_binary_vs_pspl(model, flux, delta_t, y, timestamps, output_dir, device)
    except Exception as e:
        print(f"  WARNING: Binary/PSPL comparison failed: {e}")
    
    # Embedding space
    if not args.no_embedding:
        print("  Computing embedding space (PCA)...")
        try:
            visualize_embedding_space(model, flux, delta_t, y, output_dir, 
                                     n_samples=500, device=device)
        except Exception as e:
            print(f"  WARNING: Embedding visualization failed: {e}")
    
    # Confidence evolution
    print("  Computing confidence evolution by class...")
    try:
        visualize_confidence_evolution(model, flux, delta_t, y, output_dir,
                                      n_events=50, device=device)
    except Exception as e:
        print(f"  WARNING: Confidence evolution failed: {e}")
    
    print("\n" + "="*70)
    print(f"All visualizations saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()