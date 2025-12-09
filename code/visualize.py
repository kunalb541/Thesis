#!/usr/bin/env python3
"""
Causal Hybrid Model Internal Visualization Suite (Fully Debugged)

Performs deep introspection of the Causal Hybrid Architecture using PyTorch forward hooks.
Generates diagnostic plots regarding attention mechanisms, embedding spaces, and temporal encoding.

Usage:
    python visualize.py --experiment_name "exp_01" --data "../data/test.npz" --n_examples 3
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

# --- Matplotlib Backend Setup ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dynamic Import Setup ---
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.append(str(current_dir.parent))

try:
    from transformer import CausalHybridModel, CausalConfig
    print("SUCCESS: Imported 'transformer.py'")
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import 'transformer.py'.")
    print(f"Ensure the model definition file is in {current_dir} or {current_dir.parent}")
    print(f"Python Error: {e}\n")
    sys.exit(1)

# Style Settings
try:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
except Exception:
    pass
plt.rcParams['figure.dpi'] = 150 

# =============================================================================
# UTILITIES
# =============================================================================

class HookManager:
    """Manages PyTorch forward hooks to capture intermediate layer outputs."""
    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def register_hook(self, module, name):
        def hook(model, input, output):
            target_tensor = None

            if isinstance(output, torch.Tensor):
                target_tensor = output
            elif isinstance(output, tuple):
                # For attention layers, look for square matrix (attention weights)
                if 'attn' in name:
                    found_weights = False
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            # Check if last two dims are equal (Square Matrix)
                            if item.ndim >= 2 and item.shape[-1] == item.shape[-2]:
                                target_tensor = item
                                found_weights = True
                                break
                    # Fallback: take first element (context vector)
                    if not found_weights and len(output) > 0:
                        target_tensor = output[0] if isinstance(output[0], torch.Tensor) else None
                else:
                    # Default for other layers
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        target_tensor = output[0]

            if target_tensor is not None:
                self.outputs[name] = target_tensor.detach().cpu()
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)

    def clear_data(self):
        self.outputs = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def interpolate_trajectory(times, values, n_points=100):
    """Normalizes time-series data to a fixed-length vector."""
    times = np.asarray(times)
    values = np.asarray(values)
    
    if len(times) < 2: 
        if len(values) > 0:
            return np.full(n_points, values[0])
        return np.zeros(n_points)
    
    t_min, t_max = times.min(), times.max()
    # Avoid division by zero
    if np.isclose(t_max, t_min):
        return np.full(n_points, values[0] if len(values) > 0 else 0.0)
    
    t_norm = (times - t_min) / (t_max - t_min)
    
    try:
        f = interp1d(t_norm, values, kind='linear', bounds_error=False, fill_value="extrapolate")
        x_new = np.linspace(0, 1, n_points)
        return f(x_new)
    except Exception as e:
        print(f"Warning: Interpolation failed ({e}), returning zeros.")
        return np.zeros(n_points)

def create_delta_t_from_timestamps(timestamps):
    """Create delta_t array from timestamps."""
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    delta_t = np.zeros_like(timestamps, dtype=np.float32)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    return np.maximum(delta_t, 0.0)

# =============================================================================
# DEEP VISUALIZER CLASS
# =============================================================================

class DeepVisualizer:
    def __init__(self, model_path, data_path, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Model
        print(f"Loading checkpoint: {Path(model_path).name}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
        
        # Config Reconstruction
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            valid_keys = CausalConfig().__dict__.keys()
            filtered_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
            self.config = CausalConfig(**filtered_conf)
        else:
            print("Warning: Config not found. Using defaults.")
            self.config = CausalConfig(d_model=128, n_heads=8, n_transformer_layers=2)
        
        print(f"Model config: d_model={self.config.d_model}, n_heads={self.config.n_heads}")
        
        self.model = CausalHybridModel(self.config).to(self.device)
        
        # Load Weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")
        except RuntimeError as e:
            print(f"Warning on weight loading: {e}")
            
        self.model.eval()
        print("Model loaded successfully.")

        # 2. Load Data
        print(f"Loading data: {Path(data_path).name}...")
        try:
            data = np.load(data_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading data file: {e}")
            sys.exit(1)

        # Try different key names
        self.flux = data.get('flux', data.get('X', None))
        self.y = data.get('labels', data.get('y', None))
        
        if self.flux is None or self.y is None:
            print(f"Invalid data keys. Found: {list(data.keys())}")
            sys.exit(1)
            
        # Fix Dimensions and Types - Ragged Array Safety
        if self.flux.dtype == np.object_:
            print("Warning: Detected ragged array. Attempting to stack...")
            try:
                # Try to stack if all same length
                self.flux = np.stack(self.flux).astype(np.float32)
            except (ValueError, TypeError):
                print("CRITICAL: Data is ragged (different lengths per sample).")
                print("This script expects padded numpy arrays.")
                sys.exit(1)
        
        # Ensure 2D
        if self.flux.ndim == 3:
            self.flux = self.flux.squeeze(1)
        elif self.flux.ndim == 1:
            print("Error: Flux data is 1D. Expected 2D array.")
            sys.exit(1)
            
        self.flux = self.flux.astype(np.float32)
        print(f"Data shape: {self.flux.shape}")

        # Time handling
        if 'delta_t' in data:
            self.delta_t = data['delta_t'].astype(np.float32)
            if self.delta_t.ndim == 3:
                self.delta_t = self.delta_t.squeeze(1)
        elif 'timestamps' in data:
            ts = data['timestamps']
            if ts.ndim == 1:
                ts = np.tile(ts, (len(self.flux), 1))
            self.delta_t = create_delta_t_from_timestamps(ts)
        else:
            print("Warning: No time information found. Using zeros.")
            self.delta_t = np.zeros_like(self.flux, dtype=np.float32)
            
        if self.delta_t.ndim == 3:
            self.delta_t = self.delta_t.squeeze(1)
        
        # Ensure delta_t matches flux shape
        if self.delta_t.shape != self.flux.shape:
            print(f"Warning: Reshaping delta_t from {self.delta_t.shape} to {self.flux.shape}")
            if self.delta_t.shape[0] == self.flux.shape[0]:
                # Repeat or truncate sequence dimension
                if self.delta_t.shape[1] < self.flux.shape[1]:
                    self.delta_t = np.pad(self.delta_t, 
                                         ((0, 0), (0, self.flux.shape[1] - self.delta_t.shape[1])),
                                         mode='constant')
                else:
                    self.delta_t = self.delta_t[:, :self.flux.shape[1]]
            else:
                self.delta_t = np.zeros_like(self.flux, dtype=np.float32)
        
        # Length Calculation
        is_padding = (self.flux == -1.0) | (np.isnan(self.flux)) | (self.flux == 0.0)
        self.lengths = np.sum(~is_padding, axis=1).astype(np.int64)
        self.lengths = np.maximum(self.lengths, 1)
        self.lengths = np.minimum(self.lengths, self.flux.shape[1])  # Cap at max sequence length
        
        print(f"Length range: {self.lengths.min()} - {self.lengths.max()}")
        
        self.hooks = HookManager()

    def attach_hooks(self):
        """Introspects the model structure and attaches hooks."""
        self.hooks.remove_hooks()
        print("Attaching internal hooks...")
        
        hook_count = 0
        
        # Try different model architectures
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                # Strategy 1: Custom attention module
                if hasattr(layer, 'attention'):
                    if hasattr(layer.attention, 'dropout'):
                        self.hooks.register_hook(layer.attention.dropout, f'attn_layer_{i}')
                        hook_count += 1
                    else:
                        self.hooks.register_hook(layer.attention, f'attn_layer_{i}')
                        hook_count += 1
                # Strategy 2: Standard TransformerEncoderLayer
                elif hasattr(layer, 'self_attn'):
                    self.hooks.register_hook(layer.self_attn, f'attn_layer_{i}')
                    hook_count += 1
        
        if hook_count == 0:
            print("Warning: No attention layers found to hook.")
        else:
            print(f"Hooked {hook_count} attention layers.")

        # Hook Final Embeddings
        if hasattr(self.model, 'final_norm'):
            self.hooks.register_hook(self.model.final_norm, 'final_embedding')
            print("Hooked final_norm for embeddings.")
        elif hasattr(self.model, 'ln_f'):
            self.hooks.register_hook(self.model.ln_f, 'final_embedding')
            print("Hooked ln_f for embeddings.")
        elif hasattr(self.model, 'norm'):
            self.hooks.register_hook(self.model.norm, 'final_embedding')
            print("Hooked norm for embeddings.")

    def visualize_event_internals(self, idx):
        """Generates diagnostic plots for a single event."""
        print(f"   Visualizing Event Index {idx}...")
        
        # Data Prep
        f_t = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d_t = torch.tensor(self.delta_t[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l_t = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        
        self.hooks.clear_data()
        
        try:
            with torch.no_grad():
                out = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
                
                # Extract probabilities
                if isinstance(out, dict) and 'probs' in out:
                    probs = out['probs'][0].cpu().numpy()
                elif isinstance(out, dict) and 'logits' in out:
                    probs = torch.softmax(out['logits'][0], dim=-1).cpu().numpy()
                elif isinstance(out, tuple):
                    t_out = out[0]
                    probs = torch.softmax(t_out[0], dim=-1).cpu().numpy()
                else:
                    probs = torch.softmax(out[0], dim=-1).cpu().numpy()
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return
        
        length = self.lengths[idx]
        valid_probs = probs[:length]
        
        # --- Plot 1: Classification Evolution ---
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            n_classes = valid_probs.shape[1]
            classes = ['Flat', 'PSPL', 'Binary'] if n_classes == 3 else [f'Class {i}' for i in range(n_classes)]
            colors = sns.color_palette("husl", n_classes)
            
            for i, (cls, col) in enumerate(zip(classes, colors)):
                ax.plot(valid_probs[:, i], label=cls, color=col, linewidth=2, alpha=0.8)
            
            ax.set_title(f"Classification Evolution: Event {idx} (True: {self.y[idx]})")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"classification_evolution_event_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting classification evolution: {e}")

        # --- Plot 2: Attention Patterns ---
        try:
            last_layer_idx = len(self.model.layers) - 1 if hasattr(self.model, 'layers') else 0
            attn_key = f'attn_layer_{last_layer_idx}'
            
            if attn_key in self.hooks.outputs:
                attn_data = self.hooks.outputs[attn_key]
                
                # Unbatch and handle shapes
                if attn_data.dim() == 4:  # [Batch, Heads, Seq, Seq]
                    attn_data = attn_data[0]
                elif attn_data.dim() == 3 and attn_data.shape[0] == 1:  # [1, Seq, Seq]
                    attn_data = attn_data[0]

                dims = attn_data.shape
                is_square_map = (len(dims) >= 2 and dims[-1] == dims[-2])
                
                if is_square_map:
                    if attn_data.dim() == 3:  # [Heads, Seq, Seq]
                        attn_avg = attn_data.mean(dim=0).numpy()
                    else:
                        attn_avg = attn_data.numpy()
                    
                    # Slice to valid length
                    h, w = attn_avg.shape
                    valid_h, valid_w = min(h, length), min(w, length)
                    attn_avg = attn_avg[:valid_h, :valid_w]

                    fig = plt.figure(figsize=(8, 7))
                    sns.heatmap(attn_avg, cmap='viridis', square=True, cbar_kws={'label': 'Attention Weight'})
                    plt.title(f"Mean Attention Map (Layer {last_layer_idx})")
                    plt.xlabel("Key Token")
                    plt.ylabel("Query Token")
                    plt.tight_layout()
                    plt.savefig(self.output_dir / f"attention_patterns_event_{idx}.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else:
                    print(f"   Note: Hooked tensor shape {dims} is not square. Likely context vector.")
            else:
                print(f"   Note: No attention data found for key {attn_key}")
        except Exception as e:
            print(f"Error plotting attention patterns: {e}")
        
        # --- Plot 3: Temporal Encoding ---
        try:
            if hasattr(self.model, 'temporal_encoding'):
                # Try different input formats
                pe = None
                try:
                    pe = self.model.temporal_encoding(d_t).detach().cpu().numpy()
                except Exception:
                    try:
                        pe = self.model.temporal_encoding(d_t.squeeze(-1)).detach().cpu().numpy()
                    except Exception:
                        try:
                            pe = self.model.temporal_encoding(d_t.unsqueeze(-1)).detach().cpu().numpy()
                        except Exception as e:
                            print(f"   Could not compute PE: {e}")

                if pe is not None:
                    pe = pe[0, :length]
                    dt_vals = self.delta_t[idx, :length]
                    
                    # Filter out zero/very small delta_t for log scale
                    valid_mask = dt_vals > 1e-6
                    if np.sum(valid_mask) > 1:
                        pe_mag = np.linalg.norm(pe, axis=1)
                        
                        fig = plt.figure(figsize=(8, 6))
                        scatter = plt.scatter(dt_vals[valid_mask], pe_mag[valid_mask], 
                                            c=np.arange(length)[valid_mask], 
                                            cmap='plasma', alpha=0.7, s=20)
                        plt.colorbar(scatter, label='Time Step Index')
                        plt.xlabel("Delta T")
                        plt.ylabel("Encoding Magnitude")
                        plt.title("Temporal Encoding Analysis")
                        plt.xscale('log')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.output_dir / f"temporal_encoding_event_{idx}.png", dpi=150, bbox_inches='tight')
                        plt.close(fig)
        except Exception as e:
            print(f"Error plotting temporal encoding: {e}")

    def analyze_embedding_space(self, n_samples=1000):
        """PCA of the learned embeddings."""
        print("Generating Embedding Space PCA...")
        n_samples = min(len(self.flux), n_samples)
        indices = np.random.choice(len(self.flux), n_samples, replace=False)
        
        embeddings = []
        labels = []
        
        self.attach_hooks()
        
        with torch.no_grad():
            for i in tqdm(indices, desc="Extracting Embeddings"):
                try:
                    f_t = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    d_t = torch.tensor(self.delta_t[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    l_t = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                    
                    self.hooks.clear_data()
                    _ = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
                    
                    if 'final_embedding' in self.hooks.outputs:
                        tensor_out = self.hooks.outputs['final_embedding']
                        # Get vector at last valid timestep
                        last_step = min(self.lengths[i] - 1, tensor_out.shape[1] - 1)
                        vec = tensor_out[0, last_step].numpy()
                        embeddings.append(vec)
                        labels.append(self.y[i])
                except Exception as e:
                    continue

        self.hooks.remove_hooks()

        if len(embeddings) < 5:
            print("Skipping PCA: Insufficient data.")
            return

        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        print(f"Performing PCA on {len(embeddings)} embeddings...")
        
        try:
            pca = PCA(n_components=2)
            emb_pca = pca.fit_transform(embeddings)
            
            fig = plt.figure(figsize=(10, 8))
            sns.scatterplot(x=emb_pca[:, 0], y=emb_pca[:, 1], hue=labels, 
                          palette='deep', alpha=0.7, s=30)
            plt.title("Latent Space PCA")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            plt.legend(title='Class')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "embedding_space_pca.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("PCA plot saved.")
        except Exception as e:
            print(f"PCA failed: {e}")

    def plot_confidence_evolution(self, n_samples=500):
        """Confidence trajectories."""
        print("Analyzing Confidence Evolution...")
        n_samples = min(len(self.flux), n_samples)
        indices = np.random.choice(len(self.flux), n_samples, replace=False)
        
        trajectories = {} 
        self.hooks.remove_hooks()

        with torch.no_grad():
            for i in tqdm(indices, desc="Processing Trajectories"):
                try:
                    f_t = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    d_t = torch.tensor(self.delta_t[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    l_t = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                    
                    out = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
                    
                    # Extract probabilities
                    if isinstance(out, dict) and 'probs' in out:
                        probs = out['probs'][0].cpu().numpy()
                    elif isinstance(out, dict) and 'logits' in out:
                        probs = torch.softmax(out['logits'][0], dim=-1).cpu().numpy()
                    elif isinstance(out, tuple):
                        logits = out[0][0]
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    else:
                        probs = torch.softmax(out[0], dim=-1).cpu().numpy()
                    
                    true_class = int(self.y[i])
                    if true_class >= probs.shape[1]:
                        continue

                    conf = probs[:self.lengths[i], true_class]
                    if len(conf) < 2:
                        continue
                        
                    traj = interpolate_trajectory(np.arange(len(conf)), conf)
                    
                    if true_class not in trajectories:
                        trajectories[true_class] = []
                    trajectories[true_class].append(traj)
                except Exception as e:
                    continue

        if not trajectories:
            print("No trajectories collected. Skipping plot.")
            return

        try:
            fig = plt.figure(figsize=(10, 6))
            for cls_idx in sorted(trajectories.keys()):
                data = trajectories[cls_idx]
                if not data:
                    continue
                arr = np.array(data)
                mean_conf = np.mean(arr, axis=0)
                std_conf = np.std(arr, axis=0)
                x = np.linspace(0, 100, 100)
                
                plt.plot(x, mean_conf, label=f'Class {cls_idx} (n={len(data)})', linewidth=2)
                plt.fill_between(x, mean_conf - std_conf, mean_conf + std_conf, alpha=0.2)
                
            plt.xlabel("% of Light Curve")
            plt.ylabel("Confidence")
            plt.legend()
            plt.title("Mean Confidence Trajectory per Class")
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(self.output_dir / "confidence_evolution.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("Confidence evolution plot saved.")
        except Exception as e:
            print(f"Error plotting confidence evolution: {e}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize Causal Hybrid Model Internals')
    parser.add_argument('--experiment_name', type=str, required=True, 
                       help='Name of experiment folder')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data file (.npz)')
    parser.add_argument('--n_examples', type=int, default=3,
                       help='Number of example events to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Locate Experiment
    possible_roots = [Path('../results'), Path('results'), Path('.')]
    exp_dir = None
    for r in possible_roots:
        if r.exists():
            matches = sorted(list(r.glob(f"{args.experiment_name}*")))
            if matches:
                exp_dir = matches[-1]
                break
            
    if not exp_dir:
        print(f"Error: Experiment '{args.experiment_name}' not found.")
        print(f"Searched in: {[str(r) for r in possible_roots]}")
        sys.exit(1)

    print(f"Found experiment: {exp_dir}")

    # Find model file
    model_path = exp_dir / "best_model.pt"
    if not model_path.exists():
        model_path = exp_dir / "final_model.pt"
    if not model_path.exists():
        model_path = exp_dir / "checkpoint.pt"
    if not model_path.exists():
        print(f"Error: No model found in {exp_dir}")
        print("Looked for: best_model.pt, final_model.pt, checkpoint.pt")
        sys.exit(1)

    vis_output = exp_dir / "visualizations"
    
    # Create visualizer
    try:
        viz = DeepVisualizer(str(model_path), args.data, str(vis_output), device=args.device)
    except Exception as e:
        print(f"Error initializing visualizer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Attach hooks
    viz.attach_hooks()
    
    # Pick diverse examples
    print(f"\nSelecting {args.n_examples} diverse examples...")
    candidates = np.random.permutation(len(viz.y))
    indices = []
    seen_classes = set()
    
    # First pass: get one from each class with reasonable length
    for i in candidates:
        if viz.y[i] not in seen_classes and viz.lengths[i] > 20:
            seen_classes.add(viz.y[i])
            indices.append(i)
        if len(indices) >= args.n_examples:
            break
    
    # Second pass: fill remaining slots
    if len(indices) < args.n_examples:
        for i in candidates:
            if i not in indices and viz.lengths[i] > 10:
                indices.append(i)
            if len(indices) >= args.n_examples:
                break
    
    print(f"Selected indices: {indices}")
    
    # Visualize individual events
    print("\n=== Visualizing Individual Events ===")
    for idx in indices:
        viz.visualize_event_internals(idx)
    
    # Global analyses
    print("\n=== Analyzing Embedding Space ===")
    viz.analyze_embedding_space()
    
    print("\n=== Analyzing Confidence Evolution ===")
    viz.plot_confidence_evolution()
    
    print(f"\n✓ Complete! Visualizations saved to: {vis_output}")

if __name__ == '__main__':
    main()
