#!/usr/bin/env python3
"""
Causal Hybrid Model Internal Visualization Suite (Fixed)

Debugged & Enhanced:
- Fixed Syntax/Indentation errors (removed non-breaking spaces).
- Fixed Matplotlib backend definition.
- Robust import handling for transformer.py.
- Verified compatibility with simulate.py padding (-1.0).
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

# --- Matplotlib Backend Setup (Must be before pyplot import) ---
import matplotlib
try:
    matplotlib.use('Agg') # Force headless backend for server environments
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dynamic Import Setup ---
# Robustly find transformer.py regardless of where script is run
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent)) 

try:
    from transformer import CausalHybridModel, CausalConfig
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import 'transformer.py'.")
    print(f"Ensure the model definition file is in {current_dir}")
    print(f"Python Error: {e}\n")
    sys.exit(1)

# Style Settings
try:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
except Exception:
    pass

# =============================================================================
# UTILITIES
# =============================================================================

class HookManager:
    """Manages PyTorch forward hooks to capture intermediate layer outputs."""
    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def register_hook(self, module, name, capture_input=False):
        """
        Registers a hook. 
        If capture_input=True, we grab the input tuple (useful for Dropout to get Attn Weights).
        If capture_input=False, we grab the output tensor.
        """
        def hook(model, input, output):
            if capture_input:
                # Input is usually a tuple, we want the first element (the tensor)
                data = input[0] if isinstance(input, tuple) else input
            else:
                data = output[0] if isinstance(output, tuple) else output
            
            # Detach and move to CPU immediately to save VRAM
            if isinstance(data, torch.Tensor):
                self.outputs[name] = data.detach().cpu()

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
    if np.isclose(t_max, t_min):
        return np.full(n_points, values[0] if len(values) > 0 else 0.0)
    
    t_norm = (times - t_min) / (t_max - t_min)
    
    try:
        f = interp1d(t_norm, values, kind='linear', bounds_error=False, fill_value="extrapolate")
        x_new = np.linspace(0, 1, n_points)
        return f(x_new)
    except Exception:
        return np.zeros(n_points)

def create_delta_t_from_timestamps(timestamps):
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
        
        # Safe Config Loading
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            # Filter valid keys only
            valid_keys = CausalConfig().__dict__.keys()
            filtered_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
            self.config = CausalConfig(**filtered_conf)
        else:
            print("Warning: Config not found. Using defaults.")
            self.config = CausalConfig(d_model=128, n_heads=8, n_transformer_layers=2)
        
        self.model = CausalHybridModel(self.config).to(self.device)
        
        # Handle state dict loading (support DDP and standard)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            self.model.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print(f"Warning on weight loading: {e}")
            
        self.model.eval()

        # 2. Load Data
        print(f"Loading data: {Path(data_path).name}...")
        try:
            data = np.load(data_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading data file: {e}")
            sys.exit(1)

        self.flux = data.get('flux', data.get('X', None))
        self.y = data.get('labels', data.get('y', None))
        
        if self.flux is None or self.y is None:
            print(f"Invalid data keys. Found: {list(data.keys())}")
            sys.exit(1)
            
        # Handle Ragged Arrays (Object Arrays)
        if self.flux.dtype == np.object_:
            print("Warning: Detected ragged array. Attempting to stack...")
            try:
                # Only works if lengths are actually same
                self.flux = np.stack(self.flux).astype(np.float32)
            except (ValueError, TypeError):
                print("CRITICAL: Data is ragged (different lengths per sample).")
                print("Visualization requires padded batch format (N, T).")
                sys.exit(1)
        
        if self.flux.ndim == 3: self.flux = self.flux.squeeze(1)
        self.flux = self.flux.astype(np.float32)

        # Time handling
        if 'delta_t' in data:
            self.delta_t = data['delta_t'].astype(np.float32)
            if self.delta_t.ndim == 3: self.delta_t = self.delta_t.squeeze(1)
        elif 'timestamps' in data:
            ts = data['timestamps']
            if ts.ndim == 1: ts = np.tile(ts, (len(self.flux), 1))
            self.delta_t = create_delta_t_from_timestamps(ts)
        else:
            self.delta_t = np.zeros_like(self.flux, dtype=np.float32)
            
        if self.delta_t.ndim == 3: self.delta_t = self.delta_t.squeeze(1)
        
        # Length Calculation compatible with simulate.py pad value (-1.0)
        # Also handle potential NaN padding
        is_padding = (self.flux == -1.0) | (np.isnan(self.flux))
        self.lengths = np.sum(~is_padding, axis=1).astype(np.int64)
        self.lengths = np.maximum(self.lengths, 1)
        self.lengths = np.minimum(self.lengths, self.flux.shape[1])
        
        self.hooks = HookManager()

    def attach_hooks(self):
        """
        Intelligently attaches hooks to:
        1. Attention Dropout (to get Attention Weights)
        2. Final Norm (to get Embeddings)
        """
        self.hooks.remove_hooks()
        hook_count = 0
        
        # Hook Transformer Layers
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                # We target the DROPOUT layer inside Attention.
                # In transformer.py: self.dropout(attn_weights)
                # Input to this layer is the Softmaxed Attention Matrix.
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'dropout'):
                    self.hooks.register_hook(layer.attention.dropout, f'layer_{i}_attn', capture_input=True)
                    hook_count += 1
                
        if hasattr(self.model, 'final_norm'):
            self.hooks.register_hook(self.model.final_norm, 'final_embedding', capture_input=False)
            
        print(f"Hooks registered: {hook_count} attention layers.")

    def visualize_event_internals(self, idx):
        print(f"   Visualizing Event Index {idx}...")
        
        # Prepare Batch of 1
        f_t = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d_t = torch.tensor(self.delta_t[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l_t = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        
        self.attach_hooks()
        self.hooks.clear_data()
        
        try:
            with torch.no_grad():
                out = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
                
                if isinstance(out, dict) and 'probs' in out:
                    probs = out['probs'][0].cpu().numpy()
                else:
                    probs = torch.softmax(out['logits'][0], dim=-1).cpu().numpy()
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return
        
        length = self.lengths[idx]
        valid_probs = probs[:length]
        
        # --- Plot 1: Classification Evolution ---
        fig, ax = plt.subplots(figsize=(10, 5))
        n_classes = valid_probs.shape[1]
        classes = ['Flat', 'PSPL', 'Binary'] if n_classes == 3 else [f'C{i}' for i in range(n_classes)]
        colors = ['gray', 'red', 'blue'] if n_classes == 3 else sns.color_palette("husl", n_classes)
        
        for i, (cls, col) in enumerate(zip(classes, colors)):
            ax.plot(valid_probs[:, i], label=cls, color=col, linewidth=2, alpha=0.8)
        
        ax.set_title(f"Prediction Evolution (Event {idx} - True: {classes[int(self.y[idx])]})")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Time Step")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"evolution_event_{idx}.png", dpi=150)
        plt.close(fig)

        # --- Plot 2: Attention Maps ---
        # We want the LAST layer usually
        last_layer_idx = len(self.model.layers) - 1
        attn_key = f'layer_{last_layer_idx}_attn'
        
        if attn_key in self.hooks.outputs:
            # Shape: [1, Heads, Seq, Seq]
            attn_matrix = self.hooks.outputs[attn_key][0] 
            
            # Average over heads for visualization: [Seq, Seq]
            avg_attn = attn_matrix.mean(dim=0).numpy()
            
            # Crop to valid length to avoid plotting padding
            # Limit to last 100 steps if sequence is huge
            view_len = min(length, 100)
            avg_attn = avg_attn[:view_len, :view_len]
            
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(avg_attn, cmap='viridis', ax=ax, vmin=0, vmax=np.percentile(avg_attn, 99))
            ax.set_title(f"Average Attention Weights (Layer {last_layer_idx})")
            ax.set_xlabel("Key (Source)")
            ax.set_ylabel("Query (Target)")
            
            # Since it's causal, upper triangle should be masked
            plt.savefig(self.output_dir / f"attention_event_{idx}.png", bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            print(f"Warning: No attention weights found for {attn_key}")
        
        self.hooks.remove_hooks()

    def analyze_embedding_space(self, n_samples=2000):
        print("Generating Embedding Space PCA...")
        n_samples = min(len(self.flux), n_samples)
        indices = np.random.choice(len(self.flux), n_samples, replace=False)
        
        embeddings, labels_list = [], []
        
        # Only hook final embedding this time
        self.hooks.remove_hooks()
        if hasattr(self.model, 'final_norm'):
            self.hooks.register_hook(self.model.final_norm, 'final_embedding', capture_input=False)
        
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
                        # Grab embedding at the last valid timestep
                        last_step = self.lengths[i] - 1
                        vec = tensor_out[0, last_step].numpy()
                        embeddings.append(vec)
                        labels_list.append(self.y[i])
                except Exception:
                    continue
        
        self.hooks.remove_hooks()
        
        if len(embeddings) < 5: 
            print("Not enough embeddings for PCA.")
            return

        try:
            pca = PCA(n_components=2)
            emb_pca = pca.fit_transform(np.array(embeddings))
            
            fig = plt.figure(figsize=(10, 8))
            classes = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'red', 'blue']
            
            # Convert labels to int for indexing
            labels_np = np.array(labels_list).astype(int)
            
            for i, cls in enumerate(classes):
                mask = labels_np == i
                if mask.sum() > 0:
                    plt.scatter(emb_pca[mask, 0], emb_pca[mask, 1], 
                                c=colors[i], label=cls, alpha=0.6, s=15)
            
            plt.title("Latent Space PCA (Last Timestep)")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "embedding_space_pca.png", bbox_inches='tight', dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"PCA failed: {e}")

    def plot_confidence_evolution(self, n_samples=500):
        print("Analyzing Confidence Evolution...")
        n_samples = min(len(self.flux), n_samples)
        indices = np.random.choice(len(self.flux), n_samples, replace=False)
        trajectories = {} 

        with torch.no_grad():
            for i in tqdm(indices, desc="Traj"):
                try:
                    f_t = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    d_t = torch.tensor(self.delta_t[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    l_t = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                    
                    out = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
                    if isinstance(out, dict) and 'probs' in out: probs = out['probs'][0].cpu().numpy()
                    else: probs = torch.softmax(out['logits'][0], dim=-1).cpu().numpy()

                    true_class = int(self.y[i])
                    # Get probability of the true class over time
                    conf = probs[:self.lengths[i], true_class]
                    
                    if len(conf) < 5: continue
                    
                    # Interpolate to 0-100% timescale
                    traj = interpolate_trajectory(np.arange(len(conf)), conf)
                    
                    if true_class not in trajectories: trajectories[true_class] = []
                    trajectories[true_class].append(traj)
                except Exception: continue

        if not trajectories: return

        fig = plt.figure(figsize=(10, 6))
        classes = ['Flat', 'PSPL', 'Binary']
        colors = ['gray', 'red', 'blue']

        for cls_idx in sorted(trajectories.keys()):
            data = trajectories[cls_idx]
            if not data: continue
            
            # Calculate Mean and Std
            arr = np.array(data)
            mean_conf = np.mean(arr, axis=0)
            std_conf = np.std(arr, axis=0)
            x = np.linspace(0, 100, 100)
            
            label_name = classes[cls_idx] if cls_idx < 3 else f"Class {cls_idx}"
            col = colors[cls_idx] if cls_idx < 3 else None
            
            plt.plot(x, mean_conf, label=label_name, linewidth=2, color=col)
            plt.fill_between(x, mean_conf - std_conf, mean_conf + std_conf, color=col, alpha=0.1)
            
        plt.xlabel("% of Light Curve Observed")
        plt.ylabel("Confidence in True Class")
        plt.legend(loc='lower right')
        plt.title("Model Confidence Trajectory (Mean ± Std)")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "confidence_evolution.png", bbox_inches='tight', dpi=150)
        plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Search for experiment folder
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
        sys.exit(1)

    # Search for model file
    model_path = exp_dir / "best_model.pt"
    if not model_path.exists(): model_path = exp_dir / "final_model.pt"
    if not model_path.exists():
        # Try any .pt file
        pts = list(exp_dir.glob("*.pt"))
        if pts: model_path = pts[0]
        else:
            print(f"Error: No model found in {exp_dir}")
            sys.exit(1)

    vis_output = exp_dir / "visualizations"
    print(f"Selected Model: {model_path}")
    print(f"Output Directory: {vis_output}")
    
    viz = DeepVisualizer(str(model_path), args.data, str(vis_output), device=args.device)
    
    # Run Visualizations
    indices = np.random.permutation(len(viz.y))
    selected_indices = indices[:args.n_examples]
        
    for idx in selected_indices:
        viz.visualize_event_internals(idx)
        
    viz.analyze_embedding_space()
    viz.plot_confidence_evolution()
    
    print(f"\nDone. Output: {vis_output}")

if __name__ == '__main__':
    main()
