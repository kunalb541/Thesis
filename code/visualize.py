#!/usr/bin/env python3
"""
Causal Hybrid Model Internal Visualization Suite

Performs deep introspection of the Causal Hybrid Architecture using PyTorch forward hooks.
Generates diagnostic plots regarding attention mechanisms, embedding spaces, and temporal encoding.

Functional Scope:
- Internal State Extraction: Captures intermediate tensors (Attention maps, Embeddings).
- Dimensionality Reduction: PCA analysis of the learned manifold.
- Uncertainty Quantification: Aggregated confidence trajectories.
- Feature Analysis: Visualizes temporal encoding behavior.

Usage:
    python visualize.py --experiment_name "exp_01" --data "../data/test.npz" --n_examples 3

Author: Kunal Bhatia
Version: 1.0
Date: December 2025
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
# Force headless backend for HPC/Server compatibility
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

# --- Dynamic Import ---
try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from transformer import CausalHybridModel, CausalConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import 'transformer.py'.")
    print(f"Ensure the model file is in: {current_dir}")
    print(f"Python Error: {e}\n")
    sys.exit(1)

# Style Settings
try:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
except:
    pass
plt.rcParams['figure.dpi'] = 300

# =============================================================================
# UTILITIES
# =============================================================================

class HookManager:
    """
    Manages PyTorch forward hooks to capture intermediate layer outputs
    without modifying the model source code.
    """
    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def register_hook(self, module, name):
        def hook(model, input, output):
            # Capture output. 
            # If tuple (common in Transformers), take index 0.
            # Detach and move to CPU immediately to save GPU memory.
            tensor = output[0] if isinstance(output, tuple) else output
            self.outputs[name] = tensor.detach().cpu()
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)

    def clear_data(self):
        """Clears captured data but keeps hooks attached."""
        self.outputs = {}

    def remove_hooks(self):
        """Removes all hooks from the model."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

def interpolate_trajectory(times, values, n_points=100):
    """
    Normalizes time-series data to a fixed-length vector (0-100% progress)
    allowing aggregation of events with different durations.
    """
    if len(times) < 2: return np.zeros(n_points)
    
    # Normalize time 0 to 1
    t_norm = (times - times.min()) / (times.max() - times.min() + 1e-9)
    
    # Linear interpolation
    f = interp1d(t_norm, values, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    x_new = np.linspace(0, 1, n_points)
    return f(x_new)

def create_delta_t_from_timestamps(timestamps):
    """Replicates the delta_t logic from training/eval scripts."""
    if timestamps.ndim == 1:
        timestamps = timestamps[np.newaxis, :]
    delta_t = np.zeros_like(timestamps)
    if timestamps.shape[1] > 1:
        delta_t[:, 1:] = np.diff(timestamps, axis=1)
    return np.maximum(delta_t, 0.0)

# =============================================================================
# DEEP VISUALIZER CLASS
# =============================================================================

class DeepVisualizer:
    def __init__(self, model_path, data_path, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Model
        print(f"Loading checkpoint: {Path(model_path).name}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Robust Config Reconstruction
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            valid_keys = CausalConfig().__dict__.keys()
            filtered_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
            self.config = CausalConfig(**filtered_conf)
        else:
            print("Warning: Config not found in checkpoint. Initializing default 128/8/2.")
            self.config = CausalConfig(d_model=128, n_heads=8, n_transformer_layers=2)
        
        self.model = CausalHybridModel(self.config).to(self.device)
        
        # Load Weights (handling DDP prefixes)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        # 2. Load Data (Robust JSON/NPZ handling)
        print(f"Loading data: {Path(data_path).name}...")
        try:
            data = np.load(data_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

        self.flux = data.get('flux', data.get('X'))
        self.y = data.get('labels', data.get('y'))
        
        if self.flux is None or self.y is None:
            print("Invalid data: missing 'flux' or 'labels'")
            sys.exit(1)
            
        if self.flux.ndim == 3: self.flux = self.flux.squeeze(1)

        # Time handling
        if 'delta_t' in data:
            self.delta_t = data['delta_t']
        elif 'timestamps' in data:
            ts = data['timestamps']
            if ts.ndim == 1: ts = np.tile(ts, (len(self.flux), 1))
            self.delta_t = create_delta_t_from_timestamps(ts)
        else:
            self.delta_t = np.zeros_like(self.flux)
            
        if self.delta_t.ndim == 3: self.delta_t = self.delta_t.squeeze(1)
        
        # Lengths
        self.lengths = np.sum(self.flux != -1.0, axis=1).astype(np.int64)
        self.lengths = np.maximum(self.lengths, 1)
        
        self.hooks = HookManager()

    def attach_hooks(self):
        """
        Introspects the model structure and attaches hooks to:
        1. Attention Dropout (to capture Attention Weights)
        2. Final Normalization (to capture Embeddings)
        """
        self.hooks.remove_hooks()
        print("Attaching internal hooks...")
        
        # Hook 1: Attention Maps
        # Target: model.layers[i].attention.dropout
        # Rationale: The input to the dropout layer in 'StrictCausalAttention' is the softmaxed attention matrix.
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'dropout'):
                # Name it 'attn_layer_N'
                self.hooks.register_hook(layer.attention.dropout, f'attn_layer_{i}')
        
        # Hook 2: Final Embeddings
        # Target: model.final_norm
        # Rationale: This is the latent representation before the classifier head.
        if hasattr(self.model, 'final_norm'):
            self.hooks.register_hook(self.model.final_norm, 'final_embedding')

    def visualize_event_internals(self, idx):
        """Generates diagnostic plots for a single event."""
        print(f"  Visualizing Event Index {idx}...")
        
        f_t = torch.tensor(self.flux[idx]).float().unsqueeze(0).to(self.device)
        d_t = torch.tensor(self.delta_t[idx]).float().unsqueeze(0).to(self.device)
        l_t = torch.tensor([self.lengths[idx]]).long().to(self.device)
        
        self.hooks.clear_data()
        
        with torch.no_grad():
            out = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
            if 'probs' in out:
                probs = out['probs'][0].cpu().numpy()
            else:
                probs = torch.softmax(out['logits'][0], dim=-1).cpu().numpy()
        
        length = self.lengths[idx]
        valid_probs = probs[:length]
        
        # --- Plot 1: Classification Evolution ---
                fig, ax = plt.subplots(figsize=(10, 6))
        n_classes = valid_probs.shape[1]
        classes = ['Flat', 'PSPL', 'Binary'] if n_classes == 3 else ['PSPL', 'Binary']
        colors = ['gray', 'red', 'blue'] if n_classes == 3 else ['red', 'blue']
        
        for i, (cls, col) in enumerate(zip(classes, colors)):
            ax.plot(valid_probs[:, i], label=cls, color=col, linewidth=2, alpha=0.8)
        
        ax.set_title(f"Classification Evolution: Event {idx} (True: {classes[self.y[idx]]})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Probability")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"classification_evolution_event_{idx}.png")
        plt.close()

        # --- Plot 2: Attention Patterns (Last Layer) ---
                # Get the last layer's attention map
        last_layer_idx = len(self.model.layers) - 1
        attn_key = f'attn_layer_{last_layer_idx}'
        
        if attn_key in self.hooks.outputs:
            # Shape: [Batch, Heads, Seq, Seq] -> [1, H, N, N]
            attn_matrix = self.hooks.outputs[attn_key][0] 
            
            # Average over heads for cleaner visualization [N, N]
            attn_avg = attn_matrix.mean(dim=0).numpy()
            
            # Slice to valid length
            attn_avg = attn_avg[:length, :length]
            
            plt.figure(figsize=(8, 7))
            sns.heatmap(attn_avg, cmap='viridis', square=True, cbar_kws={'label': 'Attention Weight'})
            plt.title(f"Mean Attention Map (Layer {last_layer_idx}): Event {idx}")
            plt.xlabel("Key Position (Time)")
            plt.ylabel("Query Position (Time)")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"attention_patterns_event_{idx}.png")
            plt.close()

        # --- Plot 3: Temporal Encoding Analysis ---
                # We manually compute encoding to visualize it
        if hasattr(self.model, 'temporal_encoding'):
            # Encoding is stateless, safe to call
            pe = self.model.temporal_encoding(d_t).cpu().numpy()[0, :length] # [Seq, D]
            dt_vals = self.delta_t[idx, :length]
            
            # Calculate L2 norm of encoding vector per timestep
            pe_mag = np.linalg.norm(pe, axis=1)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(dt_vals, pe_mag, c=np.arange(length), cmap='plasma', alpha=0.7, s=20)
            plt.colorbar(label='Time Step Index')
            plt.xlabel("Delta T (Input)")
            plt.ylabel("Encoding Magnitude (L2)")
            plt.title(f"Temporal Encoding Analysis: Event {idx}")
            plt.xscale('log')
            plt.grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"temporal_encoding_event_{idx}.png")
            plt.close()

    def analyze_embedding_space(self, n_samples=1000):
        """PCA of the learned embeddings at the point of classification."""
                print("Generating Embedding Space PCA...")
        indices = np.random.choice(len(self.flux), min(len(self.flux), n_samples), replace=False)
        
        embeddings = []
        labels = []
        
        # Ensure hooks are ready
        self.attach_hooks()
        
        with torch.no_grad():
            for i in tqdm(indices, desc="Extracting Embeddings"):
                f_t = torch.tensor(self.flux[i]).float().unsqueeze(0).to(self.device)
                d_t = torch.tensor(self.delta_t[i]).float().unsqueeze(0).to(self.device)
                l_t = torch.tensor([self.lengths[i]]).long().to(self.device)
                
                self.hooks.clear_data()
                _ = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
                
                if 'final_embedding' in self.hooks.outputs:
                    # Get vector at last valid timestep
                    last_step = self.lengths[i] - 1
                    vec = self.hooks.outputs['final_embedding'][0, last_step].numpy()
                    embeddings.append(vec)
                    labels.append(self.y[i])
        
        if len(embeddings) == 0:
            print("Warning: No embeddings captured (Hook failed?). Skipping PCA.")
            return

        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        # PCA
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        n_classes = len(np.unique(self.y))
        classes = ['Flat', 'PSPL', 'Binary'] if n_classes == 3 else ['PSPL', 'Binary']
        
        sns.scatterplot(
            x=emb_pca[:, 0], y=emb_pca[:, 1], 
            hue=[classes[l] for l in labels], 
            style=[classes[l] for l in labels],
            palette='deep', s=60, alpha=0.8
        )
        plt.title("Latent Space PCA (Final Layer)")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.tight_layout()
        plt.savefig(self.output_dir / "embedding_space_pca.png")
        plt.close()

    def plot_confidence_evolution(self, n_samples=500):
        """Aggregated confidence trajectories by class."""
                print("Analyzing Confidence Evolution...")
        indices = np.random.choice(len(self.flux), min(len(self.flux), n_samples), replace=False)
        
        trajectories = {0: [], 1: [], 2: []}
        
        # Hooks not needed for this
        self.hooks.remove_hooks()

        with torch.no_grad():
            for i in tqdm(indices, desc="Processing Trajectories"):
                f_t = torch.tensor(self.flux[i]).float().unsqueeze(0).to(self.device)
                d_t = torch.tensor(self.delta_t[i]).float().unsqueeze(0).to(self.device)
                l_t = torch.tensor([self.lengths[i]]).long().to(self.device)
                
                out = self.model(f_t, d_t, lengths=l_t, return_all_timesteps=True)
                
                if 'probs' in out:
                    probs = out['probs'][0].cpu().numpy()
                else:
                    probs = torch.softmax(out['logits'][0], dim=-1).cpu().numpy()
                
                # Confidence of the TRUE class
                true_class = self.y[i]
                conf = probs[:self.lengths[i], true_class]
                
                # Interpolate
                traj = interpolate_trajectory(np.arange(len(conf)), conf)
                trajectories[true_class].append(traj)

        plt.figure(figsize=(10, 6))
        n_classes = len(np.unique(self.y))
        classes = ['Flat', 'PSPL', 'Binary'] if n_classes == 3 else ['PSPL', 'Binary']
        colors = ['gray', 'red', 'blue'] if n_classes == 3 else ['red', 'blue']
        
        for cls_idx, cls_name in enumerate(classes):
            data = trajectories.get(cls_idx)
            if not data: continue
            
            arr = np.array(data)
            mean_conf = np.mean(arr, axis=0)
            std_conf = np.std(arr, axis=0)
            x_axis = np.linspace(0, 100, 100)
            
            plt.plot(x_axis, mean_conf, label=f'{cls_name}', color=colors[cls_idx], linewidth=2)
            plt.fill_between(x_axis, mean_conf - 0.5*std_conf, mean_conf + 0.5*std_conf, color=colors[cls_idx], alpha=0.1)

        plt.xlabel("% of Light Curve Observed")
        plt.ylabel("Confidence in True Class")
        plt.title("Confidence Evolution by Class (Mean ± 0.5 SD)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confidence_evolution_by_class.png")
        plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deep Model Visualization")
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment folder')
    parser.add_argument('--data', type=str, required=True, help='Path to .npz data')
    parser.add_argument('--n_examples', type=int, default=3, help='Number of events to inspect')
    args = parser.parse_args()

    # 1. Locate Experiment
    possible_roots = [Path('../results'), Path('results'), Path('.')]
    exp_dir = None
    for r in possible_roots:
        if not r.exists(): continue
        # Find directory starting with experiment_name
        matches = sorted(list(r.glob(f"{args.experiment_name}*")))
        if matches:
            exp_dir = matches[-1] # Take latest
            break
    
    if not exp_dir:
        print(f"Error: Experiment '{args.experiment_name}' not found.")
        sys.exit(1)

    # 2. Locate Model
    model_path = exp_dir / "best_model.pt"
    if not model_path.exists():
        model_path = exp_dir / "final_model.pt"
    
    if not model_path.exists():
        print(f"Error: No .pt model file found in {exp_dir}")
        sys.exit(1)

    vis_output = exp_dir / "visualizations"
    
    print("="*60)
    print(f"Experiment: {exp_dir.name}")
    print(f"Model:      {model_path.name}")
    print(f"Output:     {vis_output}")
    print("="*60)

    # 3. Initialize Visualizer
    viz = DeepVisualizer(
        model_path=str(model_path),
        data_path=args.data,
        output_dir=str(vis_output)
    )

    # 4. Individual Event Analysis
    viz.attach_hooks()
    
    # Selection Heuristic: Try to get one of each class
    indices_to_plot = []
    found_classes = set()
    
    # Shuffle indices to get random examples each run
    candidates = np.random.permutation(len(viz.y))
    
    for i in candidates:
        lbl = viz.y[i]
        # Only take decent length events for plotting
        if lbl not in found_classes and viz.lengths[i] > 30:
            found_classes.add(lbl)
            indices_to_plot.append(i)
        if len(found_classes) >= 3: break
    
    # Fill remaining slots if n_examples > 3
    while len(indices_to_plot) < args.n_examples:
        indices_to_plot.append(np.random.randint(len(viz.y)))

    for idx in indices_to_plot:
        viz.visualize_event_internals(idx)

    # 5. Global Analysis
    viz.analyze_embedding_space()
    viz.plot_confidence_evolution()

    print(f"\nDone. Visualizations saved to: {vis_output}")

if __name__ == '__main__':
    main()
