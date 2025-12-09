import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib
try:
    matplotlib.use('Agg') # Force headless backend for clusters
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset

# --- Dynamic Import ---
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

try:
    from model import CausalHybridModel, CausalConfig
except ImportError:
    print("CRITICAL: 'transformer.py' not found.")
    sys.exit(1)

sns.set_style("whitegrid")
COLORS = ['#95a5a6', '#e74c3c', '#3498db'] # Grey, Red, Blue

# =============================================================================
# UTILS
# =============================================================================
class HookManager:
    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def register_hook(self, module, name):
        def hook(model, input, output):
            # Output of GRU is (out, h_n), Transformer is (x, incremental_state)
            if isinstance(output, tuple):
                data = output[0]
            else:
                data = output
            self.outputs[name] = data.detach().cpu()
        self.hooks.append(module.register_forward_hook(hook))

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []
        self.outputs = {}

def create_delta_t(ts):
    if ts.ndim == 1: ts = ts[np.newaxis, :]
    dt = np.zeros_like(ts, dtype=np.float32)
    if ts.shape[1] > 1: dt[:, 1:] = np.diff(ts, axis=1)
    return np.maximum(dt, 0.0)

def interpolate_traj(y, n=100):
    # Ensure input is numpy and float
    y = np.array(y, dtype=np.float64)
    if len(y) < 2: return np.zeros(n)
    x = np.linspace(0, 1, len(y))
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f(np.linspace(0, 1, n))

# =============================================================================
# VISUALIZER
# =============================================================================
class GodModeVisualizer:
    def __init__(self, model_path, data_path, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Model
        print(f"Loading Model: {Path(model_path).name}...")
        ckpt = torch.load(model_path, map_location=self.device)
        config_dict = ckpt.get('config', {})
        # Filter config
        valid_keys = CausalConfig().__dict__.keys()
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
        self.config = CausalConfig(**clean_conf)
        
        self.model = CausalHybridModel(self.config).to(self.device)
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(clean_state, strict=False)
        self.model.eval()
        
        # Load Data
        print("Loading Data...")
        data = np.load(data_path, allow_pickle=True)
        self.flux = data.get('flux', data.get('X')).astype(np.float32)
        self.y = data.get('labels', data.get('y'))
        
        if 'delta_t' in data: self.dt = data['delta_t'].astype(np.float32)
        else: self.dt = create_delta_t(data.get('timestamps'))
        
        if self.flux.ndim == 3: self.flux = self.flux.squeeze(1)
        if self.dt.ndim == 3: self.dt = self.dt.squeeze(1)
        
        # Lengths (Strict 0.0 is padding)
        self.lengths = (self.flux != 0.0).sum(axis=1)
        self.lengths = np.maximum(self.lengths, 1)
        
        self.hooks = HookManager()

    def plot_single_event_analysis(self, idx):
        """Visualizes: Flux, Probabilities, Attention, GRU Norm, and Feature Norm."""
        print(f"Visualizing Event {idx}...")
        
        f = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d = torch.tensor(self.dt[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        true_len = self.lengths[idx]
        
        # Register Hooks
        self.hooks.remove_hooks()
        
        # 1. Attention Dropout (Weights)
        attn_layer = self.model.layers[0].attention
        def attn_hook(module, input, output):
            self.hooks.outputs['attn'] = input[0].detach().cpu()
        self.hooks.hooks.append(attn_layer.dropout.register_forward_hook(attn_hook))
        
        # 2. GRU Output
        self.hooks.register_hook(self.model.gru_norm, 'gru_out')
        
        # 3. Final Norm (Before Masking)
        self.hooks.register_hook(self.model.final_norm, 'final_emb')
        
        with torch.no_grad():
            out = self.model(f, d, lengths=l, return_all_timesteps=True)
            probs = out['probs'][0].cpu().numpy()
            
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Plot 1: Flux & Probs
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(np.arange(true_len), self.flux[idx][:true_len], c='k', s=10, alpha=0.5, label='Flux')
        cls_names = ['Flat', 'PSPL', 'Binary']
        for c in range(3):
            ax1.plot(probs[:true_len, c], label=f'Prob({cls_names[c]})', color=COLORS[c], lw=2)
        ax1.axvline(x=true_len-1, color='k', ls='--')
        ax1.set_title(f"Event {idx} (True: {cls_names[self.y[idx]]})")
        ax1.legend()
        
        # Plot 2: Attention 
        ax2 = fig.add_subplot(gs[1, 0])
        if 'attn' in self.hooks.outputs:
            attn = self.hooks.outputs['attn'][0].mean(dim=0).numpy()
            view = min(true_len + 5, attn.shape[0])
            sns.heatmap(attn[:view, :view], cmap='viridis', ax=ax2, cbar=False)
            ax2.plot([0, view], [0, view], 'r--', lw=1)
            ax2.set_title("Causal Attention (Mask Verification)")
            
        # Plot 3: GRU Dynamics
        ax3 = fig.add_subplot(gs[1, 1])
        if 'gru_out' in self.hooks.outputs:
            gru_norm = self.hooks.outputs['gru_out'][0].norm(dim=-1).numpy()
            ax3.plot(gru_norm, color='purple', lw=2)
            ax3.set_title("GRU State Accumulation")
            
        # Plot 4: Padding Integrity
        ax4 = fig.add_subplot(gs[2, :])
        if 'final_emb' in self.hooks.outputs:
            emb_norm = self.hooks.outputs['final_emb'][0].norm(dim=-1).numpy()
            ax4.plot(emb_norm, color='green', lw=2, label='Pre-Mask Feature Norm')
            
            pad_start = true_len
            if pad_start < len(emb_norm):
                ax4.axvspan(pad_start, len(emb_norm), color='red', alpha=0.1, label='Padding')
                # Note: Model manually masks this after FinalNorm.
                ax4.text(pad_start, ax4.get_ylim()[1]*0.8, "Arch. Forces 0 here", color='red')
            
            ax4.set_title("Feature Integrity Check")
            ax4.legend()

        plt.tight_layout()
        plt.savefig(self.out_dir / f"deep_analysis_{idx}.png", dpi=150)
        plt.close(fig)
        self.hooks.remove_hooks()

    def analyze_confidence_trajectory(self, n=500):
        print("Analyzing Confidence Trajectories...")
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        trajs = {0: [], 1: [], 2: []}
        
        # Batching this is hard due to ragged lengths, loop is acceptable for N=500
        with torch.no_grad():
            for i in tqdm(idxs):
                f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                
                out = self.model(f, d, lengths=l, return_all_timesteps=True)
                probs = out['probs'][0].cpu().numpy()
                
                true_c = int(self.y[i])
                conf = probs[:self.lengths[i], true_c]
                trajs[true_c].append(interpolate_traj(conf))
                
        fig, ax = plt.subplots(figsize=(10, 6))
        cls_names = ['Flat', 'PSPL', 'Binary']
        for c in range(3):
            if not trajs[c]: continue
            arr = np.array(trajs[c])
            mu, sigma = arr.mean(axis=0), arr.std(axis=0)
            x = np.linspace(0, 100, 100)
            ax.plot(x, mu, color=COLORS[c], label=cls_names[c], lw=2)
            ax.fill_between(x, mu-sigma, mu+sigma, color=COLORS[c], alpha=0.1)
            
        ax.set_title("Early Detection Capability")
        ax.set_ylabel("True Class Confidence")
        ax.set_xlabel("% Observed")
        ax.legend()
        plt.savefig(self.out_dir / "confidence_evolution.png", dpi=300)
        plt.close(fig)

    def plot_latent_space(self, n=2000):
        print("Generating PCA (Batched)...")
        # Optimized: Batch Processing
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        
        # Create TensorDataset for fast loading
        ds = TensorDataset(
            torch.tensor(self.flux[idxs], dtype=torch.float32),
            torch.tensor(self.dt[idxs], dtype=torch.float32),
            torch.tensor(self.lengths[idxs], dtype=torch.long),
            torch.tensor(self.y[idxs], dtype=torch.long)
        )
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        
        vecs, labs = [], []
        self.hooks.register_hook(self.model.final_norm, 'emb')
        
        with torch.no_grad():
            for f, d, l, y in tqdm(dl):
                f, d, l = f.to(self.device), d.to(self.device), l.to(self.device)
                _ = self.model(f, d, lengths=l)
                
                # Extract last valid timestep
                # emb output: [B, T, D]
                emb = self.hooks.outputs['emb'].to(self.device) # Move back to GPU for gathering
                
                # Gather last timestep indices: [B, 1, D]
                idx_gather = (l - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, emb.size(2))
                last_emb = emb.gather(1, idx_gather).squeeze(1)
                
                vecs.append(last_emb.cpu().numpy())
                labs.append(y.numpy())
                
        self.hooks.remove_hooks()
        
        # 

[Image of Principal Component Analysis scatter plot]

        pca = PCA(n_components=2)
        proj = pca.fit_transform(np.concatenate(vecs))
        all_labs = np.concatenate(labs)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        for c in range(3):
            mask = all_labs == c
            ax.scatter(proj[mask, 0], proj[mask, 1], c=COLORS[c], label=['Flat', 'PSPL', 'Binary'][c], alpha=0.6, s=15)
            
        ax.set_title("Latent Space PCA")
        ax.legend()
        plt.savefig(self.out_dir / "latent_pca.png", dpi=300)
        plt.close(fig)

# =============================================================================
# RUN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    
    res_roots = [Path('../results'), Path('results'), Path('.')]
    exp_path = None
    for r in res_roots:
        matches = list(r.glob(f"{args.experiment_name}*"))
        if matches: 
            exp_path = sorted(matches)[-1]
            break
            
    if not exp_path: print("Exp not found."); sys.exit(1)
    model_file = list(exp_path.glob("*.pt"))[0]
    
    viz = GodModeVisualizer(str(model_file), args.data, str(exp_path / "vis"))
    
    idxs = np.random.choice(len(viz.flux), 3, replace=False)
    for i in idxs: viz.plot_single_event_analysis(i)
        
    viz.analyze_confidence_trajectory()
    viz.plot_latent_space()
    print("Done.")

if __name__ == '__main__':
    main()
