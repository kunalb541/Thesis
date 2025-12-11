import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.interpolate import interp1d
from collections import defaultdict
import gc

# --- Dynamic Import ---
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# --- Global Config ---
np.random.seed(42)
torch.manual_seed(42)
sns.set_style("whitegrid")
COLORS = ['#95a5a6', '#e74c3c', '#3498db'] 
CLASS_NAMES = ['Flat', 'PSPL', 'Binary']

# =============================================================================
# VISUALIZER ENGINE
# =============================================================================
class GodModeVisualizer:
    def __init__(self, model_path, data_path, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Checkpoint
        print(f"Loading Checkpoint: {Path(model_path).name}...")
        try:
            ckpt = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"CRITICAL: Failed to load checkpoint: {e}")
            sys.exit(1)
        
        # 2. Config & Model
        try:
            from model import GodModeCausalGRU, GRUConfig
        except ImportError:
            print("CRITICAL: 'model.py' not found.")
            sys.exit(1)

        config_dict = ckpt.get('config', {})
        valid_keys = GRUConfig().__init__.__code__.co_varnames
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys and k != 'self'}
        config = GRUConfig(**clean_conf)
        
        self.model = GodModeCausalGRU(config, dtype=torch.float32).to(self.device)
        
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(clean_state, strict=False)
        self.model.eval()
        self.config = config
        
        # 3. Load & Normalize Data
        print(f"Loading Data from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        self.flux = data['flux'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        
        if 'delta_t' in data:
            self.dt = data['delta_t'].astype(np.float32)
        else:
            self.dt = np.zeros_like(self.flux)

        if 'lengths' in data:
            self.lengths = data['lengths'].astype(np.int64)
        else:
            self.lengths = np.maximum((self.flux != 0).sum(axis=1), 1)
            
        # --- CRITICAL NORMALIZATION FIX ---
        print("Normalizing flux (Median/IQR matching training)...")
        self.flux = self._normalize_flux(self.flux)
        
        # Hooks for feature extraction
        self.hook_handle = None
        self.hook_output = {}

    def _normalize_flux(self, flux):
        """Matches train.py normalization logic exactly."""
        subset = flux[:10000].flatten()
        subset = subset[subset != 0]
        if len(subset) == 0: return flux
        
        median = np.median(subset)
        q75, q25 = np.percentile(subset, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-6: iqr = 1.0
        
        print(f"  > Norm Stats: Median={median:.4f}, IQR={iqr:.4f}")
        
        mask = (flux != 0)
        # Apply normalization only to valid parts
        return np.where(mask, (flux - median) / iqr, 0.0)

    def _register_hook(self):
        def hook(model, input, output):
            # Capture the output of the GRU layer (before final norm/pool)
            # FusedLayerNormGRU returns (seq, hidden)
            self.hook_output['gru_seq'] = output[0].detach()
            
        # Target the GRU layer
        if hasattr(self.model, 'gru'):
             self.hook_handle = self.model.gru.register_forward_hook(hook)

    def _remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

    def reconstruct_trajectory(self, f, d, l):
        self._register_hook()
        try:
            with torch.no_grad():
                outputs = self.model(f, d, lengths=l, return_all_timesteps=True)
                
                # Get embeddings from hook
                embeddings = self.hook_output.get('gru_seq', None)
                if embeddings is None:
                    embeddings = torch.zeros(f.size(0), f.size(1), self.config.d_model).to(self.device)
                
                # Get probabilities
                temp = getattr(self.model, 'raw_temperature', torch.tensor(1.0).to(self.device))
                temp = F.softplus(temp).clamp(min=0.1, max=10.0)
                
                if self.config.hierarchical and 'logits_seq' in outputs:
                    # Hierarchical Logic
                    logits = outputs['logits_seq'] / temp
                    probs = torch.exp(logits) # Logits are actually log_probs in hierarchical output
                else:
                    logits = outputs['logits_seq'] / temp
                    probs = F.softmax(logits, dim=-1)
                    
        finally:
            self._remove_hook()
            
        return probs[0].cpu().numpy(), embeddings[0].cpu().numpy()

    def plot_deep_analysis(self, idx):
        print(f"  Plotting Event {idx}...")
        f = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d = torch.tensor(self.dt[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        
        probs, embeddings = self.reconstruct_trajectory(f, d, l)
        T = self.lengths[idx]
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Flux
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.flux[idx][:T], c='k', alpha=0.6, label='Normalized Flux')
        ax1.set_title(f"Event {idx} | Class: {CLASS_NAMES[self.labels[idx]]}")
        ax1.legend()
        
        # 2. Probability Evolution
        ax2 = fig.add_subplot(gs[1, 0])
        for c in range(3):
            ax2.plot(probs[:T, c], color=COLORS[c], label=CLASS_NAMES[c], lw=2)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title("Probability Evolution")
        ax2.legend()
        
        # 3. Latent Norm (Activity)
        ax3 = fig.add_subplot(gs[1, 1])
        norms = np.linalg.norm(embeddings[:T], axis=1)
        ax3.plot(norms, color='purple', lw=2)
        ax3.set_title("Latent State Activity (Norm)")
        
        plt.tight_layout()
        plt.savefig(self.out_dir / f"event_{idx}_deep_dive.png")
        plt.close()
        gc.collect()

    def plot_latent_space(self, n=1000):
        print("Mapping Latent Space...")
        n = min(len(self.flux), n)
        idxs = np.random.choice(len(self.flux), n, replace=False)
        
        embeddings = []
        labels = []
        
        self._register_hook()
        with torch.no_grad():
            # Batch process for speed
            batch_size = 128
            for i in range(0, n, batch_size):
                batch_idx = idxs[i:i+batch_size]
                f = torch.tensor(self.flux[batch_idx], dtype=torch.float32).to(self.device)
                d = torch.tensor(self.dt[batch_idx], dtype=torch.float32).to(self.device)
                l = torch.tensor(self.lengths[batch_idx], dtype=torch.long).to(self.device)
                
                self.model(f, d, lengths=l)
                
                # Get last valid embedding
                seq_emb = self.hook_output['gru_seq'] # B, T, D
                
                # Gather last states
                last_idx = (l - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, seq_emb.size(-1))
                last_emb = seq_emb.gather(1, last_idx).squeeze(1)
                
                embeddings.append(last_emb.cpu().numpy())
                labels.extend(self.labels[batch_idx])
        self._remove_hook()
        
        embeddings = np.concatenate(embeddings)
        labels = np.array(labels)
        
        if len(embeddings) < 5: return

        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        for c in range(3):
            mask = labels == c
            plt.scatter(proj[mask, 0], proj[mask, 1], c=COLORS[c], label=CLASS_NAMES[c], alpha=0.6, s=15)
        plt.title("Latent Space (PCA)")
        plt.legend()
        plt.savefig(self.out_dir / "latent_space.png")
        plt.close()

# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True, help="Specific output folder for plots")
    parser.add_argument('--n_events', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Resolve Model Path
    exp_path = None
    res_roots = [Path('../results'), Path('results'), Path('.')]
    for r in res_roots:
        matches = list(r.glob(f"*{args.experiment_name}*"))
        if matches:
            exp_path = sorted(matches, key=lambda x: x.stat().st_mtime)[-1]
            break
            
    if not exp_path:
        print(f"Error: Experiment {args.experiment_name} not found.")
        sys.exit(1)
        
    model_file = exp_path / "best_model.pt"
    
    viz = GodModeVisualizer(model_file, args.data, args.output_dir, device=args.device)
    viz.plot_latent_space(n=2000)
    
    # Deep Dives
    idxs = np.random.choice(len(viz.flux), min(len(viz.flux), args.n_events), replace=False)
    for i in idxs:
        viz.plot_deep_analysis(i)

if __name__ == "__main__":
    main()
