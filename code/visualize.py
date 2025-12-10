import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.interpolate import interp1d
from collections import defaultdict
import gc

# --- Backend Configuration ---
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt

# --- Dynamic Import with Error Handling ---
# We append the parent directory to path to ensure model.py is findable
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# --- Global Config ---
np.random.seed(42)
torch.manual_seed(42)
sns.set_style("whitegrid")
COLORS = ['#95a5a6', '#3498db', '#e74c3c'] 
CLASS_NAMES = ['Flat', 'PSPL', 'Binary']

# =============================================================================
# UTILS & HOOKS
# =============================================================================
class HookManager:
    """Safe context manager for extracting intermediate layer activations."""
    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def register_hook(self, module, name):
        if module is None:
            # Fallback: Try to find a submodule if direct access fails, or warn
            print(f"(!) Warning: Module for hook '{name}' is None. Visualization will fail.")
            return None
        
        def hook(model, input, output):
            # Handle tuple outputs (common in RNNs: output, hidden)
            data = output[0] if isinstance(output, tuple) else output
            self.outputs[name] = data.detach() 
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)
        return handle

    def clear(self):
        self.outputs = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.outputs = {}

def interpolate_traj(y, n=100):
    """Interpolate trajectory to fixed length using Numpy (faster/safer)."""
    if len(y) < 2:
        return np.repeat(y[0] if len(y) > 0 else 0.5, n)
    
    # Handle NaNs
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=0.5) 
        
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, n)
    
    # Use numpy interp (more robust for flat lines than scipy interp1d)
    res = np.interp(x_new, x_old, y)
    return np.clip(res, 0.0, 1.0)

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
        
        # 2. Reconstruct Config & Import Model
        # We perform the import HERE to ensure sys.path is set correctly by main
        try:
            from model import GodModeCausalGRU, GRUConfig
        except ImportError:
            print("CRITICAL: 'model.py' not found. Ensure it is in the same folder as this script.")
            sys.exit(1)

        config_dict = ckpt.get('config', {})
        # Filter config keys
        valid_keys = GRUConfig().__init__.__code__.co_varnames
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys and k != 'self'}
        config = GRUConfig(**clean_conf)
        
        # 3. Initialize Model
        self.model = GodModeCausalGRU(config, dtype=torch.float32).to(self.device)
        
        # 4. Load State Dict
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        # Handle DataParallel 'module.' prefix
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(clean_state, strict=False) # strict=False to be robust against minor version changes
        self.model.eval()
        
        print(f"Model Active: {config.d_model}d | Layers: {config.n_layers} | Hierarchical: {config.hierarchical}")
        self.config = config
        
        # 5. Load Data
        print("Loading Data...")
        data = np.load(data_path, allow_pickle=True)
        
        self.flux = data['flux'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        
        if 'delta_t' in data:
            self.dt = data['delta_t'].astype(np.float32)
        else:
            print("Warning: 'delta_t' not found. Assuming constant cadence.")
            self.dt = np.zeros_like(self.flux) # Standard GRU assumption if no time provided

        # --- SHAPE SANITIZATION ---
        # Ensure (N, T) for storage
        if self.flux.ndim == 3 and self.flux.shape[2] == 1:
            self.flux = self.flux.squeeze(2)
        if self.dt.ndim == 3: 
            self.dt = self.dt.squeeze(-1)
            
        # Length Calculation
        if 'lengths' in data:
            self.lengths = data['lengths'].astype(np.int64)
        else:
            print("(!) Warning: Inferring lengths from padding (0.0).")
            # Assuming 0.0 is padding. 
            self.lengths = np.sum(np.abs(self.flux) > 1e-8, axis=1).astype(np.int64)
            self.lengths = np.maximum(self.lengths, 1)
        
        self.hooks = HookManager()

    def _compute_hierarchical_probs(self, outputs, temperature):
        """Robust probability calculation."""
        # Use softplus for temperature to avoid negative/zero division
        if isinstance(temperature, torch.Tensor):
            temp = F.softplus(temperature).clamp(min=0.1, max=10.0)
        else:
            temp = 1.0

        if self.config.hierarchical:
            # Check keys exist
            if 'aux_dev' in outputs and 'aux_type' in outputs:
                 # Map 'aux' keys from model.py to visualizer logic if needed
                 # model.py outputs: 'logits', 'probs', 'aux_dev', 'aux_type'
                 dev_logits = outputs['aux_dev']
                 type_logits = outputs['aux_type']
            elif 'deviation_logits' in outputs:
                 # Compatibility with older checkpoints
                 dev_logits = outputs['deviation_logits']
                 type_logits = outputs['type_logits']
            else:
                # Fallback to standard if keys missing
                # print("(!) Warning: Hierarchical config true but keys missing in output.")
                return outputs['probs'], {}

            dev_logits = dev_logits / temp
            type_logits = type_logits / temp
            
            dev_log_probs = F.log_softmax(dev_logits, dim=-1) # [..., 2]
            type_log_probs = F.log_softmax(type_logits, dim=-1) # [..., 2]
            
            # Recompute probs for consistency with temp
            log_p_flat = dev_log_probs[..., 0:1]
            log_p_pspl = dev_log_probs[..., 1:2] + type_log_probs[..., 0:1]
            log_p_binary = dev_log_probs[..., 1:2] + type_log_probs[..., 1:2]
            
            final_log_probs = torch.cat([log_p_flat, log_p_pspl, log_p_binary], dim=-1)
            probs = torch.exp(final_log_probs)
            
            aux = {
                'dev_probs': torch.exp(dev_log_probs),
                'type_probs': torch.exp(type_log_probs)
            }
        else:
            logits = outputs['logits'] / temp
            probs = F.softmax(logits, dim=-1)
            aux = {}
            
        return probs, aux

    def reconstruct_trajectory(self, f, d, l):
        self.hooks.clear()
        
        # --- ROBUST INPUT SHAPING ---
        # FIX: The model expects (B, T) and handles unsqueezing internally.
        # We ensure inputs are 2D.
        if f.dim() == 3 and f.shape[-1] == 1:
            f_in = f.squeeze(-1)
        else:
            f_in = f

        # Attempt to find the GRU layer. 
        target_layer = getattr(self.model, 'gru', None)
        if target_layer is None:
            # Try finding it in submodules (common in complex models)
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.GRU, nn.LSTM)) or 'GRU' in str(type(module)):
                    target_layer = module
                    break
        
        if target_layer is None:
            # Fallback for FusedLayerNormGRU in model.py
            # If it's a ModuleList wrapper, hook the last cell
            if hasattr(self.model, 'gru') and hasattr(self.model.gru, 'cells'):
                 target_layer = self.model.gru.cells[-1]

        if target_layer is None:
             print("Warning: Could not find GRU layer for hooks. Visuals may be incomplete.")
             embeddings = torch.zeros(f_in.size(0), f_in.size(1), self.config.d_model).to(self.device)
             outputs = self.model(f_in, d, lengths=l)
             probs, aux = self._compute_hierarchical_probs(outputs, 1.0)
             return probs[0].cpu().numpy(), aux, embeddings[0].cpu().numpy()

        # Register Hook
        self.hooks.register_hook(target_layer, 'gru_seq')
        
        try:
            with torch.no_grad():
                outputs = self.model(f_in, d, lengths=l)
                
                gru_seq = self.hooks.outputs.get('gru_seq')
                if gru_seq is None:
                    # If hook failed (e.g. FusedLayerNormGRU internal return structure), fallback
                    gru_seq = torch.zeros(f_in.size(0), f_in.size(1), self.config.d_model).to(self.device)

                # Check if norm_final exists, else identity
                if hasattr(self.model, 'norm_final'):
                    normed_seq = self.model.norm_final(gru_seq)
                else:
                    normed_seq = gru_seq
                
                # Temperature handling
                temp = getattr(self.model, 'raw_temperature', torch.tensor(1.0).to(self.device))
                probs, aux = self._compute_hierarchical_probs(outputs, temp)
                
        finally:
            # FIX: Properly remove all hooks to prevent memory leak
            self.hooks.remove_hooks()
            
        return probs[0].cpu().numpy(), aux, normed_seq[0].cpu().numpy()

    def plot_deep_analysis(self, idx):
        print(f"Deep Analysis for Event {idx}...")
        
        # Prepare Inputs
        f = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d = torch.tensor(self.dt[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        
        true_len = self.lengths[idx]
        true_cls = self.labels[idx]
        
        probs, aux, embeddings = self.reconstruct_trajectory(f, d, l)
        
        # --- PLOTTING ---
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Flux
        ax1 = fig.add_subplot(gs[0, :])
        # Only plot up to true length to avoid padding confusion
        plot_len = true_len 
        ax1.scatter(np.arange(plot_len), self.flux[idx][:plot_len], 
                   c='k', s=15, alpha=0.6, label='Flux')
        ax1.set_title(f"Event {idx} | Class: {CLASS_NAMES[true_cls]}", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Probability
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        for c in range(3):
            ax2.plot(probs[:plot_len, c], color=COLORS[c], linewidth=2.5, 
                    label=f'P({CLASS_NAMES[c]})', alpha=0.9)
        ax2.set_ylabel("Probability")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Heads
        if self.config.hierarchical and aux:
            ax3 = fig.add_subplot(gs[2, 0])
            dev_probs = aux['dev_probs'][0].cpu().numpy()
            ax3.plot(dev_probs[:plot_len, 1], color='purple', linewidth=2)
            ax3.set_title("Detection Head P(Event)", fontweight='bold')
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, alpha=0.3)
            
            ax4 = fig.add_subplot(gs[2, 1])
            type_probs = aux['type_probs'][0].cpu().numpy()
            ax4.plot(type_probs[:plot_len, 0], color=COLORS[1], label='PSPL')
            ax4.plot(type_probs[:plot_len, 1], color=COLORS[2], label='Binary')
            ax4.set_title("Classification Head", fontweight='bold')
            ax4.set_ylim(0, 1.1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Placeholder if not hierarchical
            ax3 = fig.add_subplot(gs[2, 0])
            ax3.axis('off')
            ax4 = fig.add_subplot(gs[2, 1])
            ax4.axis('off')

        # 4. Latent Norm (Padding Check)
        ax5 = fig.add_subplot(gs[2, 2])
        emb_norms = np.linalg.norm(embeddings, axis=1)
        ax5.plot(np.arange(true_len), emb_norms[:true_len], color='green')
        
        # Visualizing padding area if it exists
        if true_len < len(emb_norms):
            ax5.plot(np.arange(true_len, len(emb_norms)), emb_norms[true_len:], color='red', alpha=0.5)
            
        ax5.set_title("Latent Norms", fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.out_dir / f"event_{idx}_deep_dive.png", dpi=100)
        plt.close(fig) # Crucial for memory
        gc.collect()

    def analyze_early_detection(self, n=500):
        print("Analyzing Early Detection...")
        n = min(len(self.flux), n)
        idxs = np.random.choice(len(self.flux), n, replace=False)
        
        trajs = defaultdict(list)
        
        for i in tqdm(idxs, desc="Tracing"):
            f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
            
            probs, _, _ = self.reconstruct_trajectory(f, d, l)
            
            true_cls = self.labels[i]
            # Valid trajectory only
            conf_curve = probs[:self.lengths[i], true_cls]
            trajs[true_cls].append(interpolate_traj(conf_curve))
            
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in range(3):
            if not trajs[c]: continue
            arr = np.array(trajs[c])
            mu = np.mean(arr, axis=0)
            sigma = np.std(arr, axis=0)
            x = np.linspace(0, 100, 100)
            
            ax.plot(x, mu, color=COLORS[c], linewidth=3, label=CLASS_NAMES[c])
            ax.fill_between(x, mu-sigma, mu+sigma, color=COLORS[c], alpha=0.15)
            
        ax.set_xlabel("% of Lightcurve")
        ax.set_ylabel("True Class Confidence")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.savefig(self.out_dir / "early_detection.png", dpi=200)
        plt.close(fig)

    def plot_latent_space(self, n=1000):
        print("Mapping Latent Space...")
        n = min(len(self.flux), n)
        idxs = np.random.choice(len(self.flux), n, replace=False)
        
        embeddings = []
        labels = []
        
        for i in tqdm(idxs, desc="Extracting"):
            f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
            
            _, _, seq_emb = self.reconstruct_trajectory(f, d, l)
            
            # Safe indexing
            last_idx = max(0, self.lengths[i]-1)
            embeddings.append(seq_emb[last_idx])
            labels.append(self.labels[i])
            
        if len(embeddings) < 5: return

        pca = PCA(n_components=2)
        proj = pca.fit_transform(np.array(embeddings))
        labels = np.array(labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for c in range(3):
            mask = labels == c
            if not mask.any(): continue
            ax.scatter(proj[mask, 0], proj[mask, 1], c=COLORS[c], label=CLASS_NAMES[c], alpha=0.6)
            
        ax.set_title("Latent Space PCA")
        ax.legend()
        plt.savefig(self.out_dir / "latent_space.png", dpi=200)
        plt.close(fig)

    def generate_confusion_matrix(self, n=2000):
        print("Generating Confusion Matrix...")
        n = min(len(self.flux), n)
        idxs = np.random.choice(len(self.flux), n, replace=False)
        
        y_true = []
        y_pred = []
        batch_size = 128
        
        for i in tqdm(range(0, len(idxs), batch_size), desc="Batch Inference"):
            batch_idxs = idxs[i:i+batch_size]
            
            # Batch Construction
            f_np = self.flux[batch_idxs]
            
            # FIX: Ensure (B, T) - do NOT expand to (B, T, 1)
            if f_np.ndim == 3 and f_np.shape[-1] == 1:
                f_np = f_np.squeeze(-1)
                
            f = torch.tensor(f_np, dtype=torch.float32).to(self.device)
            d = torch.tensor(self.dt[batch_idxs], dtype=torch.float32).to(self.device)
            l = torch.tensor(self.lengths[batch_idxs], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(f, d, lengths=l)
                temp = getattr(self.model, 'raw_temperature', torch.tensor(1.0).to(self.device))
                probs, _ = self._compute_hierarchical_probs(outputs, temp)
                
                # --- GATHER LOGIC ---
                # Ensure l is on same device and shaped correctly for gather
                # Gather indices: (B, 1, C)
                # We want the probability at the LAST valid step: l-1
                last_steps = (l - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, probs.size(-1))
                
                final_probs = probs.gather(1, last_steps).squeeze(1)
                preds = final_probs.argmax(dim=1).cpu().numpy()
                
            y_true.extend(self.labels[batch_idxs])
            y_pred.extend(preds)
            
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        # Normalize safely
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        plt.savefig(self.out_dir / "confusion_matrix.png", dpi=200)
        plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--n_events', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Path Resolution
    res_roots = [Path('../results'), Path('results'), Path('.')]
    exp_path = None
    
    # Try finding folder
    for r in res_roots:
        matches = list(r.glob(f"*{args.experiment_name}*"))
        if matches:
            exp_path = sorted(matches)[-1]
            break
            
    # Fallback: Maybe user passed full path in experiment_name
    if not exp_path and Path(args.experiment_name).exists():
        exp_path = Path(args.experiment_name)
        
    if not exp_path:
        print(f"Error: Experiment {args.experiment_name} not found.")
        sys.exit(1)
        
    model_file = exp_path / "best_model.pt"
    if not model_file.exists():
        model_file = exp_path / "model.pt"
        
    viz = GodModeVisualizer(model_file, args.data, exp_path / "vis", device=args.device)
    
    viz.plot_latent_space()
    viz.generate_confusion_matrix()
    viz.analyze_early_detection()
    
    # Do deep dives last
    idxs = np.random.choice(len(viz.flux), min(len(viz.flux), args.n_events), replace=False)
    for i in idxs:
        viz.plot_deep_analysis(i)
        
    print(f"\nDone. Output: {exp_path / 'vis'}")

if __name__ == "__main__":
    main()
