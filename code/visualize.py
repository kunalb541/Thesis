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
import warnings

# --- Dynamic Import ---
# Ensures we can find model.py in the same directory
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

try:
    from model import GodModeCausalGRU, GRUConfig
except ImportError:
    print("CRITICAL: 'model.py' not found. Ensure it is in the same directory.")
    sys.exit(1)

# Set visualization seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

sns.set_style("whitegrid")
# Standard Roman Telescope Palette: Flat (Grey), PSPL (Blue), Binary (Red)
COLORS = ['#95a5a6', '#3498db', '#e74c3c'] 
CLASS_NAMES = ['Flat', 'PSPL', 'Binary']

# =============================================================================
# UTILS
# =============================================================================
class HookManager:
    """
    Manages forward hooks to capture intermediate GRU states for trajectory reconstruction.
    """
    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def register_hook(self, module, name):
        if module is None:
            raise ValueError(f"Cannot register hook '{name}': module is None")
        
        def hook(model, input, output):
            # Capture tuple outputs (like GRU) or single tensors
            data = output[0] if isinstance(output, tuple) else output
            self.outputs[name] = data.detach() # Keep on device for faster processing
        
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
    """Interpolate trajectory to fixed length for aggregation."""
    if len(y) < 2:
        return np.zeros(n)
    x = np.linspace(0, 1, len(y))
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f(np.linspace(0, 1, n))

# =============================================================================
# VISUALIZER ENGINE
# =============================================================================
class GodModeVisualizer:
    """
    Comprehensive visualization suite for GodModeCausalGRU.
    
    Features:
    - Reconstructs temporal probability trajectories from GRU hidden states.
    - Analyzes Hierarchical Decision Making (Deviation vs Type).
    - Validates "God Mode" padding integrity.
    """
    def __init__(self, model_path, data_path, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Checkpoint
        print(f"Loading Checkpoint: {Path(model_path).name}...")
        ckpt = torch.load(model_path, map_location=self.device)
        
        # Reconstruct Config
        config_dict = ckpt.get('config', {})
        # Filter strictly for GRUConfig fields to avoid unexpected arg errors
        valid_keys = GRUConfig().__init__.__code__.co_varnames
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
        self.config = GRUConfig(**clean_conf)
        
        # Initialize Model
        # Use float32 for visualization stability (overriding amp_dtype)
        self.model = GodModeCausalGRU(self.config, dtype=torch.float32).to(self.device)
        
        # Load State Dict
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        # Handle DDP prefixes
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(clean_state, strict=True)
        self.model.eval()
        
        print(f"Model Loaded: {self.config.d_model}d, Layers={self.config.n_layers}")
        
        # Load Data
        print("Loading Data...")
        data = np.load(data_path, allow_pickle=True)
        self.flux = data['flux'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        
        if 'delta_t' in data:
            self.dt = data['delta_t'].astype(np.float32)
        else:
            print("Warning: delta_t not found, assuming constant cadence.")
            self.dt = np.ones_like(self.flux)
            self.dt[:, 0] = 0
            
        # Squeeze if necessary (B, 1, T) -> (B, T)
        if self.flux.ndim == 3: self.flux = self.flux.squeeze(1)
        if self.dt.ndim == 3: self.dt = self.dt.squeeze(1)
            
        # Calculate lengths
        self.lengths = (self.flux != 0.0).sum(axis=1)
        self.lengths = np.maximum(self.lengths, 1)
        
        self.hooks = HookManager()
        
    def reconstruct_trajectory(self, f, d, l):
        """
        Mathematically reconstructs the probability trajectory for the whole sequence.
        
        Since the model outputs a single prediction at the end, we hook the 
        GRU output sequence and manually pass every timestep through the 
        classification heads. This is 100x faster than looping inference.
        """
        self.hooks.clear()
        # Register hook on the GRU to get the full sequence of hidden states
        handle = self.hooks.register_hook(self.model.gru, 'gru_seq')
        
        with torch.no_grad():
            # Forward pass to trigger hooks
            _ = self.model(f, d, lengths=l)
            
            # shape: (1, T, hidden_size)
            gru_seq = self.hooks.outputs['gru_seq']
            
            # Apply Final Norm (Shared across time)
            normed_seq = self.model.norm_final(gru_seq)
            
            # Get Temperature
            temp = F.softplus(self.model.raw_temperature).clamp(min=0.1, max=10.0)
            
            if self.config.hierarchical:
                # Apply Heads to every timestep
                dev_logits = self.model.head_deviation(normed_seq) / temp
                type_logits = self.model.head_type(normed_seq) / temp
                
                dev_log_probs = F.log_softmax(dev_logits, dim=-1)
                type_log_probs = F.log_softmax(type_logits, dim=-1)
                
                # Reconstruct Hierarchical Logic
                # 0: Flat, 1: PSPL, 2: Binary
                log_p_flat = dev_log_probs[:, :, 0:1]
                log_p_dev_yes = dev_log_probs[:, :, 1:2]
                
                log_p_classA = log_p_dev_yes + type_log_probs[:, :, 0:1] # PSPL
                log_p_classB = log_p_dev_yes + type_log_probs[:, :, 1:2] # Binary
                
                final_log_probs = torch.cat([log_p_flat, log_p_classA, log_p_classB], dim=-1)
                probs = torch.exp(final_log_probs)
                
                aux = {
                    'dev_probs': torch.exp(dev_log_probs),
                    'type_probs': torch.exp(type_log_probs)
                }
            else:
                logits = self.model.classifier(normed_seq) / temp
                probs = F.softmax(logits, dim=-1)
                aux = {}
                
        handle.remove()
        return probs[0].cpu().numpy(), aux, normed_seq[0].cpu().numpy()

    def plot_deep_analysis(self, idx):
        """
        Generates the "God Mode" diagnostic plot for a single event.
        Includes Flux, Probability Trajectory, Hierarchical Logic, and Embedding Norm.
        """
        print(f"Deep Analysis for Event {idx}...")
        
        # Prepare Data
        f = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d = torch.tensor(self.dt[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        
        true_len = self.lengths[idx]
        true_cls = self.labels[idx]
        
        # Get Trajectories
        probs, aux, embeddings = self.reconstruct_trajectory(f, d, l)
        
        # --- PLOTTING ---
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Flux Lightcurve
        ax1 = fig.add_subplot(gs[0, :])
        time_axis = np.arange(len(probs))
        ax1.scatter(time_axis[:true_len], self.flux[idx][:true_len], 
                   c='k', s=15, alpha=0.6, label='Flux Data')
        ax1.set_title(f"Event {idx} | True Class: {CLASS_NAMES[true_cls]}", fontweight='bold', fontsize=14)
        ax1.set_ylabel("Flux (Standardized)", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 2. Probability Evolution (The "Mind" of the Model)
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        for c in range(3):
            ax2.plot(probs[:true_len, c], color=COLORS[c], linewidth=2.5, 
                    label=f'P({CLASS_NAMES[c]})', alpha=0.9)
        
        ax2.axhline(y=0.5, color='k', linestyle=':', alpha=0.3)
        ax2.set_ylabel("Probability", fontweight='bold')
        ax2.set_title("Model Confidence Evolution", fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # 3. Hierarchical Internals (If applicable)
        if self.config.hierarchical and aux:
            # 3a. Deviation Head (Is there something?)
            ax3 = fig.add_subplot(gs[2, 0])
            dev_probs = aux['dev_probs'][0].cpu().numpy()
            ax3.plot(dev_probs[:true_len, 1], color='purple', linewidth=2)
            ax3.set_title("Head 1: Deviation Detection\n(Flat vs. Event)", fontweight='bold')
            ax3.set_ylabel("P(Event)")
            ax3.set_xlabel("Time")
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            # 3b. Type Head (What is it?)
            ax4 = fig.add_subplot(gs[2, 1])
            type_probs = aux['type_probs'][0].cpu().numpy()
            ax4.plot(type_probs[:true_len, 0], color=COLORS[1], label='PSPL', linewidth=2)
            ax4.plot(type_probs[:true_len, 1], color=COLORS[2], label='Binary', linewidth=2)
            ax4.set_title("Head 2: Classification\n(PSPL vs. Binary)", fontweight='bold')
            ax4.set_xlabel("Time")
            ax4.set_ylim(0, 1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 4. Padding Integrity Check (God Mode Validation)
        ax5 = fig.add_subplot(gs[2, 2])
        emb_norms = np.linalg.norm(embeddings, axis=1)
        
        # Plot Valid Region
        ax5.plot(np.arange(true_len), emb_norms[:true_len], color='green', label='Valid')
        
        # Plot Padding Region (if exists)
        if true_len < len(emb_norms):
            pad_norms = emb_norms[true_len:]
            ax5.plot(np.arange(true_len, len(emb_norms)), pad_norms, color='red', label='Padding')
            
            mean_pad_noise = pad_norms.mean()
            status = "PASS" if mean_pad_noise < 1e-4 else "FAIL"
            color = 'green' if status == "PASS" else 'red'
            
            ax5.text(0.5, 0.9, f"Padding Leakage: {mean_pad_noise:.2e}\nStatus: {status}", 
                    transform=ax5.transAxes, color=color, fontweight='bold', ha='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
            
        ax5.set_title("Padding Integrity Check\n(Norm should be 0)", fontweight='bold')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.out_dir / f"event_{idx}_deep_dive.png", dpi=150)
        plt.close(fig)

    def analyze_early_detection(self, n=500):
        """
        Aggregates confidence trajectories to validate Early Detection capabilities.
        Crucial for Roman Telescope alert triggers.
        """
        print("Analyzing Early Detection Capabilities...")
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        
        trajs = defaultdict(list)
        
        for i in tqdm(idxs, desc="Tracing"):
            f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
            
            probs, _, _ = self.reconstruct_trajectory(f, d, l)
            
            true_cls = self.labels[i]
            # Get confidence in the TRUE class over time
            conf_curve = probs[:self.lengths[i], true_cls]
            
            # Normalize time to 0-100%
            trajs[true_cls].append(interpolate_traj(conf_curve))
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for c in range(3):
            if not trajs[c]: continue
            
            arr = np.array(trajs[c])
            mu = np.mean(arr, axis=0)
            sigma = np.std(arr, axis=0)
            x = np.linspace(0, 100, 100)
            
            ax.plot(x, mu, color=COLORS[c], linewidth=3, label=f'{CLASS_NAMES[c]} (Avg)')
            ax.fill_between(x, mu-sigma, mu+sigma, color=COLORS[c], alpha=0.15)
            
        ax.axhline(0.5, color='k', linestyle='--')
        ax.set_xlabel("% of Lightcurve Observed", fontweight='bold')
        ax.set_ylabel("Confidence in True Class", fontweight='bold')
        ax.set_title("Early Detection Profile: How fast does the model learn?", fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.out_dir / "early_detection_profile.png", dpi=300)
        plt.close(fig)

    def plot_latent_space(self, n=2000):
        """
        PCA of the final embedding vectors.
        """
        print("Mapping Latent Space...")
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        
        embeddings = []
        labels = []
        
        # Batch processing for speed
        batch_size = 128
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            
            f = torch.tensor(self.flux[batch_idxs], dtype=torch.float32).to(self.device)
            d = torch.tensor(self.dt[batch_idxs], dtype=torch.float32).to(self.device)
            l = torch.tensor(self.lengths[batch_idxs], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                out = self.model(f, d, lengths=l) # Regular forward pass gets features
                
                # In model.py, forward calls _hierarchical_inference
                # We need to extract the raw features before the heads.
                # But GodModeCausalGRU doesn't expose them in the output dict of forward()
                # Hack: Use the hook on norm_final again.
                
                # Hook is cleaner
                pass
        
        # Actually, let's just loop with the reconstruction helper which is robust
        for i in tqdm(idxs, desc="Extracting"):
            f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
            
            _, _, seq_emb = self.reconstruct_trajectory(f, d, l)
            # Take final valid embedding
            final_emb = seq_emb[self.lengths[i]-1]
            embeddings.append(final_emb)
            labels.append(self.labels[i])
            
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for c in range(3):
            mask = labels == c
            ax.scatter(proj[mask, 0], proj[mask, 1], c=COLORS[c], label=CLASS_NAMES[c],
                      alpha=0.6, s=25, edgecolors='w', linewidth=0.5)
            
        ax.set_title(f"Latent Space PCA (Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)", 
                    fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(self.out_dir / "latent_space_pca.png", dpi=300)
        plt.close(fig)

    def generate_confusion_matrix(self, n=2000):
        print("Generating Confusion Matrix...")
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        
        y_true = []
        y_pred = []
        
        # Batch inference
        batch_size = 64
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            f = torch.tensor(self.flux[batch_idxs], dtype=torch.float32).to(self.device)
            d = torch.tensor(self.dt[batch_idxs], dtype=torch.float32).to(self.device)
            l = torch.tensor(self.lengths[batch_idxs], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                out = self.model(f, d, lengths=l)
                preds = out['probs'].argmax(dim=1).cpu().numpy()
                y_true.extend(self.labels[batch_idxs])
                y_pred.extend(preds)
                
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        
        ax.set_ylabel('True Class', fontweight='bold')
        ax.set_xlabel('Predicted Class', fontweight='bold')
        ax.set_title("Normalized Confusion Matrix", fontweight='bold')
        
        plt.savefig(self.out_dir / "confusion_matrix.png", dpi=300)
        plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="God Mode Visualizer")
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--n_events', type=int, default=5, help="Number of deep dive plots")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Locate Experiment
    res_roots = [Path('../results'), Path('results'), Path('.')]
    exp_path = None
    for r in res_roots:
        matches = list(r.glob(f"{args.experiment_name}*"))
        if matches:
            exp_path = sorted(matches)[-1]
            break
            
    if not exp_path:
        print(f"Error: Experiment {args.experiment_name} not found.")
        sys.exit(1)
        
    model_file = exp_path / "best_model.pt"
    if not model_file.exists():
        print(f"Error: best_model.pt not found in {exp_path}")
        sys.exit(1)
        
    print(f"Initializing Visualizer for {args.experiment_name}...")
    viz = GodModeVisualizer(model_file, args.data, exp_path / "vis", device=args.device)
    
    # 1. Deep Dives
    idxs = np.random.choice(len(viz.flux), args.n_events, replace=False)
    for i in idxs:
        viz.plot_deep_analysis(i)
        
    # 2. Early Detection Profile
    viz.analyze_early_detection()
    
    # 3. Latent Space
    viz.plot_latent_space()
    
    # 4. Confusion Matrix
    viz.generate_confusion_matrix()
    
    print("\nVisualizations Generated Successfully.")
    print(f"Output: {exp_path / 'vis'}")

if __name__ == "__main__":
    main()
