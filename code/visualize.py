import os
import sys
import torch
import torch.nn as nn
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
from torch.cuda.amp import autocast
from collections import defaultdict

# --- Dynamic Import ---
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

try:
    from model import CausalHybridModel, CausalConfig
except ImportError:
    print("CRITICAL: 'model.py' not found.")
    sys.exit(1)

# Set visualization seed for reproducibility
np.random.seed(42)

sns.set_style("whitegrid")
COLORS = ['#95a5a6', '#e74c3c', '#3498db']  # Grey, Red, Blue

# =============================================================================
# UTILS
# =============================================================================
class HookManager:
    """
    Manages forward hooks with validation for robust visualization.
    Ensures hooks are properly registered and outputs are captured.
    """
    def __init__(self):
        self.outputs = {}
        self.hooks = []

    def register_hook(self, module, name):
        """Register forward hook with validation."""
        if module is None:
            raise ValueError(f"Cannot register hook '{name}': module is None")
        
        def hook(model, input, output):
            if isinstance(output, tuple):
                data = output[0]
            else:
                data = output
            self.outputs[name] = data.detach().cpu()
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)
        return handle

    def validate_outputs(self, required_keys):
        """Validate that expected hooks were triggered."""
        missing = [k for k in required_keys if k not in self.outputs]
        if missing:
            raise RuntimeError(
                f"Hook outputs missing: {missing}. "
                f"Model architecture may have changed. "
                f"Available keys: {list(self.outputs.keys())}"
            )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.outputs = {}

def create_delta_t(ts):
    """Create causal time intervals from timestamps."""
    if ts.ndim == 1:
        ts = ts[np.newaxis, :]
    dt = np.zeros_like(ts, dtype=np.float32)
    if ts.shape[1] > 1:
        dt[:, 1:] = np.diff(ts, axis=1)
    return np.maximum(dt, 0.0)

def interpolate_traj(y, n=100):
    """Interpolate trajectory to fixed length for aggregation."""
    if len(y) < 2:
        return np.zeros(n)
    x = np.linspace(0, 1, len(y))
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f(np.linspace(0, 1, n))

# =============================================================================
# VISUALIZER
# =============================================================================
class GodModeVisualizer:
    """
    Comprehensive visualization suite for CausalHybridModel.
    
    Features:
    - Per-event deep analysis with causality verification
    - Confidence trajectory analysis for early detection validation
    - Latent space PCA for representation quality
    - Confusion matrix and confidence distributions
    - Roman Telescope readiness validation
    
    The "God Mode Check" validates padding integrity, ensuring zero-masking
    effectiveness - critical for thesis defense.
    """
    def __init__(self, model_path, data_path, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Visualization seed set to 42 for reproducibility")
        
        # Load Model
        print(f"Loading Model: {Path(model_path).name}...")
        ckpt = torch.load(model_path, map_location=self.device)
        config_dict = ckpt.get('config', {})
        
        # Filter config to valid keys only
        valid_keys = CausalConfig().__dict__.keys()
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
        self.config = CausalConfig(**clean_conf)
        
        self.model = CausalHybridModel(self.config).to(self.device)
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(clean_state, strict=False)
        self.model.eval()
        
        print(f"Model loaded: {self.config.d_model}d, RF={self.model.get_receptive_field()}")
        
        # Load Data
        print("Loading Data...")
        data = np.load(data_path, allow_pickle=True)
        self.flux = data.get('flux', data.get('X')).astype(np.float32)
        self.y = data.get('labels', data.get('y'))
        
        if 'delta_t' in data:
            self.dt = data['delta_t'].astype(np.float32)
        else:
            self.dt = create_delta_t(data.get('timestamps'))
        
        if self.flux.ndim == 3:
            self.flux = self.flux.squeeze(1)
        if self.dt.ndim == 3:
            self.dt = self.dt.squeeze(1)
        
        # FIXED: Validate data shapes
        assert self.flux.shape == self.dt.shape, \
            f"Shape mismatch: flux={self.flux.shape}, dt={self.dt.shape}"
        assert self.flux.shape[1] <= self.config.max_seq_len, \
            f"Sequence length {self.flux.shape[1]} exceeds model max {self.config.max_seq_len}"
        
        # Lengths (Strict 0.0 is padding)
        self.lengths = (self.flux != 0.0).sum(axis=1)
        self.lengths = np.maximum(self.lengths, 1)
        
        print(f"Data validated: {len(self.flux)} events, seq_len={self.flux.shape[1]}")
        
        self.hooks = HookManager()

    def plot_single_event_analysis(self, idx):
        """
        Deep analysis of single event with causality validation.
        
        Visualizes:
        1. Flux and probability evolution
        2. Model architecture info
        3. Confidence in true class over time
        4. Padding integrity check (God Mode validation)
        5. Receptive field warmup period
        
        Critical for thesis: Demonstrates zero-masking effectiveness.
        """
        print(f"Visualizing Event {idx}...")
        
        # Prepare Input
        f = torch.tensor(self.flux[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d = torch.tensor(self.dt[idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l = torch.tensor([self.lengths[idx]], dtype=torch.long).to(self.device)
        true_len = self.lengths[idx]
        
        # Register Hooks
        self.hooks.remove_hooks()
        self.hooks.register_hook(self.model.final_norm, 'final_emb')
        
        with torch.no_grad(), autocast():
            out = self.model(f, d, lengths=l, return_all_timesteps=True)
            probs = out['probs'][0].cpu().numpy()
        
        # CRITICAL: Validate hook captured data
        self.hooks.validate_outputs(['final_emb'])
        
        # --- FIGURE SETUP ---
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # 1. Flux & Classification
        ax1 = fig.add_subplot(gs[0, :])
        time_axis = np.arange(len(probs))
        ax1.scatter(time_axis[:true_len], self.flux[idx][:true_len], 
                   c='black', s=10, label='Flux', alpha=0.5, zorder=1)
        
        cls_names = ['Flat', 'PSPL', 'Binary']
        for c in range(3):
            ax1.plot(probs[:true_len, c], label=f'Prob({cls_names[c]})', 
                    color=COLORS[c], linewidth=2, zorder=2)
        
        # FIXED: Highlight receptive field warmup period
        rf = self.model.get_receptive_field()
        if rf > 1 and rf < true_len:
            ax1.axvspan(0, rf-1, color='yellow', alpha=0.15, 
                       label=f'RF Warmup ({rf} steps)', zorder=0)
            ax1.text(rf/2, ax1.get_ylim()[1] * 0.95, 
                    f"Partial Context\n(RF={rf})", 
                    ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax1.axvline(x=true_len-1, color='k', linestyle='--', 
                   label='End of Seq', linewidth=1.5, zorder=3)
        ax1.set_title(f"Event {idx} (True: {cls_names[self.y[idx]]})", fontweight='bold')
        ax1.set_xlabel("Time", fontweight='bold')
        ax1.set_ylabel("Probability / Flux", fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Model Architecture Info
        ax2 = fig.add_subplot(gs[1, 0])
        info_text = (
            f"Causal Hybrid Model\n"
            f"{'='*30}\n"
            f"Receptive Field: {rf} timesteps\n"
            f"d_model: {self.config.d_model}\n"
            f"Transformer Layers: {self.config.n_transformer_layers}\n"
            f"CNN Layers: {self.config.n_conv_layers}\n"
            f"Attention Heads: {self.config.n_heads}\n"
            f"Kernel Size: {self.config.kernel_size}\n"
            f"Dropout: {self.config.dropout}"
        )
        ax2.text(0.5, 0.5, info_text,
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title("Model Configuration", fontweight='bold')
        ax2.axis('off')
        
        # 3. Confidence Evolution
        ax3 = fig.add_subplot(gs[1, 1])
        true_class = self.y[idx]
        confidence = probs[:true_len, true_class]
        ax3.plot(confidence, color='purple', linewidth=2, marker='o', 
                markersize=2, alpha=0.7)
        ax3.set_title(f"Confidence in True Class ({cls_names[true_class]})", 
                     fontweight='bold')
        ax3.set_xlabel("Time", fontweight='bold')
        ax3.set_ylabel("Probability", fontweight='bold')
        ax3.axvline(x=true_len-1, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
        
        # Highlight RF warmup in confidence plot
        if rf > 1 and rf < true_len:
            ax3.axvspan(0, rf-1, color='yellow', alpha=0.1)
        
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Padding Integrity Check (THE GOD MODE CHECK)
        ax4 = fig.add_subplot(gs[2, :])
        emb_norm = self.hooks.outputs['final_emb'][0].norm(dim=-1).numpy()
        ax4.plot(emb_norm, color='green', linewidth=2, label='Feature Vector Norm')
        
        # Highlight Padding Region
        pad_start = true_len
        if pad_start < len(emb_norm):
            ax4.axvspan(pad_start, len(emb_norm), color='red', alpha=0.1, 
                       label='Padding Region')
            
            # Check mean norm in padding
            pad_noise = emb_norm[pad_start:].mean()
            status = "✓ PASS" if pad_noise < 1e-4 else "✗ FAIL"
            status_color = 'green' if pad_noise < 1e-4 else 'red'
            
            ax4.text(pad_start + (len(emb_norm) - pad_start) / 2, 
                    ax4.get_ylim()[1] * 0.5, 
                    f"Padding Noise: {pad_noise:.2e}\n{status}", 
                    color=status_color, fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', 
                             edgecolor=status_color, alpha=0.95, linewidth=2.5))
        
        # Highlight valid region
        ax4.axvspan(0, true_len, color='green', alpha=0.05, label='Valid Region')
        
        ax4.set_title("Causality Check: Feature Integrity (Should be ~0 in Padding)", 
                     fontweight='bold')
        ax4.set_xlabel("Time", fontweight='bold')
        ax4.set_ylabel("L2 Norm", fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')  # Log scale for better visualization

        plt.tight_layout()
        plt.savefig(self.out_dir / f"deep_analysis_{idx}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.hooks.remove_hooks()

    def analyze_confidence_trajectory(self, n=500):
        """
        Analyze confidence evolution for early detection validation.
        
        Critical for Roman Telescope: Demonstrates at what % of event
        the model achieves reliable classification.
        """
        print("Analyzing Confidence Trajectories...")
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        trajs = {0: [], 1: [], 2: []}
        
        with torch.no_grad(), autocast():
            for i in tqdm(idxs, desc="Computing trajectories"):
                f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                
                out = self.model(f, d, lengths=l, return_all_timesteps=True)
                probs = out['probs'][0].cpu().numpy()
                
                true_c = int(self.y[i])
                # Confidence in true class over time
                conf = probs[:self.lengths[i], true_c]
                trajs[true_c].append(interpolate_traj(conf))
                
        fig, ax = plt.subplots(figsize=(10, 6))
        cls_names = ['Flat', 'PSPL', 'Binary']
        
        for c in range(3):
            if not trajs[c]:
                continue
            arr = np.array(trajs[c])
            mu = arr.mean(axis=0)
            sigma = arr.std(axis=0)
            x = np.linspace(0, 100, 100)
            
            ax.plot(x, mu, color=COLORS[c], label=f'{cls_names[c]} (n={len(trajs[c])})', 
                   linewidth=2.5)
            ax.fill_between(x, mu-sigma, mu+sigma, color=COLORS[c], alpha=0.15)
            
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1, 
                  label='Decision Threshold')
        ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.3, linewidth=1, 
                  label='High Confidence')
        ax.set_xlabel("% of Event Observed", fontweight='bold', fontsize=12)
        ax.set_ylabel("Confidence in True Class", fontweight='bold', fontsize=12)
        ax.set_title("Early Detection Capability: Confidence Evolution", 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.out_dir / "confidence_evolution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_latent_space(self, n=2000):
        """
        PCA visualization of learned representations.
        
        Validates that model learns separable feature representations.
        """
        print("Generating PCA of latent space...")
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        vecs, labs = [], []
        
        self.hooks.register_hook(self.model.final_norm, 'emb')
        with torch.no_grad(), autocast():
            for i in tqdm(idxs, desc="Extracting embeddings"):
                f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                
                _ = self.model(f, d, lengths=l)
                # Take last valid timestep
                v = self.hooks.outputs['emb'][0, self.lengths[i]-1].numpy()
                vecs.append(v)
                labs.append(self.y[i])
        
        self.hooks.validate_outputs(['emb'])
        self.hooks.remove_hooks()
        
        pca = PCA(n_components=2)
        proj = pca.fit_transform(np.array(vecs))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        labs = np.array(labs)
        cls_names = ['Flat', 'PSPL', 'Binary']
        
        for c in range(3):
            mask = labs == c
            ax.scatter(proj[mask, 0], proj[mask, 1], c=COLORS[c], 
                      label=f'{cls_names[c]} (n={mask.sum()})', 
                      alpha=0.6, s=20, edgecolors='white', linewidths=0.5)
            
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                     fontweight='bold', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                     fontweight='bold', fontsize=12)
        ax.set_title("Latent Space Visualization (PCA)", fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.out_dir / "latent_pca.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_confusion_matrix(self, n=1000):
        """
        Generate confusion matrix for final predictions.
        Essential thesis metric for overall model performance.
        """
        print("Computing confusion matrix...")
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        y_true, y_pred = [], []
        
        with torch.no_grad(), autocast():
            for i in tqdm(idxs, desc="Predictions"):
                f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                
                out = self.model(f, d, lengths=l, return_all_timesteps=False)
                pred = out['probs'][0].argmax().item()
                
                y_true.append(self.y[i])
                y_pred.append(pred)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Flat', 'PSPL', 'Binary'],
                   yticklabels=['Flat', 'PSPL', 'Binary'],
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
        ax.set_ylabel('True', fontweight='bold', fontsize=12)
        
        # Add accuracy to title
        overall_acc = cm.diagonal().sum() / cm.sum()
        ax.set_title(f'Confusion Matrix (Accuracy: {overall_acc:.3f})', 
                    fontweight='bold', fontsize=14)
        
        # Add per-class accuracy text
        info_text = '\n'.join([
            f'{cls}: {acc:.3f}' 
            for cls, acc in zip(['Flat', 'PSPL', 'Binary'], per_class_acc)
        ])
        ax.text(1.15, 0.5, f'Per-Class Acc:\n{info_text}', 
               transform=ax.transAxes, fontsize=10, va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.out_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_confidence_distributions(self, n=1000):
        """
        Plot final confidence distributions by true class.
        Reveals model calibration and per-class discrimination ability.
        """
        print("Analyzing confidence distributions...")
        confs = defaultdict(lambda: defaultdict(list))
        idxs = np.random.choice(len(self.flux), min(len(self.flux), n), replace=False)
        
        with torch.no_grad(), autocast():
            for i in tqdm(idxs, desc="Computing confidences"):
                f = torch.tensor(self.flux[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                d = torch.tensor(self.dt[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                l = torch.tensor([self.lengths[i]], dtype=torch.long).to(self.device)
                
                out = self.model(f, d, lengths=l, return_all_timesteps=False)
                probs = out['probs'][0].cpu().numpy()
                
                true_c = int(self.y[i])
                for pred_c in range(3):
                    confs[true_c][pred_c].append(probs[pred_c])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        cls_names = ['Flat', 'PSPL', 'Binary']
        
        for true_c in range(3):
            ax = axes[true_c]
            for pred_c in range(3):
                data = confs[true_c][pred_c]
                if data:
                    ax.hist(data, bins=30, alpha=0.6, color=COLORS[pred_c],
                           label=f'→{cls_names[pred_c]}', density=True)
            
            ax.set_title(f'True Class: {cls_names[true_c]}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Predicted Probability', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
        
        plt.suptitle('Confidence Distributions by True Class', 
                    fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.out_dir / "confidence_distributions.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def batch_process_events(self, batch_size=32):
        """
        Batch process all events for efficient large-scale analysis.
        Critical for Roman telescope data volumes (1M+ events).
        
        Returns:
            probs_all: (N, T, n_classes) probability arrays
            embeddings_all: (N, d_model) final embeddings
        """
        print(f"Batch processing {len(self.flux)} events with batch_size={batch_size}...")
        n_batches = (len(self.flux) + batch_size - 1) // batch_size
        
        all_probs = []
        all_embeds = []
        
        self.hooks.register_hook(self.model.final_norm, 'emb')
        
        with torch.no_grad(), autocast():
            for i in tqdm(range(n_batches), desc="Batch inference"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.flux))
                
                # Prepare batch
                f_batch = torch.tensor(self.flux[start_idx:end_idx], 
                                      dtype=torch.float32).to(self.device)
                d_batch = torch.tensor(self.dt[start_idx:end_idx], 
                                      dtype=torch.float32).to(self.device)
                l_batch = torch.tensor(self.lengths[start_idx:end_idx], 
                                      dtype=torch.long).to(self.device)
                
                # Forward pass
                out = self.model(f_batch, d_batch, lengths=l_batch, 
                               return_all_timesteps=True)
                
                all_probs.append(out['probs'].cpu().numpy())
                
                # Extract final embeddings per sequence
                B = f_batch.size(0)
                batch_indices = torch.arange(B, device=self.device)
                last_idx = (l_batch - 1).clamp(min=0)
                embeds = self.hooks.outputs['emb'][batch_indices, last_idx].cpu().numpy()
                all_embeds.append(embeds)
        
        self.hooks.validate_outputs(['emb'])
        self.hooks.remove_hooks()
        
        return np.concatenate(all_probs, axis=0), np.concatenate(all_embeds, axis=0)

# =============================================================================
# RUN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize CausalHybridModel predictions and internals (Roman Telescope Ready)"
    )
    parser.add_argument('--experiment_name', required=True, 
                       help='Experiment name to search in results directory')
    parser.add_argument('--data', required=True, 
                       help='Path to test data NPZ file')
    parser.add_argument('--n_events', type=int, default=3,
                       help='Number of individual events to visualize in detail')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Locate Experiment
    res_roots = [Path('../results'), Path('results'), Path('.')]
    exp_path = None
    for r in res_roots:
        if r.exists():
            matches = list(r.glob(f"{args.experiment_name}*"))
            if matches:
                exp_path = sorted(matches)[-1]
                break
            
    if not exp_path:
        print(f"ERROR: Experiment '{args.experiment_name}*' not found in {res_roots}")
        sys.exit(1)
    
    model_file = exp_path / "best_model.pt"
    if not model_file.exists():
        candidates = list(exp_path.glob("*.pt"))
        model_file = candidates[0] if candidates else None
    
    if not model_file:
        print(f"ERROR: No model file found in {exp_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Roman Telescope Causal Model Visualization")
    print(f"{'='*80}")
    print(f"Model: {model_file}")
    print(f"Data: {args.data}")
    print(f"Output: {exp_path / 'vis'}")
    print(f"{'='*80}\n")
    
    viz = GodModeVisualizer(str(model_file), args.data, str(exp_path / "vis"), 
                           device=args.device)
    
    # 1. Deep Analysis of individual events
    print(f"\n[1/5] Generating detailed analysis for {args.n_events} events...")
    idxs = np.random.choice(len(viz.flux), min(args.n_events, len(viz.flux)), replace=False)
    for i in idxs:
        viz.plot_single_event_analysis(i)
        
    # 2. Confidence Trajectories
    print("\n[2/5] Analyzing confidence trajectories (early detection)...")
    viz.analyze_confidence_trajectory()
    
    # 3. Latent Space
    print("\n[3/5] Generating latent space PCA...")
    viz.plot_latent_space()
    
    # 4. Confusion Matrix
    print("\n[4/5] Computing confusion matrix...")
    viz.plot_confusion_matrix()
    
    # 5. Confidence Distributions
    print("\n[5/5] Analyzing confidence distributions...")
    viz.plot_confidence_distributions()
    
    print(f"\n{'='*80}")
    print("✓ Visualization complete!")
    print(f"Results saved to: {exp_path / 'vis'}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
