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
import gc
import h5py

# Dynamic import setup
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Global configuration
np.random.seed(42)
torch.manual_seed(42)
sns.set_style("whitegrid")
COLORS = ['#95a5a6', '#e74c3c', '#3498db'] 
CLASS_NAMES = ['Flat', 'PSPL', 'Binary']

# Physical constants
AB_ZEROPOINT_JY = 3631.0
MISSION_DURATION_DAYS = 1826.25


# =============================================================================
# ROMAN VISUALIZER
# =============================================================================
class RomanVisualizer:
    """
    Advanced visualization suite for Roman microlensing classifier.
    
    Capabilities:
        - Latent space visualization via PCA
        - Deep event analysis with probability trajectories
        - Embedding norm analysis
        - Feature extraction visualization
    """
    
    def __init__(
        self, 
        model_path: str, 
        data_path: str, 
        output_dir: str, 
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: Path to model checkpoint (.pt file)
            data_path: Path to test data (.npz file)
            output_dir: Directory for output visualizations
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("ROMAN SPACE TELESCOPE - VISUALIZATION SUITE")
        print("=" * 80)
        print(f"Model:      {Path(model_path).name}")
        print(f"Data:       {Path(data_path).name}")
        print(f"Output:     {self.out_dir}")
        print(f"Device:     {self.device}")
        
        # Load model
        self.model, self.config, self.checkpoint = self._load_model(model_path)
        
        # Load data
        self.flux, self.delta_t, self.labels, self.lengths = self._load_data(data_path)
        
        # =====================================================================
        # CRITICAL FIX: APPLY NORMALIZATION FROM CHECKPOINT
        # =====================================================================
        # The model was trained on normalized data (approx mean=0, std=1).
        # We must apply the EXACT same statistics to the visualization data.
        if 'normalization_stats' in self.checkpoint:
            stats = self.checkpoint['normalization_stats']
            print(f"Applying normalization from checkpoint: median={stats['median']:.4f}, iqr={stats['iqr']:.4f}")
            
            # Create mask for valid data (non-zero, non-nan)
            mask = (self.flux != 0) & (~np.isnan(self.flux))
            
            # Apply (Flux - Median) / IQR
            # This matches the logic in train.py exactly
            self.flux = np.where(
                mask, 
                (self.flux - stats['median']) / stats['iqr'], 
                0.0
            )
        else:
            print("⚠️ WARNING: No normalization stats found in checkpoint.")
            print("   Visualizations will likely be incorrect (garbage in -> garbage out).")
            print("   Ensure you are using a checkpoint trained with the latest train.py.")
        # =====================================================================
        
        # Hook for feature extraction
        self.hook_handle = None
        self.hook_output = {}
        
        print("=" * 80 + "\n")

    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        print(f"\nLoading model from {Path(model_path).name}...")
        
        try:
            from model import RomanMicrolensingGRU, ModelConfig
        except ImportError as e:
            print(f"CRITICAL: Cannot import model: {e}")
            print("Make sure model.py is in the same directory")
            sys.exit(1)
        
        try:
            ckpt = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"CRITICAL: Failed to load checkpoint: {e}")
            sys.exit(1)
        
        # Extract config
        config_dict = ckpt.get('config', {})
        valid_keys = set(ModelConfig.__annotations__.keys())
        clean_conf = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = ModelConfig(**clean_conf)
        
        # Create model
        model = RomanMicrolensingGRU(config, dtype=torch.float32).to(self.device)
        
        # Load weights
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        clean_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"  Config: d_model={config.d_model}, n_layers={config.n_layers}")
        
        return model, config, ckpt

    def _load_data(self, data_path: str):
        """Load data from HDF5 or NPZ."""
        data_path = Path(data_path)
        
        if data_path.suffix in ['.h5', '.hdf5']:
            print(f"Loading HDF5 data: {data_path.name}")
            
            with h5py.File(data_path, 'r') as f:
                flux = f['flux'][:]
                delta_t = f['delta_t'][:]
                labels = f['labels'][:]
                
                # Compute lengths
                valid_mask = (flux != 0) & (~np.isnan(flux))
                lengths = valid_mask.sum(axis=1)
        
        elif data_path.suffix == '.npz':
            print(f"Loading NPZ data: {data_path.name}")
            print("⚠️  WARNING: NPZ is 37x slower than HDF5")
            
            data = np.load(data_path)
            flux = data['flux']
            delta_t = data['delta_t']
            labels = data['labels']
            
            valid_mask = (flux != 0) & (~np.isnan(flux))
            lengths = valid_mask.sum(axis=1)
        
        else:
            raise ValueError(f"Unsupported format: {data_path.suffix}")
        
        return flux, delta_t, labels, lengths
    
    def _compute_normalization_stats(self, flux: np.ndarray) -> dict:
        """Compute normalization statistics matching train.py."""
        # Kept for compatibility, though we prioritize checkpoint stats
        subset = flux[:10000].flatten()
        subset = subset[subset != 0]
        
        if len(subset) == 0:
            return {'median': 0.0, 'iqr': 1.0}
        
        median = float(np.median(subset))
        q75, q25 = np.percentile(subset, [75, 25])
        iqr = float(q75 - q25)
        
        if iqr < 1e-6:
            iqr = 1.0
        
        return {'median': median, 'iqr': iqr}

    def _register_hook(self):
        """Register forward hook to capture GRU embeddings."""
        def hook(module, input, output):
            # Capture GRU output sequence
            self.hook_output['gru_seq'] = output[0].detach()
        
        # Hook into the GRU stack
        if hasattr(self.model, 'gru'):
            self.hook_handle = self.model.gru.register_forward_hook(hook)

    def _remove_hook(self):
        """Remove forward hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

    def extract_trajectory(self, flux_idx: int):
        """
        Extract probability and embedding trajectory for single event.
        
        Args:
            flux_idx: Index of event in dataset
            
        Returns:
            probs: (T, 3) probability array
            embeddings: (T, d_model) embedding array
        """
        f = torch.tensor(self.flux[flux_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        d = torch.tensor(self.delta_t[flux_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        l = torch.tensor([self.lengths[flux_idx]], dtype=torch.long).to(self.device)
        
        self._register_hook()
        try:
            with torch.no_grad():
                output = self.model(f, d, lengths=l, return_all_timesteps=True)
                
                # Get embeddings from hook
                embeddings = self.hook_output.get('gru_seq', None)
                if embeddings is None:
                    embeddings = torch.zeros(1, f.size(1), self.config.d_model).to(self.device)
                
                # Get probabilities
                if 'probs_seq' in output:
                    probs = output['probs_seq']
                else:
                    # Fallback: compute from logits
                    if self.config.hierarchical:
                        probs = torch.exp(output['logits_seq'])
                    else:
                        probs = F.softmax(output['logits_seq'], dim=-1)
                    
        finally:
            self._remove_hook()
        
        return probs[0].cpu().numpy(), embeddings[0].cpu().numpy()

    def plot_deep_analysis(self, idx: int):
        """Generate comprehensive deep analysis plot for single event."""
        print(f"  Plotting Event {idx}...")
        
        probs, embeddings = self.extract_trajectory(idx)
        T = self.lengths[idx]
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Flux time series
        ax1 = fig.add_subplot(gs[0, :])
        # Plot only valid part
        ax1.plot(self.flux[idx][:T], c='k', alpha=0.7, linewidth=1.5, label='Normalized Flux')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_title(f"Event {idx} | True Class: {CLASS_NAMES[self.labels[idx]]}")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Probability evolution
        ax2 = fig.add_subplot(gs[1, :])
        for c in range(3):
            ax2.plot(probs[:T, c], color=COLORS[c], label=CLASS_NAMES[c], lw=2)
        ax2.set_ylim([-0.05, 1.05])
        ax2.set_ylabel('Probability')
        ax2.set_title("Probability Evolution")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Latent state activity (L2 norm)
        ax3 = fig.add_subplot(gs[2, 0])
        norms = np.linalg.norm(embeddings[:T], axis=1)
        ax3.plot(norms, color='purple', lw=2)
        ax3.set_ylabel('Embedding Norm')
        ax3.set_xlabel('Time Step')
        ax3.set_title("Latent State Activity")
        ax3.grid(True, alpha=0.3)
        
        # 4. Confidence over time
        ax4 = fig.add_subplot(gs[2, 1])
        confidence = probs[:T].max(axis=1)
        ax4.plot(confidence, color='orange', lw=2)
        ax4.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='High Confidence')
        ax4.set_ylabel('Max Probability')
        ax4.set_xlabel('Time Step')
        ax4.set_title("Classification Confidence")
        ax4.set_ylim([0, 1.05])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.out_dir / f"deep_analysis_{idx}.png", dpi=300)
        plt.close()
        gc.collect()

    def plot_latent_space(self, n: int = 2000, method: str = 'pca'):
        """
        Visualize latent space using dimensionality reduction.
        
        Args:
            n: Number of samples to visualize
            method: Dimensionality reduction method ('pca' supported)
        """
        print(f"\nMapping latent space ({n} samples)...")
        
        n = min(len(self.flux), n)
        idxs = np.random.choice(len(self.flux), n, replace=False)
        
        embeddings = []
        labels = []
        
        self._register_hook()
        
        with torch.no_grad():
            # Batch processing for efficiency
            batch_size = 128
            for i in tqdm(range(0, n, batch_size), desc="Extracting embeddings"):
                batch_end = min(i + batch_size, n)
                batch_idx = idxs[i:batch_end]
                
                f = torch.tensor(self.flux[batch_idx], dtype=torch.float32).to(self.device)
                d = torch.tensor(self.delta_t[batch_idx], dtype=torch.float32).to(self.device)
                l = torch.tensor(self.lengths[batch_idx], dtype=torch.long).to(self.device)
                
                # Forward pass
                self.model(f, d, lengths=l)
                
                # Extract last valid embedding for each sequence
                seq_emb = self.hook_output['gru_seq']  # (B, T, D)
                
                # Gather last valid timestep
                last_idx = (l - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, seq_emb.size(-1))
                last_emb = seq_emb.gather(1, last_idx).squeeze(1)  # (B, D)
                
                embeddings.append(last_emb.cpu().numpy())
                labels.extend(self.labels[batch_idx])
        
        self._remove_hook()
        
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.array(labels)
        
        if len(embeddings) < 5:
            print("  Insufficient samples for visualization")
            return
        
        # Apply dimensionality reduction
        print(f"  Applying {method.upper()} projection...")
        if method == 'pca':
            reducer = PCA(n_components=2)
            proj = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
            print(f"    Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        plt.figure(figsize=(12, 9))
        
        for c in range(3):
            mask = labels == c
            if mask.sum() > 0:
                plt.scatter(
                    proj[mask, 0], proj[mask, 1], 
                    c=COLORS[c], label=CLASS_NAMES[c], 
                    alpha=0.6, s=20, edgecolors='none'
                )
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Latent Space Visualization ({n} samples)')
        plt.legend(loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.out_dir / f'latent_space_{method}.png', dpi=300)
        plt.close()
        
        print(f"  Saved to: {self.out_dir / f'latent_space_{method}.png'}")

    def plot_confusion_with_embeddings(self, n_per_class: int = 100):
        """
        Plot confusion matrix with latent space visualization of errors.
        """
        print(f"\nAnalyzing classification errors ({n_per_class} per class)...")
        
        # Get predictions
        all_probs = []
        
        with torch.no_grad():
            batch_size = 128
            for i in tqdm(range(0, len(self.flux), batch_size), desc="Inference"):
                batch_end = min(i + batch_size, len(self.flux))
                
                f = torch.tensor(self.flux[i:batch_end], dtype=torch.float32).to(self.device)
                d = torch.tensor(self.delta_t[i:batch_end], dtype=torch.float32).to(self.device)
                l = torch.tensor(self.lengths[i:batch_end], dtype=torch.long).to(self.device)
                
                output = self.model(f, d, lengths=l)
                all_probs.append(output['probs'].cpu().numpy())
        
        probs = np.concatenate(all_probs, axis=0)
        preds = probs.argmax(axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(self.labels, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Confusion matrix
        sns.heatmap(
            cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax1, cbar_kws={'label': 'Fraction'}
        )
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Error distribution
        correct = (self.labels == preds)
        error_counts = [np.sum(~correct & (self.labels == i)) for i in range(3)]
        ax2.bar(CLASS_NAMES, error_counts, color=COLORS, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Number of Errors')
        ax2.set_title('Classification Errors by Class')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.out_dir / 'confusion_analysis.png', dpi=300)
        plt.close()
        
        print(f"  Accuracy: {correct.mean():.4f}")
        print(f"  Saved to: {self.out_dir / 'confusion_analysis.png'}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualization suite for Roman microlensing classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--experiment_name', required=True,
                       help="Name of experiment (searches for model automatically)")
    parser.add_argument('--data', required=True,
                       help="Path to test data (.npz or .h5)")
    parser.add_argument('--output_dir', required=True,
                       help="Output directory for plots")
    
    parser.add_argument('--n_latent', type=int, default=2000,
                       help="Number of samples for latent space visualization")
    parser.add_argument('--n_deep_dives', type=int, default=10,
                       help="Number of deep dive analysis plots")
    parser.add_argument('--device', default='cuda',
                       help="Device: cuda or cpu")
    
    args = parser.parse_args()
    
    # Find model path
    print("Searching for experiment...")
    search_roots = [Path('../results'), Path('results'), Path('.')]
    exp_path = None
    
    for root in search_roots:
        if root.exists():
            matches = list(root.glob(f"*{args.experiment_name}*"))
            if matches:
                exp_path = sorted(matches, key=lambda x: x.stat().st_mtime)[-1]
                break
    
    if not exp_path:
        print(f"Error: Experiment '{args.experiment_name}' not found")
        sys.exit(1)
    
    model_file = exp_path / "best_model.pt"
    if not model_file.exists():
        print(f"Error: No best_model.pt found in {exp_path}")
        sys.exit(1)
    
    # Create visualizer
    viz = RomanVisualizer(
        model_path=str(model_file),
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Latent space
    viz.plot_latent_space(n=args.n_latent)
    
    # Confusion with embeddings
    viz.plot_confusion_with_embeddings(n_per_class=100)
    
    # Deep dives
    print(f"\nGenerating {args.n_deep_dives} deep analysis plots...")
    n_samples = min(len(viz.flux), args.n_deep_dives)
    indices = np.random.choice(len(viz.flux), n_samples, replace=False)
    
    for idx in tqdm(indices, desc="Deep dives"):
        viz.plot_deep_analysis(idx)
    
    print("\n" + "=" * 80)
    print(f"Visualization complete. Results saved to: {viz.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
