#!/usr/bin/env python3
"""
FIXED - Visualize High-Confidence Binary Classifications
=========================================================
Uses CORRECT MicrolensingTransformer architecture
Matches training code exactly

Author: Kunal Bhatia
Version: FIXED for thesis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from tqdm import tqdm

sys.path.insert(0, '/pfs/data6/home/hd/hd_hd/hd_vm305/Thesis/code')
from transformer import MicrolensingTransformer  # CORRECT import!

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300


class StableNormalizer:
    """Same normalizer as training"""
    
    def __init__(self, pad_value=-1.0):
        self.pad_value = pad_value
        self.mean = 0.0
        self.std = 1.0
    
    def fit(self, X):
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            valid_values = X[valid_mask]
            self.mean = np.median(valid_values)
            self.std = np.median(np.abs(valid_values - self.mean))
            
            if self.std < 1e-8:
                self.std = 1.0
            
            self.mean = np.clip(self.mean, -100, 100)
            self.std = np.clip(self.std, 0.01, 100)
        
        return self
    
    def transform(self, X):
        X_norm = X.copy()
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            X_norm[valid_mask] = (X[valid_mask] - self.mean) / self.std
            X_norm[valid_mask] = np.clip(X_norm[valid_mask], -10, 10)
        
        return np.nan_to_num(X_norm, nan=0.0, posinf=10.0, neginf=-10.0)


class FixedVisualizer:
    def __init__(self, model_path, data_path, output_dir='./figures', max_samples=None, use_cuda=True):
        """
        Initialize visualizer with automatic architecture detection
        
        Args:
            model_path: Path to model checkpoint
            data_path: Path to test data (.npz)
            output_dir: Where to save figures
            max_samples: Limit number of samples (None = use all)
            use_cuda: Use GPU if available
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load and setup model
        print("\n📦 Loading model...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Load data
        print("\n📊 Loading data...")
        self.X, self.y, self.meta = self._load_data(data_path, max_samples)
        
        # Normalize data
        print("\n🔄 Normalizing data...")
        self.normalizer = StableNormalizer(pad_value=-1.0)
        self.X_norm = self.normalizer.fit(self.X).transform(self.X)
        
        # Get predictions
        print(f"\n🔮 Getting predictions for {len(self.X)} samples...")
        self.predictions, self.confidences = self._get_predictions()
        
        # Print summary
        self._print_summary()
    
    def _load_model(self, model_path):
        """Load model with auto-detected architecture"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Auto-detect architecture from checkpoint
        # Method 1: Try from pos_encoding
        if 'pos_encoding' in state_dict:
            d_model = state_dict['pos_encoding'].shape[2]
        # Method 2: Try from input_embed
        elif 'input_embed.0.weight' in state_dict:
            d_model = state_dict['input_embed.4.weight'].shape[0]  # Last layer of input_embed
        else:
            # Fallback
            d_model = 256
        
        # Count transformer layers
        num_layers = len([k for k in state_dict.keys() 
                         if 'layers.' in k and '.norm.weight' in k])
        
        # Detect nhead (should be from config, default 8)
        nhead = 8
        
        print(f"   Detected architecture:")
        print(f"   - d_model: {d_model}")
        print(f"   - num_layers: {num_layers}")
        print(f"   - nhead: {nhead}")
        print(f"   - dim_ff: {d_model * 4}")
        
        # Create model with CORRECT architecture
        model = MicrolensingTransformer(
            n_points=1500,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            pad_value=-1.0
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        
        return model
    
    def _load_data(self, data_path, max_samples):
        """Load and preprocess data"""
        data = np.load(data_path)
        
        # Load X and y
        X = data['X']
        y = data['y']
        
        # Handle different X shapes
        if X.ndim == 3:  # (N, 1, 1500)
            X = X.squeeze(1)
        elif X.ndim == 2:  # (N, 1500)
            pass
        else:
            raise ValueError(f"Unexpected X shape: {X.shape}")
        
        # Load metadata if available
        meta = None
        try:
            if 'meta_json' in data:
                meta_raw = data['meta_json']
                
                if meta_raw.ndim == 0 or len(meta_raw) == 0:
                    print("   ⚠️  Metadata unavailable")
                else:
                    meta = []
                    for m in meta_raw:
                        try:
                            if isinstance(m, (bytes, str)) and len(m) > 0:
                                meta.append(json.loads(m))
                            else:
                                meta.append({})
                        except:
                            meta.append({})
                    
                    if len(meta) > 0 and any(len(m) > 0 for m in meta):
                        print(f"   ✓ Loaded metadata for {len(meta)} events")
                    else:
                        meta = None
        except Exception as e:
            print(f"   ⚠️  Error loading metadata: {e}")
            meta = None
        
        # Subsample if requested
        if max_samples is not None and len(X) > max_samples:
            print(f"   Subsampling {max_samples} from {len(X)} total events")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
            if meta is not None:
                meta = [meta[i] for i in indices]
        
        print(f"   ✓ Loaded {len(X)} events")
        print(f"   - Binary: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
        print(f"   - Single: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        
        return X, y, meta
    
    def _get_predictions(self):
        """Get model predictions with batching and GPU"""
        predictions = []
        confidences = []
        
        batch_size = 128 if self.device.type == 'cuda' else 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.X_norm), batch_size), desc="   Predicting"):
                # Get batch
                batch_end = min(i + batch_size, len(self.X_norm))
                x_batch = torch.tensor(self.X_norm[i:batch_end], dtype=torch.float32)
                x_batch = x_batch.to(self.device)
                
                # Forward pass
                output = self.model(x_batch, return_all=False)
                logits = output['binary']
                probs = F.softmax(logits, dim=1)
                
                # Get predictions and confidence
                preds = probs.argmax(dim=1).cpu().numpy()
                confs = probs.max(dim=1).values.cpu().numpy()
                
                predictions.extend(preds)
                confidences.extend(confs)
        
        return np.array(predictions), np.array(confidences)
    
    def _print_summary(self):
        """Print summary statistics"""
        accuracy = (self.predictions == self.y).mean()
        
        tp = ((self.y == 1) & (self.predictions == 1)).sum()
        tn = ((self.y == 0) & (self.predictions == 0)).sum()
        fp = ((self.y == 0) & (self.predictions == 1)).sum()
        fn = ((self.y == 1) & (self.predictions == 0)).sum()
        
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Overall Accuracy:  {accuracy*100:.2f}%")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:   {tp:5d}  (Binary → Binary)")
        print(f"  True Negatives:   {tn:5d}  (Single → Single)")
        print(f"  False Positives:  {fp:5d}  (Single → Binary)")
        print(f"  False Negatives:  {fn:5d}  (Binary → Single)")
        
        # High confidence stats
        high_conf_90 = (self.confidences >= 0.9).sum()
        high_conf_95 = (self.confidences >= 0.95).sum()
        
        print(f"\nHigh Confidence Predictions:")
        print(f"  ≥90% confidence:  {high_conf_90:5d}  ({high_conf_90/len(self.X)*100:.1f}%)")
        print(f"  ≥95% confidence:  {high_conf_95:5d}  ({high_conf_95/len(self.X)*100:.1f}%)")
        
        # Confidence by outcome
        tp_mask = (self.y == 1) & (self.predictions == 1)
        tn_mask = (self.y == 0) & (self.predictions == 0)
        fp_mask = (self.y == 0) & (self.predictions == 1)
        fn_mask = (self.y == 1) & (self.predictions == 0)
        
        print(f"\nMean Confidence by Outcome:")
        print(f"  True Positives:   {self.confidences[tp_mask].mean()*100:.1f}%")
        print(f"  True Negatives:   {self.confidences[tn_mask].mean()*100:.1f}%")
        if fp_mask.sum() > 0:
            print(f"  False Positives:  {self.confidences[fp_mask].mean()*100:.1f}%  ← Lower!")
        if fn_mask.sum() > 0:
            print(f"  False Negatives:  {self.confidences[fn_mask].mean()*100:.1f}%  ← Lower!")
        print("="*60 + "\n")
    
    def plot_high_confidence_binary(self, min_confidence=0.9, n_examples=9):
        """Plot grid of high-confidence binary predictions"""
        binary_predicted = self.predictions == 1
        high_conf = self.confidences >= min_confidence
        correct = self.predictions == self.y
        
        correct_high = np.where(binary_predicted & high_conf & correct)[0]
        incorrect_high = np.where(binary_predicted & high_conf & ~correct)[0]
        
        print(f"High-confidence binary (≥{min_confidence*100:.0f}%):")
        print(f"  Correct:   {len(correct_high)}")
        print(f"  Incorrect: {len(incorrect_high)}")
        
        if len(correct_high) == 0:
            print(f"  ⚠️  No high-confidence correct predictions, trying lower threshold...")
            min_confidence = 0.85
            high_conf = self.confidences >= min_confidence
            correct_high = np.where(binary_predicted & high_conf & correct)[0]
            incorrect_high = np.where(binary_predicted & high_conf & ~correct)[0]
            print(f"  With ≥{min_confidence*100:.0f}%: {len(correct_high)} correct, {len(incorrect_high)} incorrect")
        
        if len(correct_high) == 0 and len(incorrect_high) == 0:
            print("  ⚠️  Still no predictions found, skipping this plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot correct (top 2 rows)
        n_correct = min(6, len(correct_high))
        for i in range(n_correct):
            idx = correct_high[i]
            ax = axes[i]
            ax.plot(self.X[idx], 'b-', linewidth=1.5, alpha=0.7)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            ax.set_title(f'✓ Binary (Conf: {self.confidences[idx]:.1%})',
                        color='green', fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Flux', fontsize=8)
        
        # Plot incorrect (bottom row)
        n_incorrect = min(3, len(incorrect_high))
        for i in range(n_incorrect):
            idx = incorrect_high[i]
            ax = axes[6+i]
            true_label = "Single" if self.y[idx] == 0 else "Binary"
            ax.plot(self.X[idx], 'r-', linewidth=1.5, alpha=0.7)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            ax.set_title(f'✗ Pred: Binary, True: {true_label}\n(Conf: {self.confidences[idx]:.1%})',
                        color='red', fontweight='bold', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Flux', fontsize=8)
        
        # Hide unused
        for i in range(n_correct + n_incorrect, 9):
            axes[i].axis('off')
        
        plt.suptitle(f'High-Confidence Binary Classifications (≥{min_confidence*100:.0f}%)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'high_confidence_binary_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    def plot_confidence_distributions(self):
        """Plot confidence distributions"""
        tp = (self.y == 1) & (self.predictions == 1)
        tn = (self.y == 0) & (self.predictions == 0)
        fp = (self.y == 0) & (self.predictions == 1)
        fn = (self.y == 1) & (self.predictions == 0)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # True Positives
        ax = axes[0, 0]
        ax.hist(self.confidences[tp], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(self.confidences[tp].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {self.confidences[tp].mean():.3f}')
        ax.set_title(f'True Positives (Binary → Binary)\n'
                    f'n={tp.sum()}, Mean Conf: {self.confidences[tp].mean():.1%}',
                    fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # True Negatives
        ax = axes[0, 1]
        ax.hist(self.confidences[tn], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(self.confidences[tn].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {self.confidences[tn].mean():.3f}')
        ax.set_title(f'True Negatives (Single → Single)\n'
                    f'n={tn.sum()}, Mean Conf: {self.confidences[tn].mean():.1%}',
                    fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # False Positives
        ax = axes[1, 0]
        if fp.sum() > 0:
            ax.hist(self.confidences[fp], bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(self.confidences[fp].mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {self.confidences[fp].mean():.3f}')
        ax.set_title(f'False Positives (Single → Binary)\n'
                    f'n={fp.sum()}, Mean Conf: {self.confidences[fp].mean():.1%}',
                    fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # False Negatives
        ax = axes[1, 1]
        if fn.sum() > 0:
            ax.hist(self.confidences[fn], bins=30, alpha=0.7, color='red', edgecolor='black')
            ax.axvline(self.confidences[fn].mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {self.confidences[fn].mean():.3f}')
        ax.set_title(f'False Negatives (Binary → Single)\n'
                    f'n={fn.sum()}, Mean Conf: {self.confidences[fn].mean():.1%}',
                    fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Confidence Score Distributions by Prediction Type',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'confidence_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    def plot_confidence_vs_correctness(self):
        """Plot calibration curve"""
        correct = self.predictions == self.y
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calibration curve
        ax = axes[0]
        conf_bins = np.linspace(0.5, 1.0, 11)
        accuracies, bin_centers, counts = [], [], []
        
        for i in range(len(conf_bins)-1):
            mask = (self.confidences >= conf_bins[i]) & (self.confidences < conf_bins[i+1])
            if mask.sum() > 0:
                accuracies.append(correct[mask].mean())
                bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                counts.append(mask.sum())
        
        if len(bin_centers) > 0:
            bars = ax.bar(bin_centers, accuracies, width=0.04, alpha=0.7, edgecolor='black', linewidth=1.5)
            for bar, cnt in zip(bars, counts):
                bar.set_facecolor(plt.cm.Blues(0.3 + 0.7 * cnt / max(counts)))
            
            # Add counts
            for bc, acc, cnt in zip(bin_centers, accuracies, counts):
                ax.text(bc, acc + 0.02, f'n={cnt}', ha='center', fontsize=7)
        
        ax.plot([0.5, 1.0], [0.5, 1.0], 'r--', linewidth=2, alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Model Calibration', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.05])
        
        # Scatter
        ax = axes[1]
        jitter = np.random.normal(0, 0.01, size=len(correct))
        binary_pred = self.predictions == 1
        
        # Subsample for visibility
        n_plot = min(5000, len(correct))
        idx = np.random.choice(len(correct), n_plot, replace=False)
        
        ax.scatter(self.confidences[idx][binary_pred[idx] & correct[idx]] + jitter[idx][binary_pred[idx] & correct[idx]],
                  correct[idx][binary_pred[idx] & correct[idx]],
                  alpha=0.3, s=5, color='green', label='Correct (Binary pred)')
        ax.scatter(self.confidences[idx][binary_pred[idx] & ~correct[idx]] + jitter[idx][binary_pred[idx] & ~correct[idx]],
                  correct[idx][binary_pred[idx] & ~correct[idx]],
                  alpha=0.3, s=5, color='red', label='Incorrect (Binary pred)')
        
        ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correctness')
        ax.set_title('Confidence vs Correctness', fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'confidence_vs_correctness.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    def generate_all(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        print("1. High-confidence binary examples...")
        self.plot_high_confidence_binary(min_confidence=0.9)
        print()
        
        print("2. Confidence distributions...")
        self.plot_confidence_distributions()
        print()
        
        print("3. Confidence vs correctness...")
        self.plot_confidence_vs_correctness()
        print()
        
        print("="*60)
        print(f"✅ ALL FIGURES SAVED TO: {self.output_dir}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize confident predictions')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to use (None = all)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = FixedVisualizer(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        use_cuda=not args.no_cuda
    )
    
    # Generate all plots
    viz.generate_all()
    
    print("\n🎉 Done! Check your figures directory.")
    print(f"📁 {args.output_dir}\n")


if __name__ == '__main__':
    main()