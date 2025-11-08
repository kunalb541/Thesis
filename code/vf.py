#!/usr/bin/env python3
"""
Fixed Visualization with Smart Interpolation for Missing Data
==============================================================
Removes artificial jumps from padding that confuse the model

Author: Kunal Bhatia (with padding fix)
Version: 2.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# Add your code path
sys.path.insert(0, '/pfs/data6/home/hd/hd_hd/hd_vm305/Thesis/code')
from transformer import SimpleStableTransformer

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300


class SmartMissingDataHandler:
    """Handle missing data without creating artificial features"""
    
    def __init__(self, pad_value=-1.0):
        self.pad_value = pad_value
        self.baseline_fraction = 0.2  # Use 20% of start/end for baseline
    
    def process_batch(self, X, verbose=True):
        """
        Process batch of light curves to remove padding artifacts
        
        Args:
            X: [N, T] array with pad_value for missing data
            verbose: Print processing statistics
            
        Returns:
            X_processed: [N, T] interpolated and centered
            validity_masks: [N, T] boolean masks (True = real data)
            stats: Dictionary with processing statistics
        """
        N, T = X.shape
        X_processed = np.zeros_like(X, dtype=np.float32)
        validity_masks = np.zeros_like(X, dtype=bool)
        
        stats = {
            'n_fully_valid': 0,
            'n_interpolated': 0,
            'n_failed': 0,
            'avg_validity': 0,
            'max_gaps': [],
            'baseline_stds': []
        }
        
        if verbose:
            iterator = tqdm(range(N), desc="Processing light curves")
        else:
            iterator = range(N)
        
        for i in iterator:
            processed, mask, info = self.process_single(X[i])
            X_processed[i] = processed
            validity_masks[i] = mask
            
            # Update statistics
            if info['status'] == 'fully_valid':
                stats['n_fully_valid'] += 1
            elif info['status'] == 'interpolated':
                stats['n_interpolated'] += 1
            else:
                stats['n_failed'] += 1
            
            if info['max_gap'] is not None:
                stats['max_gaps'].append(info['max_gap'])
            if info['baseline_std'] is not None:
                stats['baseline_stds'].append(info['baseline_std'])
        
        stats['avg_validity'] = validity_masks.mean()
        
        if verbose:
            print(f"\nProcessing Statistics:")
            print(f"  Fully valid: {stats['n_fully_valid']} ({100*stats['n_fully_valid']/N:.1f}%)")
            print(f"  Interpolated: {stats['n_interpolated']} ({100*stats['n_interpolated']/N:.1f}%)")
            print(f"  Failed (too sparse): {stats['n_failed']} ({100*stats['n_failed']/N:.1f}%)")
            print(f"  Average validity: {stats['avg_validity']*100:.1f}%")
            if stats['max_gaps']:
                print(f"  Max gap size: {np.mean(stats['max_gaps']):.1f} points (average)")
            if stats['baseline_stds']:
                print(f"  Baseline stability: {np.mean(stats['baseline_stds']):.4f} (std)")
        
        return X_processed, validity_masks, stats
    
    def process_single(self, light_curve):
        """
        Process single light curve
        
        Returns:
            processed: Smoothly interpolated curve
            mask: Validity mask
            info: Processing information
        """
        T = len(light_curve)
        timestamps = np.arange(T)
        
        # Identify valid data
        valid_mask = (light_curve != self.pad_value) & np.isfinite(light_curve)
        n_valid = valid_mask.sum()
        
        info = {
            'status': None,
            'max_gap': None,
            'baseline_std': None
        }
        
        # Case 1: Too few valid points
        if n_valid < 10:
            info['status'] = 'failed'
            # Return zeros for failed curves
            return np.zeros(T, dtype=np.float32), valid_mask, info
        
        # Case 2: Fully valid (no padding)
        if n_valid == T:
            info['status'] = 'fully_valid'
            # Just center on baseline
            baseline = self._estimate_baseline(light_curve, valid_mask)
            info['baseline_std'] = self._calculate_baseline_std(light_curve, valid_mask)
            return (light_curve - baseline).astype(np.float32), valid_mask, info
        
        # Case 3: Needs interpolation
        info['status'] = 'interpolated'
        
        # Get valid points
        valid_indices = np.where(valid_mask)[0]
        valid_data = light_curve[valid_mask]
        
        # Estimate baseline from quiet regions
        baseline = self._estimate_baseline(light_curve, valid_mask)
        info['baseline_std'] = self._calculate_baseline_std(light_curve, valid_mask)
        
        # Center the valid data
        centered_data = valid_data - baseline
        
        # Find gaps
        gaps = self._find_gaps(valid_mask)
        if gaps:
            info['max_gap'] = max(gap[1] - gap[0] for gap in gaps)
        
        # Create interpolation function
        if len(valid_indices) >= 2:
            # Use cubic spline for smooth interpolation if enough points
            # Otherwise linear
            kind = 'cubic' if len(valid_indices) >= 4 else 'linear'
            try:
                f_interp = interp1d(valid_indices, centered_data, 
                                  kind=kind, 
                                  fill_value='extrapolate',
                                  bounds_error=False)
            except:
                # Fallback to linear if cubic fails
                f_interp = interp1d(valid_indices, centered_data, 
                                  kind='linear', 
                                  fill_value='extrapolate',
                                  bounds_error=False)
        else:
            # Only one valid point - can't interpolate
            info['status'] = 'failed'
            return np.zeros(T, dtype=np.float32), valid_mask, info
        
        # Interpolate
        processed = f_interp(timestamps)
        
        # Handle extrapolation regions more carefully
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        
        # Before first valid point: use first valid value (flat extrapolation)
        processed[:first_valid] = centered_data[0]
        
        # After last valid point: use last valid value (flat extrapolation)
        processed[last_valid+1:] = centered_data[-1]
        
        # Smooth transitions at gap boundaries
        processed = self._smooth_gap_transitions(processed, valid_mask, gaps)
        
        return processed.astype(np.float32), valid_mask, info
    
    def _estimate_baseline(self, light_curve, valid_mask):
        """Estimate baseline from quiet regions of the light curve"""
        valid_data = light_curve[valid_mask]
        n_valid = len(valid_data)
        
        if n_valid == 0:
            return 0.0
        
        # Use first and last 20% of valid points
        n_baseline = max(int(n_valid * self.baseline_fraction), 10)
        
        baseline_points = np.concatenate([
            valid_data[:n_baseline],
            valid_data[-n_baseline:]
        ])
        
        # Use median for robustness
        return np.median(baseline_points)
    
    def _calculate_baseline_std(self, light_curve, valid_mask):
        """Calculate baseline stability (std of quiet regions)"""
        valid_data = light_curve[valid_mask]
        n_valid = len(valid_data)
        
        if n_valid < 20:
            return None
        
        n_baseline = max(int(n_valid * self.baseline_fraction), 10)
        baseline_points = np.concatenate([
            valid_data[:n_baseline],
            valid_data[-n_baseline:]
        ])
        
        return np.std(baseline_points)
    
    def _find_gaps(self, valid_mask):
        """Find continuous gaps in the data"""
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, is_valid in enumerate(valid_mask):
            if not is_valid and not in_gap:
                # Start of gap
                in_gap = True
                gap_start = i
            elif is_valid and in_gap:
                # End of gap
                in_gap = False
                gaps.append((gap_start, i))
        
        # Handle gap at the end
        if in_gap:
            gaps.append((gap_start, len(valid_mask)))
        
        return gaps
    
    def _smooth_gap_transitions(self, processed, valid_mask, gaps, smooth_width=3):
        """Apply smoothing at gap boundaries to avoid sharp transitions"""
        smoothed = processed.copy()
        
        for gap_start, gap_end in gaps:
            # Smooth transition at gap start
            if gap_start > smooth_width:
                transition_start = gap_start - smooth_width
                for i in range(smooth_width):
                    alpha = (i + 1) / (smooth_width + 1)
                    smoothed[gap_start - i - 1] = (
                        alpha * processed[gap_start - i - 1] +
                        (1 - alpha) * processed[transition_start]
                    )
            
            # Smooth transition at gap end
            if gap_end < len(processed) - smooth_width:
                transition_end = gap_end + smooth_width
                for i in range(smooth_width):
                    alpha = (i + 1) / (smooth_width + 1)
                    smoothed[gap_end + i] = (
                        alpha * processed[gap_end + i] +
                        (1 - alpha) * processed[transition_end - 1]
                    )
        
        return smoothed


class FixedVisualizerWithInterpolation:
    """Visualizer with smart missing data handling"""
    
    def __init__(self, model_path, data_path, output_dir='./figures', 
                 max_samples=None, use_cuda=True, pad_value=-1.0):
        """Initialize with data preprocessing"""
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pad_value = pad_value
        
        # Setup device
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize data handler
        self.data_handler = SmartMissingDataHandler(pad_value=pad_value)
        
        # Load model
        print("\n📦 Loading model...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Load and preprocess data
        print("\n📊 Loading and preprocessing data...")
        self.X_raw, self.y, self.meta = self._load_data(data_path, max_samples)
        
        # Apply smart interpolation
        print("\n🔧 Applying smart interpolation to remove padding artifacts...")
        self.X, self.validity_masks, self.process_stats = self.data_handler.process_batch(
            self.X_raw, verbose=True
        )
        
        # Compare before/after
        self._compare_preprocessing()
        
        # Get predictions on cleaned data
        print(f"\n🔮 Getting predictions for {len(self.X)} samples...")
        self.predictions, self.confidences = self._get_predictions()
        
        # Print summary
        self._print_summary()
    
    def _load_model(self, model_path):
        """Load model with auto-detected architecture"""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Auto-detect architecture
        d_model = state_dict['input_proj.0.weight'].shape[0]
        num_layers = len([k for k in state_dict.keys() if 'blocks.' in k and '.norm1.weight' in k])
        
        possible_nheads = [2, 4, 8, 16]
        nhead = 4
        for nh in possible_nheads:
            if d_model % nh == 0:
                nhead = nh
                break
        
        print(f"   Detected architecture: d_model={d_model}, layers={num_layers}, nhead={nhead}")
        
        model = SimpleStableTransformer(
            n_points=1500,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=d_model * 4,
            dropout=0.2
        )
        
        model.load_state_dict(state_dict)
        return model
    
    def _load_data(self, data_path, max_samples):
        """Load data"""
        data = np.load(data_path)
        
        X = data['X']
        y = data['y']
        
        # Handle shape
        if X.ndim == 3:
            X = X.squeeze(1)
        
        # Subsample if requested
        if max_samples is not None and len(X) > max_samples:
            print(f"   Subsampling {max_samples} from {len(X)} total events")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        print(f"   ✓ Loaded {len(X)} events")
        print(f"   - Binary: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
        print(f"   - Single: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        
        return X, y, None
    
    def _compare_preprocessing(self):
        """Show the effect of preprocessing"""
        print("\n📈 Preprocessing Effect:")
        
        # Find an example with gaps
        gap_example = None
        for i in range(len(self.X_raw)):
            if self.validity_masks[i].sum() < len(self.validity_masks[i]) * 0.9:
                gap_example = i
                break
        
        if gap_example is not None:
            # Calculate jumps before and after
            raw = self.X_raw[gap_example]
            processed = self.X[gap_example]
            
            # Jumps in raw data
            raw_valid = raw != self.pad_value
            if raw_valid.sum() > 1:
                raw_jumps = np.abs(np.diff(raw))
                max_raw_jump = raw_jumps.max()
            else:
                max_raw_jump = 0
            
            # Jumps in processed data
            processed_jumps = np.abs(np.diff(processed))
            max_processed_jump = processed_jumps.max()
            
            print(f"   Example light curve {gap_example}:")
            print(f"   - Validity: {self.validity_masks[gap_example].sum()}/{len(raw)} points")
            print(f"   - Max jump before: {max_raw_jump:.2f}")
            print(f"   - Max jump after: {max_processed_jump:.2f}")
            print(f"   - Jump reduction: {(1 - max_processed_jump/max_raw_jump)*100:.1f}%")
            
            # Save comparison plot
            self._plot_preprocessing_comparison(gap_example)
    
    def _plot_preprocessing_comparison(self, idx):
        """Plot before/after preprocessing"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        timestamps = np.arange(len(self.X_raw[idx]))
        
        # Before
        ax = axes[0]
        raw = self.X_raw[idx]
        valid = self.validity_masks[idx]
        
        # Plot raw data
        ax.plot(timestamps[valid], raw[valid], 'b.', label='Valid data', markersize=3)
        if (~valid).any():
            ax.plot(timestamps[~valid], raw[~valid], 'r.', label='Padding', markersize=2, alpha=0.5)
        
        ax.set_ylabel('Raw Flux', fontsize=11)
        ax.set_title('Before: Raw Data with Padding Artifacts', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight jumps
        jumps = np.abs(np.diff(raw))
        large_jumps = np.where(jumps > 2.0)[0]
        for jump_idx in large_jumps[:5]:  # Show first 5 large jumps
            ax.axvspan(jump_idx, jump_idx+1, color='red', alpha=0.2)
        
        # After
        ax = axes[1]
        processed = self.X[idx]
        
        ax.plot(timestamps, processed, 'g-', label='Interpolated', linewidth=1.5, alpha=0.7)
        ax.plot(timestamps[valid], processed[valid], 'b.', label='Original points', markersize=4)
        
        ax.set_xlabel('Time Point', fontsize=11)
        ax.set_ylabel('Processed Flux', fontsize=11)
        ax.set_title('After: Smooth Interpolation (No Artifacts)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Preprocessing Effect on Light Curve {idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f'preprocessing_comparison_{idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved comparison plot: {output_path}")
        plt.close()
    
    def _get_predictions(self):
        """Get model predictions"""
        predictions = []
        confidences = []
        
        batch_size = 128 if self.device.type == 'cuda' else 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.X), batch_size), desc="   Predicting"):
                batch_end = min(i + batch_size, len(self.X))
                x_batch = torch.tensor(self.X[i:batch_end], dtype=torch.float32).to(self.device)
                
                output = self.model(x_batch, return_all_timesteps=False, pad_value=0.0)  # Note: processed data doesn't use -1 padding
                logits = output['binary']
                probs = torch.softmax(logits, dim=1)
                
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
        print("PREDICTION SUMMARY (After Preprocessing)")
        print("="*60)
        print(f"Overall Accuracy:  {accuracy*100:.2f}%")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:   {tp:5d}  (Binary → Binary)")
        print(f"  True Negatives:   {tn:5d}  (Single → Single)")
        print(f"  False Positives:  {fp:5d}  (Single → Binary)")
        print(f"  False Negatives:  {fn:5d}  (Binary → Single)")
        
        # Improvement check
        if tp > 0:
            print(f"\n🎉 SUCCESS! Model is now detecting binary events!")
            print(f"   Binary recall: {tp/(tp+fn)*100:.1f}%")
            print(f"   Binary precision: {tp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else "N/A")
        else:
            print(f"\n⚠️ Still not detecting binaries, but preprocessing is correct.")
            print(f"   Model may need retraining on preprocessed data.")
        
        print("="*60)
    
    def plot_fixed_predictions(self, n_examples=9):
        """Plot examples with the fixed preprocessing"""
        
        # Get some correctly and incorrectly classified examples
        tp_idx = np.where((self.y == 1) & (self.predictions == 1))[0][:3]
        tn_idx = np.where((self.y == 0) & (self.predictions == 0))[0][:3]
        fp_idx = np.where((self.y == 0) & (self.predictions == 1))[0][:3]
        fn_idx = np.where((self.y == 1) & (self.predictions == 0))[0][:3]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        all_indices = []
        all_labels = []
        
        # Add examples
        for indices, label, color in [
            (tp_idx, 'TP: Binary→Binary', 'green'),
            (tn_idx, 'TN: Single→Single', 'blue'),
            (fp_idx, 'FP: Single→Binary', 'orange'),
            (fn_idx, 'FN: Binary→Single', 'red')
        ]:
            for idx in indices[:2]:  # Max 2 per category
                if len(all_indices) < 9:
                    all_indices.append(idx)
                    all_labels.append((label, color))
        
        for i, (idx, (label, color)) in enumerate(zip(all_indices, all_labels)):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Plot processed light curve
            timestamps = np.arange(len(self.X[idx]))
            ax.plot(timestamps, self.X[idx], color=color, linewidth=1.5, alpha=0.7)
            
            # Mark valid points
            valid = self.validity_masks[idx]
            ax.scatter(timestamps[valid], self.X[idx][valid], 
                      color=color, s=5, alpha=0.5)
            
            # Find peaks in the processed data
            peaks, _ = find_peaks(self.X[idx], height=1.0, distance=50)
            if len(peaks) > 0:
                ax.scatter(peaks, self.X[idx][peaks], color='red', s=50, 
                          marker='v', zorder=5)
            
            conf = self.confidences[idx]
            ax.set_title(f'{label}\nConf: {conf:.1%}, Peaks: {len(peaks)}', 
                        fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Flux', fontsize=8)
        
        # Hide unused
        for i in range(len(all_indices), 9):
            axes[i].axis('off')
        
        plt.suptitle('Predictions After Removing Padding Artifacts', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'fixed_predictions_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved fixed predictions: {output_path}")
        plt.close()
    
    def analyze_validity_impact(self):
        """Analyze how data completeness affects predictions"""
        
        # Bin by validity percentage
        validity_pct = self.validity_masks.mean(axis=1) * 100
        
        bins = [0, 60, 70, 80, 90, 100]
        bin_labels = ['<60%', '60-70%', '70-80%', '80-90%', '90-100%']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy by validity
        ax = axes[0]
        accuracies = []
        counts = []
        
        for i in range(len(bins)-1):
            mask = (validity_pct >= bins[i]) & (validity_pct < bins[i+1])
            if mask.sum() > 0:
                acc = (self.predictions[mask] == self.y[mask]).mean()
                accuracies.append(acc * 100)
                counts.append(mask.sum())
            else:
                accuracies.append(0)
                counts.append(0)
        
        bars = ax.bar(bin_labels, accuracies, alpha=0.7, edgecolor='black')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'n={count}', ha='center', fontsize=9)
        
        ax.set_xlabel('Data Completeness', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('Accuracy vs. Data Completeness', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Binary detection by validity
        ax = axes[1]
        binary_recalls = []
        
        for i in range(len(bins)-1):
            mask = (validity_pct >= bins[i]) & (validity_pct < bins[i+1]) & (self.y == 1)
            if mask.sum() > 0:
                recall = (self.predictions[mask] == 1).mean()
                binary_recalls.append(recall * 100)
            else:
                binary_recalls.append(0)
        
        bars = ax.bar(bin_labels, binary_recalls, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Data Completeness', fontsize=11)
        ax.set_ylabel('Binary Recall (%)', fontsize=11)
        ax.set_title('Binary Detection vs. Data Completeness', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'validity_impact_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved validity analysis: {output_path}")
        plt.close()
    
    def generate_all(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS WITH FIXED DATA")
        print("="*60)
        
        print("\n1. Fixed predictions grid...")
        self.plot_fixed_predictions()
        
        print("\n2. Validity impact analysis...")
        self.analyze_validity_impact()
        
        print("\n" + "="*60)
        print(f"✅ ALL FIGURES SAVED TO: {self.output_dir}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed visualization with smart interpolation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data (.npz)')
    parser.add_argument('--output_dir', type=str, default='./figures_fixed',
                       help='Output directory for figures')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to use')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--pad_value', type=float, default=-1.0,
                       help='Padding value in the data')
    
    args = parser.parse_args()
    
    # Create visualizer with preprocessing
    viz = FixedVisualizerWithInterpolation(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        use_cuda=not args.no_cuda,
        pad_value=args.pad_value
    )
    
    # Generate all plots
    viz.generate_all()
    
    print("\n🎉 Done! Check your figures directory.")
    print(f"📁 {args.output_dir}")
    print("\nIf predictions are still poor, the model needs retraining on preprocessed data.")
    print("Use the SmartMissingDataHandler in your training pipeline!\n")


if __name__ == '__main__':
    main()
