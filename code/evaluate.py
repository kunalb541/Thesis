#!/usr/bin/env python3
"""
Evaluation Script for Masked Transformer
Properly evaluates model with missing data handling

Author: Kunal Bhatia
Version: 3.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from transformer import MaskedMicrolensingTransformer


class MaskedModelEvaluator:
    """Evaluator for masked transformer models"""
    
    def __init__(self, model_path, normalizer_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Auto-detect architecture
        d_model = state_dict['input_proj.0.weight'].shape[0]
        num_layers = len([k for k in state_dict.keys() 
                         if 'blocks.' in k and '.norm1.weight' in k])
        
        # Create model
        self.model = MaskedMicrolensingTransformer(
            n_points=1500,
            d_model=d_model,
            nhead=4,
            num_layers=num_layers,
            dim_ff=d_model * 4,
            dropout=0.2
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    @torch.no_grad()
    def predict(self, X, pad_value=-1.0, batch_size=64):
        """Get predictions with automatic validity detection"""
        self.model.eval()
        
        n_samples = len(X)
        all_probs = []
        all_preds = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
            batch = X[i:i+batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
            
            # Auto-detect validity masks
            validity_masks = (X_tensor != pad_value)
            
            # Forward pass
            outputs = self.model(X_tensor, validity_masks, return_all_timesteps=False)
            logits = outputs['binary']
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=1).cpu().numpy())
        
        return {
            'probs': np.vstack(all_probs),
            'preds': np.concatenate(all_preds)
        }
    
    def evaluate_full(self, X, y, save_dir=None):
        """Complete evaluation with visualizations"""
        
        # Get predictions
        results = self.predict(X)
        probs = results['probs']
        preds = results['preds']
        
        # Calculate metrics
        accuracy = accuracy_score(y, preds)
        roc_auc = roc_auc_score(y, probs[:, 1])
        cm = confusion_matrix(y, preds)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              PSPL  Binary")
        print(f"Actual PSPL   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Binary {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Analyze by confidence
        confidences = probs.max(axis=1)
        high_conf_mask = confidences >= 0.9
        
        print(f"\nHigh Confidence (≥90%):")
        print(f"  Count: {high_conf_mask.sum()} ({high_conf_mask.mean()*100:.1f}%)")
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y[high_conf_mask], preds[high_conf_mask])
            print(f"  Accuracy: {high_conf_acc:.4f} ({high_conf_acc*100:.2f}%)")
        
        # Analyze errors
        errors = preds != y
        error_confs = confidences[errors]
        
        print(f"\nError Analysis:")
        print(f"  Total errors: {errors.sum()}")
        if errors.sum() > 0:
            print(f"  Mean confidence on errors: {error_confs.mean():.3f}")
            print(f"  High-confidence errors: {(error_confs >= 0.9).sum()}")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics = {
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist(),
                'n_samples': len(y),
                'n_high_conf': int(high_conf_mask.sum())
            }
            
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(cm, save_dir / 'confusion_matrix.png')
            
            # Plot confidence distribution
            self._plot_confidence_dist(confidences, errors, save_dir / 'confidence_dist.png')
            
            print(f"\n✓ Results saved to {save_dir}")
        
        return metrics
    
    def _plot_confusion_matrix(self, cm, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        classes = ['PSPL', 'Binary']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_dist(self, confidences, errors, save_path):
        """Plot confidence distributions"""
        plt.figure(figsize=(10, 6))
        
        # Correct predictions
        plt.hist(confidences[~errors], bins=50, alpha=0.7, 
                label='Correct', color='green', edgecolor='black')
        
        # Incorrect predictions
        plt.hist(confidences[errors], bins=50, alpha=0.7,
                label='Incorrect', color='red', edgecolor='black')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Masked Transformer')
    parser.add_argument('--model_path', required=True, help='Path to model')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--output_dir', default='evaluation', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    if X.ndim == 3:
        X = X.squeeze(1)
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: Binary={np.sum(y==1)}, PSPL={np.sum(y==0)}")
    
    # Check validity
    pad_value = -1.0
    validity_ratio = (X != pad_value).mean()
    print(f"Data validity: {validity_ratio*100:.1f}%")
    
    # Create evaluator
    evaluator = MaskedModelEvaluator(args.model_path)
    
    # Evaluate
    metrics = evaluator.evaluate_full(X, y, save_dir=args.output_dir)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()