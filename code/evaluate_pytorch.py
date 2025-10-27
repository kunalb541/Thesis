"""
PyTorch evaluation script
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json

from train import TimeDistributedCNN, LightCurveDataset
from torch.utils.data import DataLoader

def evaluate_model(model, loader, device):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)  # (batch, time, classes)
            
            # Use last timestep
            probs = torch.softmax(outputs[:, -1, :], dim=1)
            _, predicted = torch.max(probs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Binary prob
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['PSPL', 'Binary'],
                yticklabels=['PSPL', 'Binary'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")

def plot_roc_curve(y_true, y_probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved: {save_path}")
    
    return roc_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model (.pt)')
    parser.add_argument('--data', required=True, help='Path to data (.npz)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Use test split (last 15%)
    n_test = int(0.15 * len(X))
    X_test = X[-n_test:]
    y_test = y_encoded[-n_test:]
    
    # Standardize
    scaler = StandardScaler()
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.fit_transform(X_test_2d).reshape(X_test.shape)
    
    # Create dataset
    test_dataset = LightCurveDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    model = TimeDistributedCNN(
        sequence_length=X_test.shape[1],
        num_channels=X_test.shape[2]
    )
    
    # Handle DataParallel
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        model = nn.DataParallel(model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    print("Evaluating...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    
    # Metrics
    accuracy = (y_true == y_pred).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=['PSPL', 'Binary'],
                                   digits=4)
    print("\n" + report)
    
    # Save report
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    roc_auc = plot_roc_curve(y_true, y_probs, roc_path)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'test_samples': int(len(y_test))
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
