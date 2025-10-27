"""
PyTorch evaluation script - IMPROVED VERSION
Adds: early detection analysis, better error handling, batched inference
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
import joblib
import sys
from pathlib import Path
from tqdm import tqdm

class TimeDistributedCNN(nn.Module):
    def __init__(self, sequence_length=1500, num_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['PSPL', 'Binary'],
                yticklabels=['PSPL', 'Binary'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")

def plot_roc(y_true, y_probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {save_path}")
    return roc_auc

def plot_precision_recall(y_true, y_probs, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curve saved to {save_path}")
    return pr_auc

def evaluate_early_detection(model, loader, device, checkpoints=[0.1, 0.25, 0.33, 0.5, 0.67, 0.83, 1.0]):
    """
    Evaluate classification accuracy at different fractions of observation.
    This tests the model's ability to classify events before they complete.
    """
    print("\n" + "="*60)
    print("EARLY DETECTION ANALYSIS")
    print("="*60)
    
    model.eval()
    all_outputs = []
    all_labels = []
    
    # Get all predictions
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc='Computing predictions'):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)  # (batch, time, classes)
            all_outputs.append(outputs.cpu())
            all_labels.append(batch_y)
    
    # Concatenate
    all_outputs = torch.cat(all_outputs, dim=0)  # (N, time, classes)
    all_labels = torch.cat(all_labels, dim=0)    # (N,)
    
    n_timesteps = all_outputs.shape[1]
    results = {}
    
    print(f"\nEvaluating at {len(checkpoints)} checkpoints:")
    for frac in checkpoints:
        timestep = int(n_timesteps * frac) - 1
        timestep = max(0, min(timestep, n_timesteps - 1))
        
        # Get predictions at this timestep
        probs = torch.softmax(all_outputs[:, timestep, :], dim=1)
        _, preds = torch.max(probs, 1)
        
        # Calculate accuracy
        acc = (preds == all_labels).float().mean().item()
        results[frac] = acc
        
        print(f"  {frac*100:5.1f}% observed (t={timestep:4d}): Accuracy = {acc:.4f}")
    
    return results

def plot_early_detection(results, save_path):
    """Plot early detection curve"""
    fractions = sorted(results.keys())
    accuracies = [results[f] for f in fractions]
    
    plt.figure(figsize=(10, 6))
    plt.plot([f*100 for f in fractions], accuracies, 'o-', 
             linewidth=2, markersize=10, color='#2E86AB')
    
    # Add horizontal lines for reference
    plt.axhline(y=0.90, color='green', linestyle='--', linewidth=1.5, 
                label='90% threshold', alpha=0.7)
    plt.axhline(y=0.95, color='orange', linestyle='--', linewidth=1.5,
                label='95% threshold', alpha=0.7)
    
    plt.xlabel('Fraction of Event Observed (%)', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Real-Time Classification Performance', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim([0.75, 1.0])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Early detection curve saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate microlensing classifier')
    parser.add_argument('--model', required=True, help='Path to trained model (.pt)')
    parser.add_argument('--data', required=True, help='Path to test data (.npz)')
    parser.add_argument('--scaler', default=None, help='Path to scaler (.pkl)')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--early_detection', action='store_true', help='Perform early detection analysis')
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MICROLENSING BINARY CLASSIFICATION - EVALUATION")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    try:
        data = np.load(args.data)
        X, y = data['X'], data['y']
    except FileNotFoundError:
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)
    
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {np.unique(y)}")
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Use test split (last 15%)
    n_test = int(0.15 * len(X))
    X_test = X[-n_test:]
    y_test = y_encoded[-n_test:]
    
    print(f"  Test samples: {len(X_test)}")
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"  Test distribution: {dict(zip(unique, counts))}")
    
    # Load or create scaler
    if args.scaler and Path(args.scaler).exists():
        print(f"\nLoading scaler from {args.scaler}")
        scaler = joblib.load(args.scaler)
    else:
        # Look for scaler in model directory
        model_dir = Path(args.model).parent
        scaler_candidates = list(model_dir.glob('scaler.pkl'))
        
        if scaler_candidates:
            scaler_path = scaler_candidates[0]
            print(f"\nFound scaler: {scaler_path}")
            scaler = joblib.load(scaler_path)
        else:
            print("\nWARNING: No scaler found. Fitting on train data...")
            # Use first 70% as train for scaler fitting
            n_train = int(0.7 * len(X))
            X_train_for_scaler = X[:n_train]
            scaler = StandardScaler()
            scaler.fit(X_train_for_scaler.reshape(-1, X_train_for_scaler.shape[-1]))
    
    # Apply scaler
    print("Standardizing test data...")
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    try:
        checkpoint = torch.load(args.model, map_location=device)
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    model = TimeDistributedCNN(X_test.shape[1], X_test.shape[2])
    
    # Handle DataParallel
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        model = nn.DataParallel(model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_acc' in checkpoint:
        print(f"  Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Create dataset and loader
    test_dataset = LightCurveDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Predict (batched inference to avoid OOM)
    print("\nRunning inference...")
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader, desc='Inference'):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs[:, -1, :], dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    accuracy = (y_test == y_pred).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, 
                                   target_names=['PSPL', 'Binary'], 
                                   digits=4)
    print(report)
    
    # Save report
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"✓ Report saved to {report_path}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_test, y_pred, cm_path)
    
    # ROC curve
    print("\nGenerating ROC curve...")
    roc_path = output_dir / 'roc_curve.png'
    roc_auc = plot_roc(y_test, y_probs, roc_path)
    
    # Precision-Recall curve
    print("\nGenerating Precision-Recall curve...")
    pr_path = output_dir / 'precision_recall_curve.png'
    pr_auc = plot_precision_recall(y_test, y_probs, pr_path)
    
    # Early detection analysis
    early_detection_results = None
    if args.early_detection:
        early_detection_results = evaluate_early_detection(
            model, test_loader, device,
            checkpoints=[0.1, 0.25, 0.33, 0.5, 0.67, 0.83, 1.0]
        )
        
        # Plot early detection
        print("\nGenerating early detection curve...")
        ed_path = output_dir / 'early_detection.png'
        plot_early_detection(early_detection_results, ed_path)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
    }
    
    if early_detection_results:
        metrics['early_detection'] = {
            f'{int(k*100)}pct': float(v) 
            for k, v in early_detection_results.items()
        }
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nMetrics Summary:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")
    
    if early_detection_results:
        print("\nEarly Detection Summary:")
        for frac, acc in sorted(early_detection_results.items()):
            if frac in [0.33, 0.5, 0.67, 1.0]:
                print(f"  {frac*100:5.1f}% observed: {acc:.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()