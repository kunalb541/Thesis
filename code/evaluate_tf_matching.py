#!/usr/bin/env python3
"""Evaluation for TensorFlow-matching architecture"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import argparse
import sys

# Import from training script
sys.path.insert(0, '.')
from train_tf_matching import TensorFlowStyleCNN, load_and_prepare_data, TimeSeriesDataset

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    
    for xb, yb in loader:
        xb = xb.to(device)
        outputs = model(xb).mean(dim=1)  # Aggregate over time
        probs_batch = torch.softmax(outputs, dim=1)
        
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        labels.extend(yb.numpy())
        probs.extend(probs_batch.cpu().numpy())
    
    return np.array(preds), np.array(labels), np.array(probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    (_, _), (_, _), (X_test, y_test), seq_len, features = load_and_prepare_data(args.data)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=128, num_workers=4)
    
    # Load model
    model = TensorFlowStyleCNN(seq_len, features).to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    
    # Evaluate
    preds, labels, probs = evaluate(model, test_loader, device)
    
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    roc_auc = roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0.5
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    if acc > 0.7: print("\n🎉 SUCCESS!")
    elif acc > 0.6: print("\n✅ Good!")
    else: print("\n⚠️  Check data quality")

if __name__ == '__main__':
    main()
