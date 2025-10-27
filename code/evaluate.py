"""
Evaluation script for trained models
Generates comprehensive metrics, confusion matrices, and ROC curves
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import argparse
import os
import json

def load_and_preprocess_test_data(data_path):
    """Load and preprocess test data"""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Take only test split (last 15%)
    n_test = int(0.15 * len(X))
    X_test = X[-n_test:]
    y_test = y_encoded[-n_test:]
    
    # Standardize (should use saved scaler in production)
    scaler = StandardScaler()
    n_test, n_time, n_feat = X_test.shape
    X_test_2d = X_test.reshape(-1, n_feat)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, n_time, n_feat)
    
    return X_test_scaled, y_test

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['PSPL', 'Binary'], 
                yticklabels=['PSPL', 'Binary'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_proba, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_proba, save_path):
    """Plot and save Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")
    
    return pr_auc

def evaluate_early_detection(model, X_test, y_test, output_dir):
    """
    Evaluate how early the model can detect binary events
    Critical for real-time classification with TimeDistributed
    """
    print("\nEvaluating early detection capability...")
    
    # Make predictions at different time steps
    time_steps = [100, 250, 500, 750, 1000, 1250, 1500]
    accuracies = []
    
    for t in time_steps:
        X_truncated = X_test[:, :t, :]
        
        # Pad to full length
        X_padded = np.zeros((X_test.shape[0], 1500, X_test.shape[2]))
        X_padded[:, :t, :] = X_truncated
        
        # Predict using last time step
        y_pred_proba = model.predict(X_padded, verbose=0)
        y_pred_proba_last = y_pred_proba[:, -1, 1]  # Probability of Binary class at last time step
        y_pred = (y_pred_proba_last > 0.5).astype(int)
        
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)
        print(f"Time step {t:4d}: Accuracy = {acc:.4f}")
    
    # Plot early detection curve
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Observations')
    plt.ylabel('Classification Accuracy')
    plt.title('Early Detection Performance')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'early_detection.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Early detection plot saved to {save_path}")
    
    return dict(zip(time_steps, accuracies))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to test data')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    model.summary()
    
    # Load test data
    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    X_test, y_test = load_and_preprocess_test_data(args.data)
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Make predictions
    print("=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    y_pred_proba = model.predict(X_test, verbose=1)
    
    # For TimeDistributed output, use last time step predictions
    y_pred_proba_last = y_pred_proba[:, -1, 1]  # Probability of Binary class at last time step
    y_pred = (y_pred_proba_last > 0.5).astype(int)
    
    # Classification report
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    report = classification_report(y_test, y_pred, target_names=['PSPL', 'Binary'], digits=4)
    print(report)
    
    # Save classification report
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    roc_auc = plot_roc_curve(y_test, y_pred_proba_last, roc_path)
    
    # Plot Precision-Recall curve
    pr_path = os.path.join(args.output_dir, 'precision_recall_curve.png')
    pr_auc = plot_precision_recall_curve(y_test, y_pred_proba_last, pr_path)
    
    # Early detection evaluation
    early_detection_results = evaluate_early_detection(model, X_test, y_test, args.output_dir)
    
    # Save all metrics
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'early_detection': early_detection_results,
        'classification_report': report
    }
    
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
