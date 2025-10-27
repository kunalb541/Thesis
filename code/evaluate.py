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
import joblib

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

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['PSPL', 'Binary'],
                yticklabels=['PSPL', 'Binary'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return roc_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    data = np.load(args.data)
    X, y = data['X'], data['y']
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Test split (last 15%)
    n_test = int(0.15 * len(X))
    X_test = X[-n_test:]
    y_test = y_encoded[-n_test:]
    
    # Load scaler
    scaler_path = args.model.replace('.pt', '_scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X_test.reshape(-1, X_test.shape[-1]))
    
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    model = TimeDistributedCNN(X_test.shape[1], X_test.shape[2])
    
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        model = nn.DataParallel(model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Predict
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs[:, -1, :], dim=1)
        _, preds = torch.max(probs, 1)
    
    y_pred = preds.cpu().numpy()
    y_probs = probs[:, 1].cpu().numpy()
    
    # Metrics
    accuracy = (y_test == y_pred).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=['PSPL', 'Binary'], digits=4)
    print(report)
    
    # Save
    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write(report)
    
    plot_confusion_matrix(y_test, y_pred, os.path.join(args.output_dir, 'confusion_matrix.png'))
    roc_auc = plot_roc(y_test, y_probs, os.path.join(args.output_dir, 'roc_curve.png'))
    
    metrics = {'accuracy': float(accuracy), 'roc_auc': float(roc_auc)}
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
