"""
PyTorch training script for microlensing classifier
Works with AMD ROCm GPUs
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
import json
import joblib
from datetime import datetime
from tqdm import tqdm

class TimeDistributedCNN(nn.Module):
    """1D CNN with per-timestep classification"""
    def __init__(self, sequence_length=1500, num_channels=1, num_classes=2):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(num_channels, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # Per-timestep classification
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, time, channels)
        x = x.transpose(1, 2)  # (batch, channels, time)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # Apply classification at each timestep
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, time, classes)
        
        return x

class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in tqdm(loader, desc='Training'):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)  # (batch, time, classes)
        
        # Use last timestep for loss
        loss = criterion(outputs[:, -1, :], batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs[:, -1, :], 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs[:, -1, :], batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs[:, -1, :], 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--experiment_name', type=str, default='baseline')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'../results/{args.experiment_name}_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading {args.data}...")
    data = np.load(args.data)
    X, y = data['X'], data['y']
    print(f"Shape: {X.shape}")
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Datasets
    train_dataset = LightCurveDataset(X_train, y_train)
    val_dataset = LightCurveDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = TimeDistributedCNN(X_train.shape[1], X_train.shape[2])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training
    print("\nTraining...")
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(results_dir, 'best_model.pt'))
    
    # Save artifacts
    joblib.dump(scaler, os.path.join(results_dir, 'scaler.pkl'))
    
    with open(os.path.join(results_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)
    
    print(f"\nDone! Results in {results_dir}")

if __name__ == "__main__":
    main()
