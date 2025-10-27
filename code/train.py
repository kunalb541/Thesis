"""
PyTorch training script for microlensing CNN classifier
GPU-optimized for AMD MI300A with TimeDistributed architecture preserved
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class TimeDistributedCNN(nn.Module):
    """
    1D CNN with TimeDistributed-like behavior
    Processes sequences and outputs predictions at each timestep
    """
    def __init__(self, sequence_length=1500, num_channels=1, num_classes=2):
        super(TimeDistributedCNN, self).__init__()
        
        # CNN feature extractor
        self.features = nn.Sequential(
            # First conv block
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second conv block
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third conv block
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # TimeDistributed classification (per-timestep)
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        features = self.features(x)  # (batch, 32, time)
        
        # Apply classifier at each timestep (TimeDistributed)
        features = features.permute(0, 2, 1)  # (batch, time, 32)
        output = self.classifier(features)  # (batch, time, num_classes)
        
        return output

class LightCurveDataset(Dataset):
    """PyTorch dataset for light curves"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def setup_gpu():
    """Configure GPU settings"""
    if not torch.cuda.is_available():
        print("WARNING: No GPUs detected! Training will be slow on CPU.")
        return 0, 'cpu'
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = 'cuda'
    return num_gpus, device

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)  # (batch, time, classes)
        
        # Expand labels for all timesteps
        batch_y_expanded = batch_y.unsqueeze(1).expand(-1, outputs.size(1))
        
        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch_y_expanded.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics (use last timestep for accuracy)
        total_loss += loss.item()
        _, predicted = torch.max(outputs[:, -1, :], 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            batch_y_expanded = batch_y.unsqueeze(1).expand(-1, outputs.size(1))
            
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_y_expanded.view(-1))
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs[:, -1, :], 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to .npz data file')
    parser.add_argument('--output', required=True, help='Path to save model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--experiment_name', type=str, default='baseline')
    args = parser.parse_args()
    
    print("=" * 80)
    print("GPU SETUP")
    print("=" * 80)
    num_gpus, device = setup_gpu()
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Split data
    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Standardize
    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    scaler = StandardScaler()
    n_train, n_time, n_feat = X_train.shape
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, n_time, n_feat)
    
    n_val = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, n_feat)
    X_val_scaled = scaler.transform(X_val_2d).reshape(n_val, n_time, n_feat)
    
    n_test = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, n_feat)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, n_time, n_feat)
    
    # Create datasets
    train_dataset = LightCurveDataset(X_train_scaled, y_train)
    val_dataset = LightCurveDataset(X_val_scaled, y_val)
    test_dataset = LightCurveDataset(X_test_scaled, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Build model
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    model = TimeDistributedCNN(
        sequence_length=X_train.shape[1],
        num_channels=X_train.shape[2]
    )
    
    # Multi-GPU if available
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {num_gpus} GPUs")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, args.output)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(args.output)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save results
    results = {
        'experiment_name': args.experiment_name,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'train_samples': int(len(train_dataset)),
        'val_samples': int(len(val_dataset)),
        'test_samples': int(len(test_dataset)),
        'epochs_trained': args.epochs,
        'batch_size': args.batch_size,
        'num_gpus': num_gpus,
        'history': history
    }
    
    results_file = args.output.replace('.pt', '_results.json').replace('.pth', '_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to {args.output}")
    print(f"Results saved to {results_file}")
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
