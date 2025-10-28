import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load small subset
data = np.load('../data/raw/events_cadence_05.npz')
X, y = data['X'], data['y']

# Encode labels
label_map = {'PSPL': 0, 'Binary': 1}
y_enc = np.array([label_map[label] for label in y])

print(f"Data shape: {X.shape}")
print(f"Labels: {np.unique(y_enc, return_counts=True)}")

# Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split
split = int(0.8 * len(X))
train_dataset = SimpleDataset(X[:split], y_enc[:split])
val_dataset = SimpleDataset(X[split:], y_enc[split:])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print("\nTraining 10 epochs...")
for epoch in range(10):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
    
    print(f"Epoch {epoch+1}: Train Acc = {train_correct/train_total:.4f}, Val Acc = {val_correct/val_total:.4f}")

print("\n✓ If accuracies increase, training works!")
print("✗ If accuracies stay ~0.5, there's a fundamental issue")
