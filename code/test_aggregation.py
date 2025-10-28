"""
Test different aggregation strategies for TimeDistributed CNN
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load the trained baseline model
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

def test_aggregation_method(model, loader, device, method='last'):
    """Test different aggregation methods"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)  # (batch, time, classes)
            
            if method == 'last':
                # Current method: use last timestep
                probs = torch.softmax(outputs[:, -1, :], dim=1)
            elif method == 'mean':
                # Average predictions across all timesteps
                probs = torch.softmax(outputs, dim=2).mean(dim=1)
            elif method == 'max':
                # Max pooling across timesteps
                probs = torch.softmax(outputs, dim=2).max(dim=1)[0]
            elif method == 'middle':
                # Use middle timestep
                mid = outputs.shape[1] // 2
                probs = torch.softmax(outputs[:, mid, :], dim=1)
            elif method == 'weighted':
                # Weighted average (weight later timesteps more)
                weights = torch.linspace(0.5, 1.5, outputs.shape[1]).to(device)
                weights = weights / weights.sum()
                probs = (torch.softmax(outputs, dim=2) * weights.view(1, -1, 1)).sum(dim=1)
            
            _, predicted = torch.max(probs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return correct / total

# Main test
print("="*60)
print("AGGREGATION METHOD COMPARISON")
print("="*60)

# Load baseline model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Load data (use cadence_05 since it worked)
print("\nLoading data...")
data = np.load('../data/raw/events_cadence_05.npz')
X, y = data['X'], data['y']

# Encode labels
label_map = {'PSPL': 0, 'Binary': 1}
y_encoded = np.array([label_map[label] for label in y])

# Use last 15% as test
n_test = int(0.15 * len(X))
X_test = X[-n_test:]
y_test = y_encoded[-n_test:]

# Standardize (fit on first 70%)
n_train = int(0.7 * len(X))
scaler = StandardScaler()
scaler.fit(X[:n_train].reshape(-1, X.shape[-1]))
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Create dataset
test_dataset = LightCurveDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Test samples: {len(X_test)}")

# Load trained model (if baseline_v2 exists, use it; otherwise use baseline)
import glob
model_paths = glob.glob('../results/baseline_v2_*/best_model.pt')
if not model_paths:
    model_paths = glob.glob('../results/baseline_*/best_model.pt')

if model_paths:
    model_path = model_paths[0]
    print(f"\nLoading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = TimeDistributedCNN(X_test.shape[1], X_test.shape[2])
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print("\nTesting different aggregation methods...")
    print("-"*60)
    
    methods = ['last', 'mean', 'max', 'middle', 'weighted']
    results = {}
    
    for method in methods:
        acc = test_aggregation_method(model, test_loader, device, method)
        results[method] = acc
        print(f"{method:12s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Find best method
    best_method = max(results, key=results.get)
    print("-"*60)
    print(f"\n✓ Best method: {best_method} with {results[best_method]:.4f} accuracy")
    print(f"  Improvement over 'last': {(results[best_method] - results['last'])*100:.2f}%")
    
else:
    print("\n❌ No trained model found!")
    print("Run baseline_v2 training first.")

print("="*60)
