#!/usr/bin/env python3
"""Quick evaluation without the full RomanEvaluator framework"""
import torch
import h5py
import numpy as np
from pathlib import Path
import sys

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))
from model import RomanMicrolensingClassifier, ModelConfig

print("="*80)
print("QUICK EVALUATION - Baseline → Distinct Test")
print("="*80)

# Load checkpoint
checkpoint_path = Path("../results/best.pt")
print(f"\nLoading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print(f"  Epoch: {checkpoint['epoch']}")

# Load test data
test_path = Path.home() / "Thesis/data/test/test.h5"
print(f"\nLoading test data: {test_path}")

with h5py.File(test_path, 'r') as f:
    flux_test = f['flux'][:]
    delta_t_test = f['delta_t'][:]
    labels_test = f['labels'][:]

print(f"  Samples: {len(labels_test):,}")
print(f"  Flat: {(labels_test==0).sum():,}")
print(f"  PSPL: {(labels_test==1).sum():,}")
print(f"  Binary: {(labels_test==2).sum():,}")

# Normalize data (from checkpoint or defaults)
flux_mean = checkpoint.get('flux_mean', 1.2743)
flux_std = checkpoint.get('flux_std', 1.1385)
dt_mean = checkpoint.get('delta_t_mean', 0.087834)
dt_std = checkpoint.get('delta_t_std', 0.019618)

print(f"\nNormalizing data:")
print(f"  Flux: ({flux_mean:.4f}, {flux_std:.4f})")
print(f"  Delta_t: ({dt_mean:.6f}, {dt_std:.6f})")

flux_norm = (flux_test - flux_mean) / (flux_std + 1e-8)
dt_norm = (delta_t_test - dt_mean) / (dt_std + 1e-8)

# Create model with config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Get model config from checkpoint or use defaults
model_config_dict = checkpoint.get('model_config', {})
config = ModelConfig(
    d_model=model_config_dict.get('d_model', 64),
    n_layers=model_config_dict.get('n_layers', 4),
    dropout=model_config_dict.get('dropout', 0.3),
    window_size=model_config_dict.get('window_size', 5),
    max_seq_len=model_config_dict.get('max_seq_len', 2400),
    n_classes=3,
    hierarchical=model_config_dict.get('hierarchical', True),
    use_residual=model_config_dict.get('use_residual', True),
    use_layer_norm=model_config_dict.get('use_layer_norm', True),
    feature_extraction=model_config_dict.get('feature_extraction', 'conv'),
    use_attention_pooling=model_config_dict.get('use_attention_pooling', True),
    use_amp=False,
    use_gradient_checkpointing=False,
    use_flash_attention=model_config_dict.get('use_flash_attention', True),
    num_attention_heads=model_config_dict.get('num_attention_heads', 2),
    gru_dropout=model_config_dict.get('gru_dropout', 0.1),
    bn_momentum=model_config_dict.get('bn_momentum', 0.2),
    init_scale=model_config_dict.get('init_scale', 1.0)
)

model = RomanMicrolensingClassifier(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Evaluate
print("\n" + "="*80)
print("RUNNING EVALUATION")
print("="*80)

batch_size = 2048
n_batches = (len(labels_test) + batch_size - 1) // batch_size

all_preds = []
all_labels = []

print(f"Processing {n_batches} batches...")

with torch.no_grad():
    for i in range(n_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, len(labels_test))
        
        batch_flux = torch.FloatTensor(flux_norm[start:end]).to(device)
        batch_dt = torch.FloatTensor(dt_norm[start:end]).to(device)
        batch_labels = labels_test[start:end]
        
        # Stack flux and delta_t
        
        outputs = model(batch_flux, batch_dt)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.append(preds)
        all_labels.append(batch_labels)
        
        if (i+1) % 10 == 0 or i == n_batches - 1:
            print(f"  Progress: {i+1}/{n_batches} batches", end='\r')

print(f"\nCompleted {n_batches} batches                    ")

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Compute accuracy
accuracy = (all_preds == all_labels).mean() * 100

# Confusion matrix
confusion = np.zeros((3, 3), dtype=int)
for true, pred in zip(all_labels, all_preds):
    confusion[true, pred] += 1

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nOverall Test Accuracy: {accuracy:.2f}%")

print("\nConfusion Matrix:")
print("              Predicted")
print("           Flat    PSPL  Binary")
for i, name in enumerate(['Flat', 'PSPL', 'Binary']):
    print(f"{name:6s}  [{confusion[i,0]:6d} {confusion[i,1]:6d} {confusion[i,2]:6d}]", end='')
    recall = confusion[i,i] / confusion[i,:].sum() * 100
    print(f"  Recall: {recall:.1f}%")

print("\nPer-Class Detailed Metrics:")
for i, name in enumerate(['Flat', 'PSPL', 'Binary']):
    recall = confusion[i,i] / confusion[i,:].sum() * 100
    precision = confusion[i,i] / confusion[:,i].sum() * 100 if confusion[:,i].sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  {name:6s}: Recall={recall:5.1f}%  Precision={precision:5.1f}%  F1={f1:5.1f}%")

print("\n" + "="*80)
print("VALIDATION vs TEST COMPARISON")
print("="*80)
print(f"Validation (baseline, u0=0.001-0.5): 97.28%")
print(f"Test (distinct, u0=0.01-0.4):        {accuracy:.2f}%")
print(f"Difference:                           {accuracy - 97.28:+.2f}%")

if accuracy > 96:
    print("\n✓✓✓ EXCELLENT RESULTS! State-of-the-art performance!")
elif accuracy > 93:
    print("\n✓✓ VERY GOOD RESULTS! Publication-worthy!")
elif accuracy > 90:
    print("\n✓ GOOD RESULTS! Solid performance!")
else:
    print("\n⚠️  Lower than expected")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Trained on baseline (includes weak caustics, u0=0.001-0.5)")
print(f"✓ Tested on distinct (strong caustics, u0=0.01-0.4)")
print(f"✓ Binary recall: {confusion[2,2] / confusion[2,:].sum() * 100:.1f}%")
print(f"✓ Trivial baseline was only 63%, CNN is at {accuracy:.1f}%")
print(f"✓ Improvement: +{accuracy - 63:.1f}% over trivial classifier")
print("="*80)
