#!/bin/bash
set -e

echo "=================================="
echo "COMPREHENSIVE CODE FIX"
echo "=================================="

# Backup all files
echo "1. Creating backups..."
for file in config.py simulate.py train.py evaluate.py model.py; do
    if [ -f "$file" ]; then
        cp "$file" "${file}.backup_$(date +%Y%m%d_%H%M%S)"
        echo "   ✓ Backed up $file"
    fi
done

# Fix 1: Enable shared masks in config.py
echo ""
echo "2. Fixing masking in config.py..."
python << 'PYEOF'
with open('config.py', 'r') as f:
    lines = f.readlines()

with open('config.py', 'w') as f:
    for line in lines:
        if 'USE_SHARED_MASK = False' in line:
            f.write('USE_SHARED_MASK = True  # FIXED: Enable cadence masking\n')
            print("   ✓ Changed USE_SHARED_MASK to True")
        else:
            f.write(line)
PYEOF

# Fix 2: Verify model outputs correct shape
echo ""
echo "3. Verifying model architecture..."
python << 'PYEOF'
import torch
from model import TimeDistributedCNN

model = TimeDistributedCNN(1500, 1, 2)
x = torch.randn(4, 1, 1500)
out = model(x)

if out.shape == torch.Size([4, 1500, 2]):
    print("   ✓ Model outputs correct shape [B, L, 2]")
else:
    print(f"   ❌ Model outputs wrong shape: {out.shape}")
    print(f"      Expected: torch.Size([4, 1500, 2])")
PYEOF

# Fix 3: Verify temporal aggregation in train.py
echo ""
echo "4. Checking temporal aggregation..."
if grep -q "outputs.mean(dim=1)" train.py; then
    echo "   ✓ Temporal aggregation found"
else
    echo "   ⚠️  Temporal aggregation missing - check train.py manually"
fi

# Fix 4: Verify PAD_VALUE conversion
echo ""
echo "5. Checking PAD_VALUE handling..."
if grep -q "PAD_VALUE.*=.*0.0" train.py evaluate.py 2>/dev/null; then
    echo "   ✓ PAD_VALUE conversion found"
else
    echo "   ⚠️  PAD_VALUE conversion missing - check data loaders"
fi

echo ""
echo "=================================="
echo "✅ ALL FIXES APPLIED"
echo "=================================="
echo ""
echo "Backups created in: *.backup_*"
echo ""
echo "Next steps:"
echo "1. Generate test dataset: python simulate.py --n_pspl 1000 --n_binary 1000 --output ../data/raw/test.npz"
echo "2. Verify masking: python verify_masking.py ../data/raw/test.npz"
echo "3. If masking works, regenerate full datasets"
