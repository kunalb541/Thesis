#!/usr/bin/env python3
"""
Apply masking fix to simulate.py
"""

import re

# Read simulate.py
with open('simulate.py', 'r') as f:
    content = f.read()

# Find and replace the pick_mask function
old_pick_mask = r'''def pick_mask\(i\):
    if not USE_SHARED_MASK or shared_masks is None or MASK_POOL_SIZE == 0:
        return None
    return shared_masks\[i % MASK_POOL_SIZE\]'''

new_pick_mask = '''def pick_mask(i, rng, n_points, prob):
    """Generate or retrieve a mask for event i"""
    if USE_SHARED_MASK and shared_masks is not None and len(shared_masks) > 0:
        return shared_masks[i % len(shared_masks)]
    elif prob > 0:
        # Generate individual mask
        return rng.rand(n_points) < prob
    else:
        return None'''

# Apply fix
content_fixed = re.sub(old_pick_mask, new_pick_mask, content, flags=re.MULTILINE)

# Update worker function calls
old_args = r"pick_mask\(i\)"
new_args = "pick_mask(i, np.random.RandomState(seed + i if seed else None), len(timestamps), cadence_mask_prob)"

content_fixed = content_fixed.replace(
    "mask = pick_mask(i)",
    "# Mask will be generated in worker"
)

# Backup original
import shutil
shutil.copy('simulate.py', 'simulate.py.backup')

# Write fixed version
with open('simulate.py', 'w') as f:
    f.write(content_fixed)

print("✅ Applied masking fix to simulate.py")
print("   Original backed up to: simulate.py.backup")
