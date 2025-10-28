#!/usr/bin/env python3
"""
Analyze u₀ distribution and detection limits
"""
import numpy as np
import matplotlib.pyplot as plt

# Load baseline data
data = np.load('../data/raw/events_baseline_1M.npz', allow_pickle=True)
binary_params_list = data['binary_params_list']

# Extract u0 values for binary events
u0_values = []
for params in binary_params_list:
    if params is not None and 'u0' in params:
        u0_values.append(params['u0'])

u0_values = np.array(u0_values)

print("="*60)
print("U₀ DISTRIBUTION ANALYSIS")
print("="*60)
print(f"\nTotal binary events: {len(u0_values):,}")
print(f"\nDetectability by u₀:")
print(f"  u₀ < 0.15 (distinct):     {(u0_values < 0.15).sum():>7,} ({(u0_values < 0.15).sum()/len(u0_values)*100:>5.1f}%)")
print(f"  u₀ < 0.30 (detectable):   {(u0_values < 0.30).sum():>7,} ({(u0_values < 0.30).sum()/len(u0_values)*100:>5.1f}%)")
print(f"  u₀ > 0.30 (PSPL-like):    {(u0_values > 0.30).sum():>7,} ({(u0_values > 0.30).sum()/len(u0_values)*100:>5.1f}%)")
print(f"\nStatistics:")
print(f"  Mean u₀:   {u0_values.mean():.3f}")
print(f"  Median u₀: {np.median(u0_values):.3f}")
print("="*60)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(u0_values, bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(0.3, color='red', linestyle='--', linewidth=2, 
                label='u₀ = 0.3 (Detection Limit)')
axes[0].axvline(0.15, color='orange', linestyle='--', linewidth=2,
                label='u₀ = 0.15 (Distinct)')
axes[0].set_xlabel('Impact Parameter u₀', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Binary Event u₀ Distribution', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Cumulative
sorted_u0 = np.sort(u0_values)
cumulative = np.arange(1, len(sorted_u0) + 1) / len(sorted_u0)
axes[1].plot(sorted_u0, cumulative, linewidth=2)
axes[1].axvline(0.3, color='red', linestyle='--', linewidth=2)
axes[1].axvline(0.15, color='orange', linestyle='--', linewidth=2)
axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.7)
axes[1].set_xlabel('Impact Parameter u₀', fontsize=12)
axes[1].set_ylabel('Cumulative Fraction', fontsize=12)
axes[1].set_title('Cumulative u₀ Distribution', fontsize=14)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('u0_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to analysis/u0_distribution.png")
