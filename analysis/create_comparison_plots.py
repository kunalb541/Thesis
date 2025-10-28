#!/usr/bin/env python3
"""
Create comparison plots across experiments
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results_summary.json') as f:
    results = json.load(f)

# 1. Cadence Comparison
fig, ax = plt.subplots(figsize=(10, 6))

cadence_experiments = {
    'cadence_05': (5, 'Dense (5% missing)'),
    'baseline': (20, 'Baseline (20% missing)'),
    'cadence_40': (40, 'Sparse (40% missing)'),
}

missing_pct = []
accuracies = []
labels = []

for exp, (pct, label) in cadence_experiments.items():
    if exp in results:
        missing_pct.append(pct)
        accuracies.append(results[exp]['accuracy'])
        labels.append(label)

ax.plot(missing_pct, accuracies, 'o-', linewidth=2, markersize=10, color='#2E86AB')
ax.set_xlabel('Missing Observations (%)', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Impact of Observing Cadence on Binary Detection', fontsize=14, pad=15)
ax.grid(alpha=0.3)
ax.set_ylim([0.5, 0.85])

# Annotate points
for x, y, label in zip(missing_pct, accuracies, labels):
    ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('cadence_comparison.png', dpi=300, bbox_inches='tight')
print("✓ cadence_comparison.png saved")

# 2. Binary Difficulty Comparison
fig, ax = plt.subplots(figsize=(10, 6))

difficulty_experiments = [
    ('distinct', 'Distinct\n(u₀<0.15)'),
    ('baseline', 'Baseline\n(wide u₀)'),
]

exp_names = []
exp_accs = []

for exp, label in difficulty_experiments:
    if exp in results:
        exp_names.append(label)
        exp_accs.append(results[exp]['accuracy'])

bars = ax.bar(range(len(exp_names)), exp_accs, alpha=0.8, edgecolor='black', linewidth=2)

# Color by performance
colors = ['green' if acc > 0.80 else 'orange' if acc > 0.70 else 'red' for acc in exp_accs]
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels(exp_names, fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Performance vs Binary Topology', fontsize=14, pad=15)
ax.set_ylim([0.5, 1.0])
ax.grid(alpha=0.3, axis='y')

# Add value labels
for i, acc in enumerate(exp_accs):
    ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('difficulty_comparison.png', dpi=300, bbox_inches='tight')
print("✓ difficulty_comparison.png saved")

# 3. Error Comparison
if 'error_low' in results and 'error_high' in results:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_experiments = [
        ('error_low', '0.05 mag\n(Space)'),
        ('baseline', '0.10 mag\n(Baseline)'),
        ('error_high', '0.20 mag\n(Poor)'),
    ]
    
    err_names = []
    err_accs = []
    
    for exp, label in error_experiments:
        if exp in results:
            err_names.append(label)
            err_accs.append(results[exp]['accuracy'])
    
    bars = ax.bar(range(len(err_names)), err_accs, alpha=0.8, edgecolor='black', linewidth=2,
                  color='#A23B72')
    
    ax.set_xticks(range(len(err_names)))
    ax.set_xticklabels(err_names, fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Impact of Photometric Error on Binary Detection', fontsize=14, pad=15)
    ax.set_ylim([0.5, 0.85])
    ax.grid(alpha=0.3, axis='y')
    
    for i, acc in enumerate(err_accs):
        ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('error_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ error_comparison.png saved")

print("\n✓ All comparison plots created")
