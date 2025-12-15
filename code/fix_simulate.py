import sys

with open('simulate.py', 'r') as f:
    content = f.read()

# Find the return statement in simulate_event function
old_return = """    return {
        'flux': mag_obs.astype(np.float32),  # Named 'flux' for compatibility, contains magnitudes
        'delta_t': delta_t.astype(np.float32),
        'label': label,
        'params': meta
    }"""

new_return = """    return {
        'flux': flux_obs.astype(np.float32),  # Actual flux values in Jansky
        'delta_t': delta_t.astype(np.float32),
        'label': label,
        'params': meta
    }"""

if old_return in content:
    content = content.replace(old_return, new_return)
    print("✓ Fixed return statement to output flux instead of magnitudes")
else:
    print("ERROR: Could not find the return statement to fix!")
    sys.exit(1)

# Also update the HDF5 save comment
content = content.replace(
    "'flux dataset contains AB magnitudes (not flux) for backward compatibility'",
    "'flux dataset contains actual flux values in Jansky'"
)

with open('simulate.py', 'w') as f:
    f.write(content)

print("✓ simulate.py fixed!")
