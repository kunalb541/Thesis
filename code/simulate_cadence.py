"""
Microlensing event simulation with configurable cadence
For testing different observational strategies (LSST, Roman, etc.)
"""

import numpy as np
import VBMicrolensing
import argparse
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.config import *

# Initialize VBMicrolensing
VBM = VBMicrolensing.VBMicrolensing()
VBM.RelTol = VBM_REL_TOL
VBM.Tol = VBM_TOL

def generate_pspl_event(n_observations=N_POINTS, 
                       mag_error_std=MAG_ERROR_STD, 
                       cadence_mask_prob=CADENCE_MASK_PROB):
    """
    Generate PSPL event
    """
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_observations)
    baseline = np.random.uniform(PSPL_BASELINE_MIN, PSPL_BASELINE_MAX)
    t0 = np.random.uniform(PSPL_T0_MIN, PSPL_T0_MAX)
    u0 = np.random.uniform(PSPL_U0_MIN, PSPL_U0_MAX)
    tE = np.random.uniform(PSPL_TE_MIN, PSPL_TE_MAX)

    # Paczynski magnification
    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))

    # Convert to magnitudes
    magnitudes = baseline - 2.5 * np.log10(magnification)

    # Add photometric errors
    magnitudes += np.random.normal(0, mag_error_std, size=magnitudes.shape)

    # Apply cadence masking (realistic gaps)
    mask = np.random.rand(n_observations) < cadence_mask_prob
    magnitudes[mask] = np.nan

    # Convert to flux
    flux = 10 ** (-(magnitudes - baseline) / 2.5)

    return timestamps, flux, {'t0': t0, 'u0': u0, 'tE': tE, 'baseline': baseline}

def generate_binary_event(n_points=N_POINTS, 
                         mag_error_std=MAG_ERROR_STD, 
                         cadence_mask_prob=CADENCE_MASK_PROB):
    """
    Generate Binary event using VBMicrolensing
    """
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points)
    
    # Sample binary parameters
    s = np.random.uniform(BINARY_S_MIN, BINARY_S_MAX)
    q = np.random.uniform(BINARY_Q_MIN, BINARY_Q_MAX)
    rho = np.random.uniform(BINARY_RHO_MIN, BINARY_RHO_MAX)
    alpha = np.random.uniform(BINARY_ALPHA_MIN, BINARY_ALPHA_MAX)
    tE = np.random.uniform(BINARY_TE_MIN, BINARY_TE_MAX)
    t0 = np.random.uniform(BINARY_T0_MIN, BINARY_T0_MAX)
    u0 = np.random.uniform(BINARY_U0_MIN, BINARY_U0_MAX)

    # VBMicrolensing expects log(s), log(q), log(rho), log(tE)
    params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    
    # Generate magnification using VBMicrolensing
    magnifications = np.array(VBM.BinaryLightCurve(params, timestamps)[0])

    # Add photometric errors
    magnifications += np.random.normal(0, mag_error_std, size=magnifications.shape)

    # Apply cadence masking
    mask = np.random.rand(n_points) < cadence_mask_prob
    magnifications[mask] = np.nan

    return timestamps, magnifications, {
        's': s, 'q': q, 'rho': rho, 'alpha': alpha, 
        'tE': tE, 't0': t0, 'u0': u0
    }

def pad_missing_data(data, pad_value=PAD_VALUE):
    """Replace NaN with pad value"""
    return np.nan_to_num(data, nan=pad_value)

def simulate_dataset(n_pspl, n_binary, cadence_prob, output_file):
    """
    Generate full dataset with specified cadence
    """
    print(f"Generating {n_pspl} PSPL events (cadence_prob={cadence_prob})...")
    pspl_data = []
    pspl_params = []
    
    for i in tqdm(range(n_pspl)):
        t, flux, params = generate_pspl_event(cadence_mask_prob=cadence_prob)
        flux_padded = pad_missing_data(flux)
        pspl_data.append(flux_padded)
        pspl_params.append(params)
    
    print(f"Generating {n_binary} Binary events (cadence_prob={cadence_prob})...")
    binary_data = []
    binary_params = []
    
    for i in tqdm(range(n_binary)):
        t, mags, params = generate_binary_event(cadence_mask_prob=cadence_prob)
        mags_padded = pad_missing_data(mags)
        binary_data.append(mags_padded)
        binary_params.append(params)
    
    # Combine into arrays
    X_pspl = np.array(pspl_data)[:, :, np.newaxis]  # Shape: (N, 1500, 1)
    X_binary = np.array(binary_data)[:, :, np.newaxis]
    
    X = np.vstack([X_pspl, X_binary])
    y = np.array(['PSPL'] * n_pspl + ['Binary'] * n_binary)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving to {output_file}...")
    np.savez_compressed(output_file, 
                       X=X, 
                       y=y,
                       pspl_params=pspl_params,
                       binary_params=binary_params,
                       cadence_prob=cadence_prob)
    
    print(f"Done! Generated {len(X)} events")
    print(f"Shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    print(f"Cadence masking probability: {cadence_prob}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pspl', type=int, default=N_PSPL)
    parser.add_argument('--n_binary', type=int, default=N_BINARY)
    parser.add_argument('--cadence_prob', type=float, default=CADENCE_MASK_PROB)
    parser.add_argument('--output', type=str, default=os.path.join(DATA_DIR, 'events.npz'))
    args = parser.parse_args()
    
    simulate_dataset(args.n_pspl, args.n_binary, args.cadence_prob, args.output)
