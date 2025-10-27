"""
Microlensing event simulation with configurable binary parameters
"""

import numpy as np
import VBMicrolensing
import argparse
from tqdm import tqdm
from config import *

VBM = VBMicrolensing.VBMicrolensing()
VBM.RelTol = VBM_REL_TOL
VBM.Tol = VBM_TOL

def generate_pspl_event(n_observations=N_POINTS, mag_error_std=MAG_ERROR_STD, 
                       cadence_mask_prob=CADENCE_MASK_PROB):
    """Generate PSPL event"""
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_observations)
    baseline = np.random.uniform(PSPL_BASELINE_MIN, PSPL_BASELINE_MAX)
    t0 = np.random.uniform(PSPL_T0_MIN, PSPL_T0_MAX)
    u0 = np.random.uniform(PSPL_U0_MIN, PSPL_U0_MAX)
    tE = np.random.uniform(PSPL_TE_MIN, PSPL_TE_MAX)

    u_t = np.sqrt(u0**2 + ((timestamps - t0) / tE)**2)
    magnification = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))
    magnitudes = baseline - 2.5 * np.log10(magnification)
    magnitudes += np.random.normal(0, mag_error_std, size=magnitudes.shape)

    mask = np.random.rand(n_observations) < cadence_mask_prob
    magnitudes[mask] = np.nan

    flux = 10 ** (-(magnitudes - baseline) / 2.5)

    return timestamps, flux, {'t0': t0, 'u0': u0, 'tE': tE, 'baseline': baseline}

def generate_binary_event(n_points=N_POINTS, mag_error_std=MAG_ERROR_STD, 
                         cadence_mask_prob=CADENCE_MASK_PROB,
                         binary_params='standard'):
    """
    Generate Binary event using VBMicrolensing
    
    Args:
        binary_params: 'standard', 'easy', or 'hard'
            - standard: Mixed difficulty
            - easy: Clear caustic crossings (small u0, s~1, small rho)
            - hard: PSPL-like (large u0, extreme s, large rho)
    """
    timestamps = np.linspace(TIME_MIN, TIME_MAX, n_points)
    
    # Get parameter set
    param_set = BINARY_PARAM_SETS.get(binary_params, BINARY_PARAMS_STANDARD)
    
    # Sample binary parameters from specified ranges
    s = np.random.uniform(param_set['s_min'], param_set['s_max'])
    q = np.random.uniform(param_set['q_min'], param_set['q_max'])
    rho = np.random.uniform(param_set['rho_min'], param_set['rho_max'])
    alpha = np.random.uniform(param_set['alpha_min'], param_set['alpha_max'])
    tE = np.random.uniform(param_set['tE_min'], param_set['tE_max'])
    t0 = np.random.uniform(param_set['t0_min'], param_set['t0_max'])
    u0 = np.random.uniform(param_set['u0_min'], param_set['u0_max'])

    # VBMicrolensing expects log parameters
    params = [np.log(s), np.log(q), u0, alpha, np.log(rho), np.log(tE), t0]
    
    # Generate magnification
    magnifications = np.array(VBM.BinaryLightCurve(params, timestamps)[0])

    # Add errors
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

def simulate_dataset(n_pspl, n_binary, output_file, cadence_mask_prob=CADENCE_MASK_PROB,
                    mag_error_std=MAG_ERROR_STD, binary_params='standard'):
    """
    Generate full dataset
    
    Args:
        n_pspl: Number of PSPL events
        n_binary: Number of binary events
        output_file: Where to save
        cadence_mask_prob: Fraction of missing observations
        mag_error_std: Photometric error (magnitudes)
        binary_params: 'standard', 'easy', or 'hard'
    """
    print(f"Configuration:")
    print(f"  PSPL events: {n_pspl}")
    print(f"  Binary events: {n_binary}")
    print(f"  Cadence: {(1-cadence_mask_prob)*100:.0f}% coverage")
    print(f"  Photometric error: {mag_error_std:.3f} mag")
    print(f"  Binary difficulty: {binary_params}")
    print()
    
    print(f"Generating {n_pspl} PSPL events...")
    pspl_data = []
    pspl_params = []
    
    for i in tqdm(range(n_pspl)):
        t, flux, params = generate_pspl_event(
            mag_error_std=mag_error_std,
            cadence_mask_prob=cadence_mask_prob
        )
        flux_padded = pad_missing_data(flux)
        pspl_data.append(flux_padded)
        pspl_params.append(params)
    
    print(f"Generating {n_binary} Binary events ({binary_params})...")
    binary_data = []
    binary_params_list = []
    
    for i in tqdm(range(n_binary)):
        t, mags, params = generate_binary_event(
            mag_error_std=mag_error_std,
            cadence_mask_prob=cadence_mask_prob,
            binary_params=binary_params
        )
        mags_padded = pad_missing_data(mags)
        binary_data.append(mags_padded)
        binary_params_list.append(params)
    
    # Combine
    X_pspl = np.array(pspl_data)[:, :, np.newaxis]
    X_binary = np.array(binary_data)[:, :, np.newaxis]
    
    X = np.vstack([X_pspl, X_binary])
    y = np.array(['PSPL'] * n_pspl + ['Binary'] * n_binary)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Saving to {output_file}...")
    np.savez_compressed(output_file, 
                       X=X, 
                       y=y,
                       pspl_params=pspl_params,
                       binary_params=binary_params_list)
    
    print(f"Done! Generated {len(X)} events")
    print(f"Shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pspl', type=int, default=N_PSPL)
    parser.add_argument('--n_binary', type=int, default=N_BINARY)
    parser.add_argument('--output', type=str, default=DATA_DIR + '/events_1M.npz')
    parser.add_argument('--cadence', type=float, default=CADENCE_MASK_PROB,
                       help='Fraction of missing observations (0-1)')
    parser.add_argument('--error', type=float, default=MAG_ERROR_STD,
                       help='Photometric error in magnitudes')
    parser.add_argument('--binary_difficulty', type=str, default='standard',
                       choices=['standard', 'easy', 'hard'],
                       help='Binary event difficulty')
    args = parser.parse_args()
    
    simulate_dataset(args.n_pspl, args.n_binary, args.output,
                    cadence_mask_prob=args.cadence,
                    mag_error_std=args.error,
                    binary_params=args.binary_difficulty)
