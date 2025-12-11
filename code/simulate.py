import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import math
import warnings
import sys
import os
import h5py
import shutil

warnings.filterwarnings("ignore")

# =============================================================================
# CAUSALITY ENFORCEMENT & CONFIGURATION
# =============================================================================
try:
    import VBBinaryLensing
    HAS_VBB = True
except ImportError:
    print("CRITICAL: VBBinaryLensing not found. Install via pip or conda.")
    sys.exit(1)

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found. Simulation will be slower.")

# ============================================================================
# CONSTANTS
# ============================================================================
ROMAN_ZP_FLUX_JY = 3631.0
ROMAN_MISSION_DURATION_DAYS = 200.0
# Define strict types for HDF5 structure to allow direct numpy writing
DTYPE_PARAMS = np.dtype([
    ('t0', 'f4'), ('tE', 'f4'), ('u0', 'f4'), 
    ('s', 'f4'), ('q', 'f4'), ('alpha', 'f4'), ('rho', 'f4'),
    ('m_base', 'f4'), ('type_id', 'i4')
])

# ============================================================================
# NUMBA FUNCTIONS
# ============================================================================
if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def flux_to_mag_numba(flux_jy):
        n = len(flux_jy)
        mag = np.empty(n, dtype=np.float32)
        zp = ROMAN_ZP_FLUX_JY
        for i in prange(n):
            f = flux_jy[i]
            if f > 0:
                mag[i] = -2.5 * math.log10(f / zp)
            else:
                mag[i] = np.nan
        return mag
    
    @njit(fastmath=True, cache=True)
    def compute_photon_noise_numba(flux_jy):
        # Simplified for speed - assuming inputs are valid
        n = len(flux_jy)
        sigma = np.empty(n, dtype=np.float32)
        
        # Pre-calc constants for Roman
        f_lim = ROMAN_ZP_FLUX_JY * 10**(-0.4 * 27.5)
        f_sky = ROMAN_ZP_FLUX_JY * 10**(-0.4 * 22.0)
        sigma_lim = f_lim / 5.0
        k_noise = sigma_lim / math.sqrt(f_lim + f_sky)
        
        for i in prange(n):
            f_total = flux_jy[i] + f_sky
            if f_total > 0:
                sigma[i] = k_noise * math.sqrt(f_total)
            else:
                sigma[i] = k_noise * 1e-5
        return sigma

    @njit(fastmath=True, cache=True)
    def single_mag_to_flux(mag):
        return ROMAN_ZP_FLUX_JY * 10**(-0.4 * mag)

    @njit(fastmath=True, cache=True, parallel=True)
    def pspl_magnification_fast(t, t_E, u_0, t_0):
        n = len(t)
        A = np.ones(n, dtype=np.float32)
        inv_tE = 1.0 / t_E
        u0_sq = u_0 * u_0
        for i in prange(n):
            tau = (t[i] - t_0) * inv_tE
            u_sq = u0_sq + tau * tau
            u_sqrt = np.sqrt(u_sq)
            A[i] = (u_sq + 2.0) / (u_sqrt * np.sqrt(u_sq + 4.0))
        return A

    @njit(fastmath=True, cache=True, parallel=True)
    def compute_delta_t_numba(times, mask):
        n = len(times)
        delta_t = np.zeros(n, dtype=np.float32)
        # Assuming times are sorted, we can do a forward pass
        # This implementation assumes the mask is boolean (1=observed, 0=gap)
        last_valid_idx = -1
        
        # Parallel is hard for dependency chains, doing serial fast scan
        # Note: For strict correctness with complex gaps, serial is safer here or 
        # a two-pass parallel algorithm. Given N=2400, serial compiled is instant.
        for i in range(n):
            if mask[i]:
                if last_valid_idx != -1:
                    delta_t[i] = times[i] - times[last_valid_idx]
                else:
                    delta_t[i] = 0.0
                last_valid_idx = i
            else:
                delta_t[i] = 0.0
        return delta_t

# ============================================================================
# CLASSES & CONFIG
# ============================================================================
class SimConfig:
    TIME_MIN = 0.0
    TIME_MAX = ROMAN_MISSION_DURATION_DAYS
    N_POINTS = 2400
    VBM_TOLERANCE = 1e-3
    CADENCE_MASK_PROB = 0.05
    PAD_VALUE = 0.0

class BinaryPresets:
    SHARED_T0_MIN = 0.2 * SimConfig.TIME_MAX
    SHARED_T0_MAX = 0.8 * SimConfig.TIME_MAX
    SHARED_TE_MIN = 5.0
    SHARED_TE_MAX = 30.0
    
    PRESETS = {
        'distinct': {'s': (0.90, 1.10), 'q': (0.1, 1.0), 'u0': (0.0001, 0.4), 'rho': (1e-4, 5e-3)},
        'planetary': {'s': (0.5, 2.0), 'q': (1e-4, 1e-2), 'u0': (0.001, 0.3), 'rho': (1e-4, 1e-2)},
        'stellar': {'s': (0.3, 3.0), 'q': (0.3, 1.0), 'u0': (0.001, 0.3), 'rho': (1e-3, 5e-2)},
        'baseline': {'s': (0.1, 3.0), 'q': (1e-4, 1.0), 'u0': (0.001, 1.0), 'rho': (1e-3, 0.1)}
    }

# ============================================================================
# LOGIC
# ============================================================================
def binary_magnification_vbb_safe(t, t_E, u_0, t_0, s, q, alpha, rho):
    """
    Attempts to calculate binary mag. Returns None if VBB fails.
    NEVER falls back to a python loop.
    """
    VBB = VBBinaryLensing.VBBinaryLensing()
    VBB.Tol = SimConfig.VBM_TOLERANCE
    tau = (t - t_0) / t_E
    u1 = -u_0 * math.sin(alpha) + tau * math.cos(alpha)
    u2 = u_0 * math.cos(alpha) + tau * math.sin(alpha)
    
    try:
        # Only use the vectorized C++ call
        return VBB.BinaryMag(s, q, u1, u2, rho)
    except:
        return None

def simulate_event(params):
    etype = params['type']
    t_grid = params['time_grid']
    n = len(t_grid)
    
    # Initialize Param Structure (Default values -1)
    p_out = np.zeros(1, dtype=DTYPE_PARAMS)[0]
    p_out['type_id'] = {'flat': 0, 'pspl': 1, 'binary': 2}[etype]
    
    m_base = np.random.uniform(18.0, 24.0)
    p_out['m_base'] = m_base
    f_base = single_mag_to_flux(m_base)

    if etype == 'flat':
        A = np.ones(n, dtype=np.float32)
        
    elif etype == 'pspl':
        t0 = np.random.uniform(BinaryPresets.SHARED_T0_MIN, BinaryPresets.SHARED_T0_MAX)
        tE = np.random.uniform(BinaryPresets.SHARED_TE_MIN, BinaryPresets.SHARED_TE_MAX)
        u0 = np.random.uniform(0.001, 1.0)
        A = pspl_magnification_fast(t_grid, tE, u0, t0)
        p_out['t0'], p_out['tE'], p_out['u0'] = t0, tE, u0
        
    elif etype == 'binary':
        ranges = BinaryPresets.PRESETS[params['preset']]
        
        # Try up to 5 times to get a valid binary event
        for _ in range(5):
            t0 = np.random.uniform(BinaryPresets.SHARED_T0_MIN, BinaryPresets.SHARED_T0_MAX)
            tE = np.random.uniform(BinaryPresets.SHARED_TE_MIN, BinaryPresets.SHARED_TE_MAX)
            s = np.random.uniform(*ranges['s'])
            q = 10**np.random.uniform(*np.log10(ranges['q']))
            u0 = np.random.uniform(*ranges['u0'])
            rho = 10**np.random.uniform(*np.log10(ranges['rho']))
            alpha = np.random.uniform(0, 2*math.pi)
            
            # Fast Check: Only run if VBB works
            A = binary_magnification_vbb_safe(t_grid, tE, u0, t0, s, q, alpha, rho)
            
            if A is not None:
                # Success
                p_out['t0'], p_out['tE'], p_out['u0'] = t0, tE, u0
                p_out['s'], p_out['q'], p_out['alpha'], p_out['rho'] = s, q, alpha, rho
                break
        else:
            # Fallback to PSPL if binary fails 5 times (keeps pipeline moving)
            u0 = np.random.uniform(0.001, 0.5)
            A = pspl_magnification_fast(t_grid, 20.0, u0, 100.0)
            p_out['type_id'] = 1 # Mark as PSPL fallback
            p_out['u0'] = u0
    
    # --- Flux & Noise ---
    flux_true = f_base * A
    noise = compute_photon_noise_numba(flux_true)
    flux_obs = flux_true + np.random.normal(0, noise) # Noise scale 1.0 fixed
    
    # --- Mag Conversion & Masking ---
    mag_obs = flux_to_mag_numba(flux_obs)
    
    # Apply Mask (0.05 prob of dropout)
    mask = np.random.random(n) > 0.05
    
    # Create output arrays
    # We leave Mag as NaN where masked, or 0.0 if padded. User preference PAD_VALUE=0.0
    mag_final = np.full(n, SimConfig.PAD_VALUE, dtype=np.float32)
    mag_final[mask] = mag_obs[mask]
    
    dt = compute_delta_t_numba(t_grid, mask)
    
    return mag_final, dt, p_out

def worker_func(args):
    # Unpack
    tasks, seed = args
    np.random.seed(seed)
    
    # Pre-allocate worker result arrays
    n_local = len(tasks)
    w_flux = np.zeros((n_local, SimConfig.N_POINTS), dtype=np.float32)
    w_dt = np.zeros((n_local, SimConfig.N_POINTS), dtype=np.float32)
    w_params = np.zeros(n_local, dtype=DTYPE_PARAMS)
    
    for i, task in enumerate(tasks):
        f, d, p = simulate_event(task)
        w_flux[i] = f
        w_dt[i] = d
        w_params[i] = p
        
    return w_flux, w_dt, w_params

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_flat', type=int, default=100)
    parser.add_argument('--n_pspl', type=int, default=100)
    parser.add_argument('--n_binary', type=int, default=100)
    parser.add_argument('--binary_preset', type=str, default='baseline')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_grid = np.linspace(SimConfig.TIME_MIN, SimConfig.TIME_MAX, SimConfig.N_POINTS)
    
    # Build Tasks
    base_params = {'time_grid': time_grid, 'preset': args.binary_preset}
    tasks = []
    tasks += [{'type': 'flat', **base_params} for _ in range(args.n_flat)]
    tasks += [{'type': 'pspl', **base_params} for _ in range(args.n_pspl)]
    tasks += [{'type': 'binary', **base_params} for _ in range(args.n_binary)]
    
    total_events = len(tasks)
    print(f"Plan: {total_events} events [{args.binary_preset}] -> {out_path}")

    # Shuffle
    np.random.seed(args.seed)
    np.random.shuffle(tasks)
    
    # Chunking
    workers = args.num_workers or cpu_count()
    chunk_size = min(500, math.ceil(total_events / (workers * 4))) # Smaller chunks for frequent writing
    chunks = [tasks[i:i + chunk_size] for i in range(0, total_events, chunk_size)]
    chunk_inputs = [(c, args.seed + i) for i, c in enumerate(chunks)]
    
    print(f"Processing in {len(chunks)} chunks using {workers} workers.")

    # ---------------------------------------------------------
    # HDF5 INCREMENTAL WRITE STRATEGY
    # ---------------------------------------------------------
    # Pre-allocate file on disk. This is much faster than dynamic resizing.
    # We use 'gzip' compression. 'lzf' is faster but less standard.
    comp = {'compression': 'gzip', 'compression_opts': 1} # Level 1 is fast
    
    with h5py.File(out_path, 'w') as f:
        d_flux = f.create_dataset('flux', (total_events, SimConfig.N_POINTS), dtype='f4', **comp)
        d_dt = f.create_dataset('delta_t', (total_events, SimConfig.N_POINTS), dtype='f4', **comp)
        d_params = f.create_dataset('params', (total_events,), dtype=DTYPE_PARAMS, **comp)
        
        # Save timestamp vector once
        f.create_dataset('timestamps', data=time_grid.astype(np.float32))

        # Start Processing
        curr_idx = 0
        with Pool(workers) as pool:
            # imap_unordered yields results as soon as they are ready
            iterator = pool.imap_unordered(worker_func, chunk_inputs)
            
            for res_flux, res_dt, res_params in tqdm(iterator, total=len(chunks)):
                n_batch = len(res_flux)
                end_idx = curr_idx + n_batch
                
                # Direct write to disk
                d_flux[curr_idx:end_idx] = res_flux
                d_dt[curr_idx:end_idx] = res_dt
                d_params[curr_idx:end_idx] = res_params
                
                curr_idx = end_idx
                
                # Optional: Flush occasionally if you fear crashes, 
                # but OS cache usually handles this fine.
                
    print(f"✅ Done. Saved {curr_idx} events to {out_path}")

if __name__ == '__main__':
    main()
