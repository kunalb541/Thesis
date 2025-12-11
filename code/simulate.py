import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import math
import warnings
import time
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
# MODULE-LEVEL CONSTANTS (NUMBA COMPATIBLE)
# ============================================================================
ROMAN_ZP_FLUX_JY = 3631.0
ROMAN_LIMITING_MAG_AB = 27.5
ROMAN_SKY_MAG_AB = 22.0
ROMAN_SOURCE_MAG_MIN = 18.0
ROMAN_SOURCE_MAG_MAX = 24.0
ROMAN_CADENCE_MINUTES = 12.1
ROMAN_MISSION_DURATION_DAYS = 200.0
ROMAN_LIMITING_SNR = 5.0

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
        n = len(flux_jy)
        sigma = np.empty(n, dtype=np.float32)
        zp = ROMAN_ZP_FLUX_JY
        f_lim = zp * 10**(-0.4 * ROMAN_LIMITING_MAG_AB)
        f_sky = zp * 10**(-0.4 * ROMAN_SKY_MAG_AB)
        sigma_lim = f_lim / ROMAN_LIMITING_SNR
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
        prev_valid = np.full(n, -1, dtype=np.int32)
        
        # Serial scan to find previous valid indices (fast enough)
        last = -1
        for i in range(n):
            if mask[i]:
                prev_valid[i] = last
                last = i
            else:
                prev_valid[i] = last # Propagate last seen
        
        # Parallel compute
        for i in prange(n):
            if mask[i] and prev_valid[i] != -1:
                delta_t[i] = times[i] - times[prev_valid[i]]
        return delta_t

# ============================================================================
# CLASSES
# ============================================================================
class RomanWFI_F146:
    N_POINTS = 2400
    
    @staticmethod
    def flux_to_mag(flux_jy):
        if HAS_NUMBA: return flux_to_mag_numba(flux_jy)
        with np.errstate(divide='ignore', invalid='ignore'):
            return -2.5 * np.log10(flux_jy / ROMAN_ZP_FLUX_JY)

    @staticmethod
    def compute_photon_noise(flux_jy):
        if HAS_NUMBA: return compute_photon_noise_numba(flux_jy)
        # Fallback implementation omitted for brevity, assuming Numba usually present
        return np.zeros_like(flux_jy) 

class SimConfig:
    TIME_MIN = 0.0
    TIME_MAX = ROMAN_MISSION_DURATION_DAYS
    N_POINTS = 2400
    VBM_TOLERANCE = 1e-3
    CADENCE_MASK_PROB = 0.05
    BASELINE_MIN = ROMAN_SOURCE_MAG_MIN
    BASELINE_MAX = ROMAN_SOURCE_MAG_MAX
    PAD_VALUE = 0.0

class BinaryPresets:
    SHARED_T0_MIN = 0.2 * SimConfig.TIME_MAX
    SHARED_T0_MAX = 0.8 * SimConfig.TIME_MAX
    SHARED_TE_MIN = 5.0
    SHARED_TE_MAX = 30.0
    
    PRESETS = {
        'distinct': {'s_range': (0.90, 1.10), 'q_range': (0.1, 1.0), 'u0_range': (0.0001, 0.4), 'rho_range': (1e-4, 5e-3), 'alpha_range': (0, 2*math.pi), 't0_range': (SHARED_T0_MIN, SHARED_T0_MAX), 'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX)},
        'planetary': {'s_range': (0.5, 2.0), 'q_range': (1e-4, 1e-2), 'u0_range': (0.001, 0.3), 'rho_range': (1e-4, 1e-2), 'alpha_range': (0, 2*math.pi), 't0_range': (SHARED_T0_MIN, SHARED_T0_MAX), 'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX)},
        'stellar': {'s_range': (0.3, 3.0), 'q_range': (0.3, 1.0), 'u0_range': (0.001, 0.3), 'rho_range': (1e-3, 5e-2), 'alpha_range': (0, 2*math.pi), 't0_range': (SHARED_T0_MIN, SHARED_T0_MAX), 'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX)},
        'baseline': {'s_range': (0.1, 3.0), 'q_range': (1e-4, 1.0), 'u0_range': (0.001, 1.0), 'rho_range': (1e-3, 0.1), 'alpha_range': (0, 2*math.pi), 't0_range': (SHARED_T0_MIN, SHARED_T0_MAX), 'tE_range': (SHARED_TE_MIN, SHARED_TE_MAX)}
    }

# ============================================================================
# MAGNIFICATION WRAPPERS
# ============================================================================
def pspl_magnification(t, t_E, u_0, t_0):
    if HAS_NUMBA: return pspl_magnification_fast(t, t_E, u_0, t_0)
    u = np.sqrt(u_0**2 + ((t - t_0) / t_E)**2)
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))

def binary_magnification_vbb_fast(t, t_E, u_0, t_0, s, q, alpha, rho):
    VBB = VBBinaryLensing.VBBinaryLensing()
    VBB.Tol = SimConfig.VBM_TOLERANCE
    tau = (t - t_0) / t_E
    u1 = -u_0 * math.sin(alpha) + tau * math.cos(alpha)
    u2 = u_0 * math.cos(alpha) + tau * math.sin(alpha)
    
    try:
        # Array-based VBB call if supported
        return VBB.BinaryMag(s, q, u1, u2, rho)
    except:
        # Fallback to loop
        n = len(t)
        mag = np.ones(n, dtype=np.float32)
        for i in range(n):
            val = VBB.BinaryMag2(s, q, u1[i], u2[i], rho)
            if val > 0: mag[i] = val
        return mag

def compute_delta_t_wrapper(times, mask):
    if HAS_NUMBA: return compute_delta_t_numba(times, mask)
    # Simple vector fallback
    valid_idx = np.where(mask)[0]
    dt = np.zeros_like(times)
    if len(valid_idx) > 0:
        diffs = np.diff(times[valid_idx])
        dt[valid_idx[1:]] = diffs
        dt[valid_idx[0]] = 0.0 # First obs has no prev
    return dt

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================
def simulate_event(params):
    """Unified simulation function."""
    etype = params['type']
    t_grid = params['time_grid']
    n = len(t_grid)
    
    # Defaults
    m_base = np.random.uniform(SimConfig.BASELINE_MIN, SimConfig.BASELINE_MAX)
    f_base = single_mag_to_flux(m_base) if HAS_NUMBA else ROMAN_ZP_FLUX_JY * 10**(-0.4*m_base)
    
    meta = {'type': etype, 'm_base': m_base}
    
    if etype == 'flat':
        A = np.ones(n, dtype=np.float32)
        label = 0
        
    elif etype == 'pspl':
        t0 = np.random.uniform(PSPLParams.T0_MIN, PSPLParams.T0_MAX)
        tE = np.random.uniform(PSPLParams.TE_MIN, PSPLParams.TE_MAX)
        u0 = np.random.uniform(PSPLParams.U0_MIN, PSPLParams.U0_MAX)
        A = pspl_magnification(t_grid, tE, u0, t0)
        label = 1
        meta.update({'t0': t0, 'tE': tE, 'u0': u0})
        
    elif etype == 'binary':
        p = BinaryPresets.PRESETS[params['preset']]
        t0 = np.random.uniform(*p['t0_range'])
        tE = np.random.uniform(*p['tE_range'])
        
        # Retry logic
        for _ in range(3):
            s = np.random.uniform(*p['s_range'])
            q = 10**np.random.uniform(*np.log10(p['q_range']))
            u0 = np.random.uniform(*p['u0_range'])
            rho = 10**np.random.uniform(*np.log10(p['rho_range']))
            alpha = np.random.uniform(*p['alpha_range'])
            
            try:
                A = binary_magnification_vbb_fast(t_grid, tE, u0, t0, s, q, alpha, rho)
                if np.max(A) > 1.1: break
            except: pass
        else:
            # Fallback to PSPL if binary fails
            u0 = np.random.uniform(0.001, 0.5)
            A = pspl_magnification(t_grid, tE, u0, t0)
            meta['used_fallback'] = True
            
        label = 2
        meta.update({'t0': t0, 'tE': tE, 'u0': u0, 's': s, 'q': q, 'alpha': alpha, 'rho': rho})

    # Apply Flux + Noise
    flux_true = f_base * A
    if HAS_NUMBA:
        noise = compute_photon_noise_numba(flux_true)
    else:
        noise = RomanWFI_F146.compute_photon_noise(flux_true)
        
    flux_obs = flux_true + np.random.normal(0, noise * params['noise_scale'])
    
    # Convert to Mag & Mask
    mag_obs = RomanWFI_F146.flux_to_mag(flux_obs)
    mask = np.random.random(n) > params['mask_prob']
    mag_obs[~mask] = SimConfig.PAD_VALUE
    
    # Delta T
    delta_t = compute_delta_t_wrapper(t_grid, mask)
    
    return {
        'flux': mag_obs.astype(np.float32),
        'delta_t': delta_t.astype(np.float32),
        'label': label,
        'params': meta
    }

def worker_func(args):
    chunk, seed = args
    np.random.seed(seed)
    return [simulate_event(p) for p in chunk]

# ============================================================================
# MASTER MERGE FUNCTION
# ============================================================================
def merge_node_files(final_output, num_nodes, compression=True):
    """Merges all partial node files into one final HDF5 file."""
    print(f"\n[Node 0] Waiting for {num_nodes-1} worker nodes to finish...")
    
    final_path = Path(final_output)
    base_name = final_path.stem
    temp_dir = final_path.parent
    
    # Wait loop
    expected_files = [temp_dir / f"{base_name}_part_{i}.h5" for i in range(num_nodes)]
    
    # Simple polling (wait up to 60 mins)
    start_wait = time.time()
    while True:
        missing = [f for f in expected_files if not f.exists()]
        if not missing:
            break
        if time.time() - start_wait > 3600:
            print(f"TIMEOUT: Missing files: {missing}")
            sys.exit(1)
        time.sleep(5)
        print(f"Waiting... {len(missing)} nodes remaining", end='\r')

    print("\n[Node 0] All parts found. Merging...")
    
    # Determine total size
    total_len = 0
    with h5py.File(expected_files[0], 'r') as f:
        shape = f['flux'].shape
        dtype = f['flux'].dtype
        n_points = shape[1]
    
    for fpath in expected_files:
        with h5py.File(fpath, 'r') as f:
            total_len += f['flux'].shape[0]

    # Create Final File
    comp_args = {'compression': 'gzip', 'compression_opts': 4} if compression else {}
    
    with h5py.File(final_path, 'w') as f_out:
        dset_flux = f_out.create_dataset('flux', (total_len, n_points), dtype='f4', **comp_args)
        dset_dt = f_out.create_dataset('delta_t', (total_len, n_points), dtype='f4', **comp_args)
        dset_lbl = f_out.create_dataset('labels', (total_len,), dtype='i4')
        dset_time = f_out.create_dataset('timestamps', (total_len, n_points), dtype='f4', **comp_args)
        
        # Merge Loop
        curr_idx = 0
        all_params = {'flat': [], 'pspl': [], 'binary': []}
        
        for i, fpath in enumerate(tqdm(expected_files, desc="Merging")):
            with h5py.File(fpath, 'r') as f_in:
                n = f_in['flux'].shape[0]
                dset_flux[curr_idx:curr_idx+n] = f_in['flux'][:]
                dset_dt[curr_idx:curr_idx+n] = f_in['delta_t'][:]
                dset_lbl[curr_idx:curr_idx+n] = f_in['labels'][:]
                dset_time[curr_idx:curr_idx+n] = f_in['timestamps'][:]
                
                # Collect params
                for k in f_in.keys():
                    if k.startswith('params_'):
                        ptype = k.split('_')[1]
                        all_params[ptype].append(f_in[k][:])
                
                curr_idx += n
            
            # Delete partial file
            os.remove(fpath)

        # Save Params
        for ptype, data_list in all_params.items():
            if data_list:
                concat_data = np.concatenate(data_list)
                f_out.create_dataset(f'params_{ptype}', data=concat_data, **comp_args)

    print(f"✅ SUCCESSFULLY MERGED {total_len} events into {final_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_flat', type=int, default=1000)
    parser.add_argument('--n_pspl', type=int, default=1000)
    parser.add_argument('--n_binary', type=int, default=1000)
    parser.add_argument('--binary_preset', type=str, default='baseline')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # SLURM AUTO-DETECTION
    rank = int(os.environ.get('SLURM_PROCID', 0))
    num_nodes = int(os.environ.get('SLURM_NNODES', 1))
    
    # Calculate Node Slice
    total_events = args.n_flat + args.n_pspl + args.n_binary
    
    # Create Task List
    time_grid = np.linspace(SimConfig.TIME_MIN, SimConfig.TIME_MAX, SimConfig.N_POINTS)
    base_params = {
        'time_grid': time_grid, 
        'mask_prob': SimConfig.CADENCE_MASK_PROB, 
        'noise_scale': 1.0,
        'preset': args.binary_preset
    }
    
    tasks = []
    tasks += [{'type': 'flat', **base_params} for _ in range(args.n_flat)]
    tasks += [{'type': 'pspl', **base_params} for _ in range(args.n_pspl)]
    tasks += [{'type': 'binary', **base_params} for _ in range(args.n_binary)]
    
    # Deterministic Shuffle & Slice
    np.random.seed(args.seed)
    np.random.shuffle(tasks)
    
    chunk_per_node = int(math.ceil(len(tasks) / num_nodes))
    start = rank * chunk_per_node
    end = min(start + chunk_per_node, len(tasks))
    my_tasks = tasks[start:end]
    
    print(f"[Node {rank}/{num_nodes}] Processing {len(my_tasks)} events ({start}-{end})...")
    
    # Parallel Generation
    workers = args.num_workers or cpu_count()
    chunks = np.array_split(my_tasks, workers * 4)
    chunk_inputs = [(c, args.seed + rank*1000 + i) for i, c in enumerate(chunks)]
    
    results = []
    with Pool(workers) as pool:
        for res in tqdm(pool.imap_unordered(worker_func, chunk_inputs), total=len(chunks), desc=f"Node {rank}"):
            results.extend(res)
            
    # Aggregate Arrays
    n_res = len(results)
    flux = np.zeros((n_res, SimConfig.N_POINTS), dtype=np.float32)
    dt = np.zeros((n_res, SimConfig.N_POINTS), dtype=np.float32)
    lbl = np.zeros(n_res, dtype=np.int32)
    ts = np.tile(time_grid.astype(np.float32), (n_res, 1))
    
    params_struct = {'flat': [], 'pspl': [], 'binary': []}
    
    for i, r in enumerate(results):
        flux[i] = r['flux']
        dt[i] = r['delta_t']
        lbl[i] = r['label']
        
        # Handle Params
        ptype = r['params']['type']
        # Convert dict to simple values list for saving (simplified for speed)
        # In real scenario, align keys
        params_struct[ptype].append(r['params'])

    # Save Partial File
    out_path = Path(args.output)
    temp_name = out_path.parent / f"{out_path.stem}_part_{rank}.h5"
    
    print(f"Saving partial file: {temp_name}")
    with h5py.File(temp_name, 'w') as f:
        f.create_dataset('flux', data=flux)
        f.create_dataset('delta_t', data=dt)
        f.create_dataset('labels', data=lbl)
        f.create_dataset('timestamps', data=ts)
        
        # Save structured params
        for ptype, plist in params_struct.items():
            if plist:
                keys = list(plist[0].keys())
                # Exclude non-scalar if any
                dt_list = [(k, 'f4') for k in keys if isinstance(plist[0][k], (float, int))]
                arr = np.zeros(len(plist), dtype=dt_list)
                for j, p in enumerate(plist):
                    for k in keys:
                        if k in p and isinstance(p[k], (float, int)):
                            arr[j][k] = p[k]
                f.create_dataset(f'params_{ptype}', data=arr)

    # Master Node Logic
    if rank == 0:
        merge_node_files(args.output, num_nodes)
    else:
        print(f"[Node {rank}] Work complete. Exiting.")

if __name__ == '__main__':
    main()
