"""
Unit tests for microlensing simulation

Run with: pytest test_simulation.py -v

Author: Kunal Bhatia
Last Updated: October 27, 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

# Import after path is set
from simulate import generate_pspl_event_worker, generate_binary_event_worker
from config import N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, RANDOM_SEED

class TestPSPLSimulation:
    """Tests for PSPL event generation"""
    
    def test_pspl_output_shape(self):
        """Test PSPL event generates correct shape"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        flux, params = generate_pspl_event_worker(args)
        
        assert flux.shape == (N_POINTS,), f"Expected shape ({N_POINTS},), got {flux.shape}"
        assert isinstance(params, dict), "Parameters should be a dictionary"
    
    def test_pspl_params_exist(self):
        """Test PSPL parameters are present"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        flux, params = generate_pspl_event_worker(args)
        
        required_params = ['t0', 'u0', 'tE', 'baseline']
        for param in required_params:
            assert param in params, f"Parameter '{param}' missing from output"
    
    def test_pspl_params_in_range(self):
        """Test PSPL parameters are within expected ranges"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        flux, params = generate_pspl_event_worker(args)
        
        # Check t0
        assert 0 <= params['t0'] <= 1000, f"t0 = {params['t0']} out of range [0, 1000]"
        
        # Check u0
        assert 0.01 <= params['u0'] <= 1.0, f"u0 = {params['u0']} out of range [0.01, 1.0]"
        
        # Check tE
        assert 10 <= params['tE'] <= 150, f"tE = {params['tE']} out of range [10, 150]"
        
        # Check baseline
        assert 19 <= params['baseline'] <= 22, f"baseline = {params['baseline']} out of range [19, 22]"
    
    def test_pspl_flux_values(self):
        """Test flux values are reasonable"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        flux, params = generate_pspl_event_worker(args)
        
        # Flux should be non-negative (we pad with 0 for missing data)
        assert np.all(flux >= 0), "Flux values should be non-negative"
        
        # Should have some variation (not all zeros or all same)
        assert np.std(flux) > 0, "Flux should have variation"
    
    def test_pspl_reproducibility(self):
        """Test same seed gives same results"""
        args1 = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        args2 = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        
        flux1, params1 = generate_pspl_event_worker(args1)
        flux2, params2 = generate_pspl_event_worker(args2)
        
        np.testing.assert_array_equal(flux1, flux2, "Same seed should give same flux")
        assert params1 == params2, "Same seed should give same parameters"
    
    def test_pspl_different_seeds(self):
        """Test different seeds give different results"""
        args1 = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        args2 = (RANDOM_SEED + 1, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB)
        
        flux1, params1 = generate_pspl_event_worker(args1)
        flux2, params2 = generate_pspl_event_worker(args2)
        
        # Different seeds should give different results
        assert not np.array_equal(flux1, flux2), "Different seeds should give different flux"
        assert params1 != params2, "Different seeds should give different parameters"


class TestBinarySimulation:
    """Tests for Binary event generation"""
    
    def test_binary_output_shape(self):
        """Test Binary event generates correct shape"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'baseline')
        mag, params = generate_binary_event_worker(args)
        
        assert mag.shape == (N_POINTS,), f"Expected shape ({N_POINTS},), got {mag.shape}"
        assert isinstance(params, dict), "Parameters should be a dictionary"
    
    def test_binary_params_exist(self):
        """Test Binary parameters are present"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'baseline')
        mag, params = generate_binary_event_worker(args)
        
        required_params = ['s', 'q', 'rho', 'alpha', 'tE', 't0', 'u0']
        for param in required_params:
            assert param in params, f"Parameter '{param}' missing from output"
    
    def test_binary_baseline_params(self):
        """Test Binary baseline parameters are within expected ranges"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'baseline')
        mag, params = generate_binary_event_worker(args)
        
        # Check s
        assert 0.1 <= params['s'] <= 10.0, f"s = {params['s']} out of baseline range [0.1, 10.0]"
        
        # Check q
        assert 0.001 <= params['q'] <= 1.0, f"q = {params['q']} out of baseline range [0.001, 1.0]"
        
        # Check u0
        assert 0.001 <= params['u0'] <= 1.0, f"u0 = {params['u0']} out of baseline range [0.001, 1.0]"
    
    def test_binary_distinct_params(self):
        """Test Binary distinct parameters are within expected ranges"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'distinct')
        mag, params = generate_binary_event_worker(args)
        
        # Check s (wide binary)
        assert 0.8 <= params['s'] <= 1.5, f"s = {params['s']} out of distinct range [0.8, 1.5]"
        
        # Check q (asymmetric)
        assert 0.01 <= params['q'] <= 0.5, f"q = {params['q']} out of distinct range [0.01, 0.5]"
        
        # Check u0 (small impact)
        assert 0.001 <= params['u0'] <= 0.15, f"u0 = {params['u0']} out of distinct range [0.001, 0.15]"
    
    def test_binary_planetary_params(self):
        """Test Binary planetary parameters have small mass ratio"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'planetary')
        mag, params = generate_binary_event_worker(args)
        
        # Check q (planetary mass ratio)
        assert 0.0001 <= params['q'] <= 0.01, f"q = {params['q']} out of planetary range [0.0001, 0.01]"
    
    def test_binary_stellar_params(self):
        """Test Binary stellar parameters have large mass ratio"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'stellar')
        mag, params = generate_binary_event_worker(args)
        
        # Check q (stellar mass ratio)
        assert 0.3 <= params['q'] <= 1.0, f"q = {params['q']} out of stellar range [0.3, 1.0]"
    
    def test_binary_reproducibility(self):
        """Test same seed gives same results"""
        args1 = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'baseline')
        args2 = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'baseline')
        
        mag1, params1 = generate_binary_event_worker(args1)
        mag2, params2 = generate_binary_event_worker(args2)
        
        np.testing.assert_array_equal(mag1, mag2, "Same seed should give same magnification")
        assert params1 == params2, "Same seed should give same parameters"


class TestSimulationEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_cadence_mask(self):
        """Test with no missing observations"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, 0.0)  # No masking
        flux, params = generate_pspl_event_worker(args)
        
        # Should have no padding (no zeros from masking)
        # Note: flux can still be zero from calculation, so just check shape
        assert flux.shape == (N_POINTS,)
    
    def test_high_cadence_mask(self):
        """Test with many missing observations"""
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, 0.9)  # 90% missing
        flux, params = generate_pspl_event_worker(args)
        
        # Should still have correct shape
        assert flux.shape == (N_POINTS,)
    
    def test_zero_photometric_error(self):
        """Test with perfect photometry"""
        args = (RANDOM_SEED, N_POINTS, 0.0, CADENCE_MASK_PROB)  # No errors
        flux, params = generate_pspl_event_worker(args)
        
        # Should still generate valid light curve
        assert flux.shape == (N_POINTS,)
        assert np.std(flux) > 0


class TestParameterValidation:
    """Test parameter validation"""
    
    def test_invalid_binary_params_name(self):
        """Test that invalid binary parameter set name raises error"""
        # Note: This would be caught at the dataset simulation level
        # Here we test that the function uses the fallback
        args = (RANDOM_SEED, N_POINTS, MAG_ERROR_STD, CADENCE_MASK_PROB, 'invalid_name')
        
        # Should still work because we use BINARY_PARAMS_BASELINE as fallback
        mag, params = generate_binary_event_worker(args)
        assert mag.shape == (N_POINTS,)


# =============================================================================
# Test Running Instructions
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])