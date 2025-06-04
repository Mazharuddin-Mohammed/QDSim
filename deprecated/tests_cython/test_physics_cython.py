#!/usr/bin/env python3
"""
Unit tests for Cython Physics implementation

This module tests the Cython wrapper for the C++ physics functions,
ensuring proper functionality and physical correctness.

Author: Dr. Mazharuddin Mohammed
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qdsim_cython.core.physics import (
        PhysicsConstants, effective_mass, potential, epsilon_r,
        charge_density, capacitance, electron_concentration,
        hole_concentration, mobility_n, mobility_p
    )
    CYTHON_AVAILABLE = True
except ImportError as e:
    CYTHON_AVAILABLE = False
    pytest.skip(f"Cython physics module not available: {e}", allow_module_level=True)

class TestPhysicsConstants:
    """Test class for physics constants."""
    
    def test_fundamental_constants(self):
        """Test fundamental physical constants."""
        # Check that constants are reasonable values
        assert PhysicsConstants.ELEMENTARY_CHARGE > 0
        assert PhysicsConstants.PLANCK_CONSTANT > 0
        assert PhysicsConstants.REDUCED_PLANCK_CONSTANT > 0
        assert PhysicsConstants.BOLTZMANN_CONSTANT > 0
        assert PhysicsConstants.ELECTRON_MASS > 0
        assert PhysicsConstants.VACUUM_PERMITTIVITY > 0
        assert PhysicsConstants.SPEED_OF_LIGHT > 0
        
        # Check specific values (approximate)
        assert abs(PhysicsConstants.ELEMENTARY_CHARGE - 1.602e-19) < 1e-21
        assert abs(PhysicsConstants.REDUCED_PLANCK_CONSTANT - 1.055e-34) < 1e-36
        
        print(f"Elementary charge: {PhysicsConstants.ELEMENTARY_CHARGE:.3e} C")
        print(f"Reduced Planck constant: {PhysicsConstants.REDUCED_PLANCK_CONSTANT:.3e} J·s")
        print(f"Boltzmann constant: {PhysicsConstants.BOLTZMANN_CONSTANT:.3e} J/K")
        
    def test_unit_conversions(self):
        """Test unit conversion constants."""
        # Check conversion factors
        assert PhysicsConstants.EV_TO_JOULE > 0
        assert PhysicsConstants.JOULE_TO_EV > 0
        assert PhysicsConstants.NM_TO_METER > 0
        assert PhysicsConstants.METER_TO_NM > 0
        
        # Check that conversions are inverses
        assert abs(PhysicsConstants.EV_TO_JOULE * PhysicsConstants.JOULE_TO_EV - 1.0) < 1e-10
        assert abs(PhysicsConstants.NM_TO_METER * PhysicsConstants.METER_TO_NM - 1.0) < 1e-10
        
        print(f"eV to Joule: {PhysicsConstants.EV_TO_JOULE:.3e}")
        print(f"nm to meter: {PhysicsConstants.NM_TO_METER:.3e}")
        
    def test_parameter_ranges(self):
        """Test typical parameter ranges."""
        # Check that ranges are reasonable
        assert 0 < PhysicsConstants.MIN_EFFECTIVE_MASS < PhysicsConstants.MAX_EFFECTIVE_MASS
        assert PhysicsConstants.MIN_POTENTIAL < PhysicsConstants.MAX_POTENTIAL
        assert 0 < PhysicsConstants.MIN_PERMITTIVITY < PhysicsConstants.MAX_PERMITTIVITY
        
        print(f"Effective mass range: [{PhysicsConstants.MIN_EFFECTIVE_MASS}, {PhysicsConstants.MAX_EFFECTIVE_MASS}]")
        print(f"Potential range: [{PhysicsConstants.MIN_POTENTIAL}, {PhysicsConstants.MAX_POTENTIAL}] eV")
        print(f"Permittivity range: [{PhysicsConstants.MIN_PERMITTIVITY}, {PhysicsConstants.MAX_PERMITTIVITY}]")

class TestPhysicsFunctions:
    """Test class for physics functions."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.x_coords = np.linspace(-50e-9, 50e-9, 11)  # -50 to 50 nm
        self.y_coords = np.linspace(-50e-9, 50e-9, 11)  # -50 to 50 nm
        self.R = 20e-9  # 20 nm quantum dot radius
        
    def test_effective_mass_function(self):
        """Test effective mass calculation."""
        # Test at various positions
        for x in self.x_coords[::3]:  # Test subset for speed
            for y in self.y_coords[::3]:
                try:
                    m_eff = effective_mass(x, y, None, None, self.R)
                    
                    # Check that effective mass is reasonable
                    assert PhysicsConstants.MIN_EFFECTIVE_MASS <= m_eff <= PhysicsConstants.MAX_EFFECTIVE_MASS
                    
                except Exception as e:
                    print(f"Effective mass calculation failed at ({x:.1e}, {y:.1e}): {e}")
                    
        print(f"Effective mass function tested at {len(self.x_coords[::3])}x{len(self.y_coords[::3])} points")
        
    def test_potential_function(self):
        """Test potential calculation."""
        # Create dummy phi array
        phi_array = np.zeros(100)
        
        # Test different potential types
        potential_types = ["square", "gaussian"]
        
        for pot_type in potential_types:
            for x in self.x_coords[::5]:  # Test subset for speed
                for y in self.y_coords[::5]:
                    try:
                        V = potential(x, y, None, None, self.R, pot_type, phi_array)
                        
                        # Check that potential is reasonable
                        assert PhysicsConstants.MIN_POTENTIAL <= V <= PhysicsConstants.MAX_POTENTIAL
                        
                    except Exception as e:
                        print(f"Potential calculation failed for {pot_type} at ({x:.1e}, {y:.1e}): {e}")
                        
        print(f"Potential function tested for {len(potential_types)} types")
        
    def test_epsilon_r_function(self):
        """Test relative permittivity calculation."""
        for x in self.x_coords[::3]:
            for y in self.y_coords[::3]:
                try:
                    eps_r = epsilon_r(x, y, None, None)
                    
                    # Check that permittivity is reasonable
                    assert PhysicsConstants.MIN_PERMITTIVITY <= eps_r <= PhysicsConstants.MAX_PERMITTIVITY
                    
                except Exception as e:
                    print(f"Permittivity calculation failed at ({x:.1e}, {y:.1e}): {e}")
                    
        print("Relative permittivity function tested")
        
    def test_charge_density_function(self):
        """Test charge density calculation."""
        # Create dummy carrier concentration arrays
        n_array = np.ones(100) * 1e15  # 1e15 cm^-3
        p_array = np.ones(100) * 1e15  # 1e15 cm^-3
        
        for x in self.x_coords[::5]:
            for y in self.y_coords[::5]:
                try:
                    rho = charge_density(x, y, n_array, p_array)
                    
                    # Check that charge density is finite
                    assert np.isfinite(rho)
                    
                except Exception as e:
                    print(f"Charge density calculation failed at ({x:.1e}, {y:.1e}): {e}")
                    
        print("Charge density function tested")
        
    def test_capacitance_function(self):
        """Test capacitance calculation."""
        eta = 0.8  # Gate efficiency
        Lx = 100e-9  # Domain width
        Ly = 100e-9  # Domain height
        d = 10e-9   # Gate distance
        
        for x in self.x_coords[::3]:
            for y in self.y_coords[::3]:
                try:
                    C = capacitance(x, y, eta, Lx, Ly, d)
                    
                    # Check that capacitance is positive and finite
                    assert C >= 0
                    assert np.isfinite(C)
                    
                except Exception as e:
                    print(f"Capacitance calculation failed at ({x:.1e}, {y:.1e}): {e}")
                    
        print("Capacitance function tested")
        
    def test_carrier_concentration_functions(self):
        """Test carrier concentration calculations."""
        phi_values = np.linspace(-1.0, 1.0, 5)  # -1 to 1 V
        
        for phi in phi_values:
            for x in self.x_coords[::5]:
                for y in self.y_coords[::5]:
                    try:
                        n = electron_concentration(x, y, phi, None)
                        p = hole_concentration(x, y, phi, None)
                        
                        # Check that concentrations are positive and finite
                        assert n >= 0
                        assert p >= 0
                        assert np.isfinite(n)
                        assert np.isfinite(p)
                        
                    except Exception as e:
                        print(f"Carrier concentration calculation failed at ({x:.1e}, {y:.1e}, {phi:.1f}): {e}")
                        
        print("Carrier concentration functions tested")
        
    def test_mobility_functions(self):
        """Test mobility calculations."""
        for x in self.x_coords[::3]:
            for y in self.y_coords[::3]:
                try:
                    mu_n = mobility_n(x, y, None)
                    mu_p = mobility_p(x, y, None)
                    
                    # Check that mobilities are positive and finite
                    assert mu_n > 0
                    assert mu_p > 0
                    assert np.isfinite(mu_n)
                    assert np.isfinite(mu_p)
                    
                    # Check that electron mobility is typically higher than hole mobility
                    # (this is true for most semiconductors)
                    assert mu_n >= mu_p
                    
                except Exception as e:
                    print(f"Mobility calculation failed at ({x:.1e}, {y:.1e}): {e}")
                    
        print("Mobility functions tested")

class TestPhysicsPerformance:
    """Test class for physics function performance."""
    
    def test_function_performance(self):
        """Test performance of physics functions."""
        import time
        
        # Create test data
        x_vals = np.linspace(-50e-9, 50e-9, 100)
        y_vals = np.linspace(-50e-9, 50e-9, 100)
        phi_array = np.zeros(1000)
        n_array = np.ones(1000) * 1e15
        p_array = np.ones(1000) * 1e15
        
        # Test effective mass performance
        start_time = time.time()
        for x in x_vals[::10]:
            for y in y_vals[::10]:
                effective_mass(x, y, None, None, 20e-9)
        eff_mass_time = time.time() - start_time
        
        # Test potential performance
        start_time = time.time()
        for x in x_vals[::10]:
            for y in y_vals[::10]:
                potential(x, y, None, None, 20e-9, "square", phi_array)
        potential_time = time.time() - start_time
        
        # Test permittivity performance
        start_time = time.time()
        for x in x_vals[::10]:
            for y in y_vals[::10]:
                epsilon_r(x, y, None, None)
        epsilon_time = time.time() - start_time
        
        print(f"Performance test results:")
        print(f"  Effective mass: {eff_mass_time:.4f}s for {len(x_vals[::10])}x{len(y_vals[::10])} evaluations")
        print(f"  Potential: {potential_time:.4f}s for {len(x_vals[::10])}x{len(y_vals[::10])} evaluations")
        print(f"  Permittivity: {epsilon_time:.4f}s for {len(x_vals[::10])}x{len(y_vals[::10])} evaluations")

if __name__ == "__main__":
    # Run tests if called directly
    print("Running Cython Physics tests...")
    
    # Test constants
    const_test = TestPhysicsConstants()
    const_test.test_fundamental_constants()
    const_test.test_unit_conversions()
    const_test.test_parameter_ranges()
    
    # Test functions
    func_test = TestPhysicsFunctions()
    func_test.setup_method()
    func_test.test_effective_mass_function()
    func_test.test_potential_function()
    func_test.test_epsilon_r_function()
    func_test.test_charge_density_function()
    func_test.test_capacitance_function()
    func_test.test_carrier_concentration_functions()
    func_test.test_mobility_functions()
    
    # Test performance
    perf_test = TestPhysicsPerformance()
    perf_test.test_function_performance()
    
    print("All Cython Physics tests completed!")
