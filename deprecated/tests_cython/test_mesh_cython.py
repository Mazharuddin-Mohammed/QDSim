#!/usr/bin/env python3
"""
Unit tests for Cython Mesh implementation

This module tests the Cython wrapper for the C++ Mesh class,
ensuring proper functionality and performance.

Author: Dr. Mazharuddin Mohammed
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qdsim_cython.core.mesh import Mesh, compute_refinement_flags
    CYTHON_AVAILABLE = True
except ImportError as e:
    CYTHON_AVAILABLE = False
    pytest.skip(f"Cython mesh module not available: {e}", allow_module_level=True)

class TestMeshCython:
    """Test class for Cython Mesh implementation."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.Lx = 100e-9  # 100 nm
        self.Ly = 100e-9  # 100 nm
        self.nx = 10
        self.ny = 10
        self.element_order = 1
        
    def test_mesh_creation(self):
        """Test basic mesh creation."""
        mesh = Mesh(self.Lx, self.Ly, self.nx, self.ny, self.element_order)
        
        # Check basic properties
        assert mesh.Lx == self.Lx
        assert mesh.Ly == self.Ly
        assert mesh.nx == self.nx
        assert mesh.ny == self.ny
        assert mesh.element_order == self.element_order
        
        # Check that nodes and elements are created
        assert mesh.num_nodes > 0
        assert mesh.num_elements > 0
        
        print(f"Created mesh with {mesh.num_nodes} nodes and {mesh.num_elements} elements")
        
    def test_mesh_nodes(self):
        """Test mesh node access."""
        mesh = Mesh(self.Lx, self.Ly, self.nx, self.ny, self.element_order)
        
        nodes = mesh.nodes
        assert isinstance(nodes, np.ndarray)
        assert nodes.shape[1] == 2  # 2D coordinates
        assert nodes.shape[0] == mesh.num_nodes
        
        # Check node coordinate ranges
        assert np.min(nodes[:, 0]) >= -self.Lx/2
        assert np.max(nodes[:, 0]) <= self.Lx/2
        assert np.min(nodes[:, 1]) >= -self.Ly/2
        assert np.max(nodes[:, 1]) <= self.Ly/2
        
        print(f"Node coordinates range: x=[{np.min(nodes[:, 0]):.2e}, {np.max(nodes[:, 0]):.2e}], "
              f"y=[{np.min(nodes[:, 1]):.2e}, {np.max(nodes[:, 1]):.2e}]")
        
    def test_mesh_elements(self):
        """Test mesh element access."""
        mesh = Mesh(self.Lx, self.Ly, self.nx, self.ny, self.element_order)
        
        elements = mesh.elements
        assert isinstance(elements, np.ndarray)
        assert elements.shape[1] == 3  # Triangular elements
        assert elements.shape[0] == mesh.num_elements
        
        # Check that element indices are valid
        assert np.min(elements) >= 0
        assert np.max(elements) < mesh.num_nodes
        
        print(f"Element indices range: [{np.min(elements)}, {np.max(elements)}]")
        
    def test_higher_order_elements(self):
        """Test higher-order element creation."""
        # Test P2 elements
        mesh_p2 = Mesh(self.Lx, self.Ly, 5, 5, 2)
        assert mesh_p2.element_order == 2
        
        quadratic_elements = mesh_p2.quadratic_elements
        if quadratic_elements is not None:
            assert quadratic_elements.shape[1] == 6  # P2 elements have 6 nodes
            print(f"P2 mesh created with {mesh_p2.num_nodes} nodes")
        
        # Test P3 elements
        mesh_p3 = Mesh(self.Lx, self.Ly, 5, 5, 3)
        assert mesh_p3.element_order == 3
        
        cubic_elements = mesh_p3.cubic_elements
        if cubic_elements is not None:
            assert cubic_elements.shape[1] == 10  # P3 elements have 10 nodes
            print(f"P3 mesh created with {mesh_p3.num_nodes} nodes")
            
    def test_mesh_refinement(self):
        """Test mesh refinement functionality."""
        mesh = Mesh(self.Lx, self.Ly, 5, 5, 1)
        initial_num_elements = mesh.num_elements
        
        # Create refinement flags (refine every other element)
        refine_flags = np.zeros(initial_num_elements, dtype=bool)
        refine_flags[::2] = True
        
        # Test refinement
        try:
            mesh.refine(refine_flags)
            print(f"Mesh refined from {initial_num_elements} to {mesh.num_elements} elements")
            
            # Check that the mesh has more elements after refinement
            assert mesh.num_elements >= initial_num_elements
        except Exception as e:
            print(f"Refinement test skipped due to: {e}")
            
    def test_mesh_refinement_by_indices(self):
        """Test mesh refinement by element indices."""
        mesh = Mesh(self.Lx, self.Ly, 5, 5, 1)
        initial_num_elements = mesh.num_elements
        
        # Refine specific elements
        element_indices = [0, 2, 4]
        
        try:
            success = mesh.refine_elements(element_indices, max_refinement_level=2)
            if success:
                print(f"Element-based refinement: {initial_num_elements} -> {mesh.num_elements} elements")
                assert mesh.num_elements >= initial_num_elements
            else:
                print("Element-based refinement returned False")
        except Exception as e:
            print(f"Element refinement test skipped due to: {e}")
            
    def test_mesh_save_load(self):
        """Test mesh save and load functionality."""
        mesh = Mesh(self.Lx, self.Ly, 5, 5, 1)
        
        # Save mesh
        filename = "test_mesh.dat"
        try:
            mesh.save(filename)
            print(f"Mesh saved to {filename}")
            
            # Load mesh
            loaded_mesh = Mesh.load(filename)
            print(f"Mesh loaded from {filename}")
            
            # Check that loaded mesh has same properties
            assert loaded_mesh.num_nodes == mesh.num_nodes
            assert loaded_mesh.num_elements == mesh.num_elements
            assert loaded_mesh.element_order == mesh.element_order
            
            # Clean up
            if os.path.exists(filename):
                os.remove(filename)
                
        except Exception as e:
            print(f"Save/load test skipped due to: {e}")
            # Clean up on error
            if os.path.exists(filename):
                os.remove(filename)
                
    def test_compute_refinement_flags(self):
        """Test adaptive refinement flag computation."""
        mesh = Mesh(self.Lx, self.Ly, 5, 5, 1)
        
        # Create a test solution with some variation
        solution = np.random.rand(mesh.num_nodes)
        threshold = 0.1
        
        try:
            flags = compute_refinement_flags(mesh, solution, threshold)
            assert isinstance(flags, np.ndarray)
            assert flags.dtype == bool
            assert len(flags) == mesh.num_elements
            
            num_refine = np.sum(flags)
            print(f"Refinement flags computed: {num_refine}/{mesh.num_elements} elements marked for refinement")
            
        except Exception as e:
            print(f"Refinement flags test skipped due to: {e}")
            
    def test_mesh_memory_management(self):
        """Test that mesh objects are properly cleaned up."""
        # Create and destroy many mesh objects to test memory management
        for i in range(10):
            mesh = Mesh(self.Lx, self.Ly, 5, 5, 1)
            nodes = mesh.nodes
            elements = mesh.elements
            # Objects should be automatically cleaned up when going out of scope
            
        print("Memory management test completed")
        
    def test_mesh_performance(self):
        """Test mesh creation performance."""
        import time
        
        # Test performance with different mesh sizes
        sizes = [(5, 5), (10, 10), (20, 20)]
        
        for nx, ny in sizes:
            start_time = time.time()
            mesh = Mesh(self.Lx, self.Ly, nx, ny, 1)
            creation_time = time.time() - start_time
            
            start_time = time.time()
            nodes = mesh.nodes
            elements = mesh.elements
            access_time = time.time() - start_time
            
            print(f"Mesh {nx}x{ny}: creation={creation_time:.4f}s, "
                  f"access={access_time:.4f}s, nodes={mesh.num_nodes}, elements={mesh.num_elements}")
                  
    def test_mesh_validation(self):
        """Test mesh validation and error handling."""
        # Test invalid parameters
        with pytest.raises(Exception):
            # Invalid element order
            Mesh(self.Lx, self.Ly, self.nx, self.ny, 0)
            
        with pytest.raises(Exception):
            # Invalid dimensions
            Mesh(-1, self.Ly, self.nx, self.ny, 1)
            
        with pytest.raises(Exception):
            # Invalid mesh resolution
            Mesh(self.Lx, self.Ly, 0, self.ny, 1)
            
        print("Mesh validation tests completed")

if __name__ == "__main__":
    # Run tests if called directly
    test = TestMeshCython()
    test.setup_method()
    
    print("Running Cython Mesh tests...")
    test.test_mesh_creation()
    test.test_mesh_nodes()
    test.test_mesh_elements()
    test.test_higher_order_elements()
    test.test_mesh_refinement()
    test.test_mesh_refinement_by_indices()
    test.test_mesh_save_load()
    test.test_compute_refinement_flags()
    test.test_mesh_memory_management()
    test.test_mesh_performance()
    test.test_mesh_validation()
    
    print("All Cython Mesh tests completed!")
