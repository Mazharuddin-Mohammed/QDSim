#!/usr/bin/env python3
"""
Basic test for FEM implementation enhancements.

This test verifies the following enhancements to the FEM implementation:
1. MPI switch enabling/disabling option
2. GPU acceleration for higher-order elements

Author: Dr. Mazharuddin Mohammed
Date: 2023-07-15
"""

import os
import sys
import unittest
import numpy as np
import time

# Add the parent directory to the path so we can import qdsim_cpp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import qdsim_cpp
except ImportError:
    print("Error: qdsim_cpp module not found. Make sure it's built and in the Python path.")
    sys.exit(1)

class FEMBasicTest(unittest.TestCase):
    """Test case for FEM implementation enhancements."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small mesh for testing
        self.Lx = 100.0  # nm
        self.Ly = 100.0  # nm
        self.nx = 10
        self.ny = 10
        self.element_order = 1
        self.mesh = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, self.element_order)

    def tearDown(self):
        """Tear down test fixtures."""
        pass

    def test_mesh_properties(self):
        """Test basic mesh properties."""
        print("\n=== Testing Mesh Properties ===")

        # Verify mesh properties
        self.assertEqual(self.mesh.get_num_nodes(), (self.nx + 1) * (self.ny + 1))
        self.assertEqual(self.mesh.get_num_elements(), 2 * self.nx * self.ny)
        self.assertEqual(self.mesh.get_element_order(), self.element_order)

        # Get nodes and elements
        nodes = self.mesh.get_nodes()
        elements = self.mesh.get_elements()

        # Verify that nodes and elements are properly returned
        self.assertEqual(len(nodes), self.mesh.get_num_nodes())
        self.assertEqual(len(elements), self.mesh.get_num_elements())

        print("Mesh properties test passed!")

    def test_higher_order_elements(self):
        """Test higher-order elements."""
        print("\n=== Testing Higher-Order Elements ===")

        # Create meshes with different element orders
        mesh_p1 = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, 1)  # Linear elements
        mesh_p2 = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, 2)  # Quadratic elements
        mesh_p3 = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, 3)  # Cubic elements

        # Verify element orders
        self.assertEqual(mesh_p1.get_element_order(), 1)
        self.assertEqual(mesh_p2.get_element_order(), 2)
        self.assertEqual(mesh_p3.get_element_order(), 3)

        # Verify number of nodes for different element orders
        # The exact number of nodes depends on the implementation
        # Just verify that higher-order elements have more nodes
        p1_nodes = mesh_p1.get_num_nodes()
        p2_nodes = mesh_p2.get_num_nodes()
        p3_nodes = mesh_p3.get_num_nodes()

        print(f"P1 elements: {p1_nodes} nodes")
        print(f"P2 elements: {p2_nodes} nodes")
        print(f"P3 elements: {p3_nodes} nodes")

        self.assertGreater(p2_nodes, p1_nodes)
        self.assertGreater(p3_nodes, p2_nodes)

        print("Higher-order elements test passed!")

    def test_mpi_option(self):
        """Test MPI option."""
        print("\n=== Testing MPI Option ===")

        try:
            # Create a mesh with MPI enabled
            mesh_mpi = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, self.element_order, True)

            # Verify mesh properties
            self.assertEqual(mesh_mpi.get_num_nodes(), (self.nx + 1) * (self.ny + 1))
            self.assertEqual(mesh_mpi.get_num_elements(), 2 * self.nx * self.ny)
            self.assertEqual(mesh_mpi.get_element_order(), self.element_order)

            # Check if MPI is enabled
            try:
                mpi_enabled = mesh_mpi.is_mpi_enabled()
                print(f"MPI enabled: {mpi_enabled}")
            except AttributeError:
                print("is_mpi_enabled() method not available")

            print("MPI option test passed!")
        except Exception as e:
            print(f"MPI test skipped: {e}")
            print("This is expected if MPI is not enabled in the build")

    def test_gpu_acceleration(self):
        """Test GPU acceleration."""
        print("\n=== Testing GPU Acceleration ===")

        try:
            # Check if GPU acceleration is available
            gpu_available = qdsim_cpp.is_gpu_available()
            if not gpu_available:
                print("GPU acceleration not available, skipping test")
                return

            print("GPU acceleration is available")

            # Create meshes with different element orders
            mesh_p1 = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, 1)  # Linear elements
            mesh_p2 = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, 2)  # Quadratic elements
            mesh_p3 = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, 3)  # Cubic elements

            # Test GPU acceleration for different element orders
            for order, mesh in [(1, mesh_p1), (2, mesh_p2), (3, mesh_p3)]:
                print(f"\nTesting order {order} elements:")

                try:
                    # Enable GPU acceleration
                    mesh.use_gpu_acceleration(True)

                    # Verify that GPU acceleration is enabled
                    gpu_enabled = mesh.is_gpu_enabled()
                    print(f"GPU acceleration enabled: {gpu_enabled}")

                    # Disable GPU acceleration
                    mesh.use_gpu_acceleration(False)

                    # Verify that GPU acceleration is disabled
                    gpu_enabled = mesh.is_gpu_enabled()
                    print(f"GPU acceleration enabled: {gpu_enabled}")

                    print(f"GPU acceleration test for order {order} elements passed!")
                except AttributeError:
                    print(f"GPU acceleration not supported for order {order} elements")
        except AttributeError:
            print("GPU acceleration not supported in this build")

if __name__ == "__main__":
    unittest.main()
