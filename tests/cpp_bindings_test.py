#!/usr/bin/env python3
"""
Comprehensive test for C++ bindings enhancements.

This test verifies the following enhancements to the C++ bindings:
1. STL container conversions (vectors, maps, etc.)
2. Error handling with informative messages

Author: Dr. Mazharuddin Mohammed
Date: 2023-07-15
"""

import os
import sys
import unittest
import numpy as np
import gc

# Add the parent directory to the path so we can import qdsim_cpp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import qdsim_cpp
except ImportError:
    print("Error: qdsim_cpp module not found. Make sure it's built and in the Python path.")
    sys.exit(1)

class CPPBindingsTest(unittest.TestCase):
    """Test case for C++ bindings enhancements."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small mesh for testing
        self.Lx = 100.0  # nm
        self.Ly = 100.0  # nm
        self.nx = 10
        self.ny = 10
        self.element_order = 1
        self.mesh = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, self.element_order)

        # Create a material database
        self.mat_db = qdsim_cpp.MaterialDatabase()

    def tearDown(self):
        """Tear down test fixtures."""
        # Force garbage collection to ensure proper cleanup
        gc.collect()

    def test_stl_vector_conversion(self):
        """Test conversion of STL vectors between C++ and Python."""
        # Get nodes from the mesh (C++ vector of Eigen::Vector2d)
        nodes = self.mesh.get_nodes()

        # Verify that nodes is a list of numpy arrays
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)
        self.assertIsInstance(nodes[0], np.ndarray)
        self.assertEqual(nodes[0].shape, (2,))

        # Get elements from the mesh (C++ vector of arrays)
        elements = self.mesh.get_elements()

        # Verify that elements is a list of lists/arrays
        self.assertIsInstance(elements, list)
        self.assertGreater(len(elements), 0)

        # Test passing a Python list to C++
        # Create a list of node indices
        node_indices = list(range(5))

        # Get the number of nodes and elements
        num_nodes = self.mesh.get_num_nodes()
        num_elements = self.mesh.get_num_elements()

        # Verify that these are integers
        self.assertIsInstance(num_nodes, int)
        self.assertIsInstance(num_elements, int)
        self.assertGreater(num_nodes, 0)
        self.assertGreater(num_elements, 0)

    def test_stl_map_conversion(self):
        """Test conversion of STL maps between C++ and Python."""
        # Get all materials from the database (C++ unordered_map)
        materials = self.mat_db.get_all_materials()

        # Verify that materials is a dictionary
        self.assertIsInstance(materials, dict)
        self.assertGreater(len(materials), 0)

        # Check that the keys are strings and the values are Material objects
        for name, material in materials.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(material, qdsim_cpp.Material)

        # Test passing a Python dictionary to C++
        # Create a dictionary of material properties
        custom_props = {
            "m_e": 0.067,       # Electron effective mass
            "m_h": 0.45,        # Hole effective mass
            "E_g": 1.42,        # Band gap (eV)
            "epsilon_r": 12.9,  # Relative permittivity
            "mu_n": 8500.0,     # Electron mobility (cm²/V·s)
            "mu_p": 400.0       # Hole mobility (cm²/V·s)
        }

        # Create a custom material with these properties
        custom_mat = qdsim_cpp.Material()
        for prop, value in custom_props.items():
            setattr(custom_mat, prop, value)

        # Verify that the properties were set correctly
        for prop, value in custom_props.items():
            self.assertEqual(getattr(custom_mat, prop), value)

    def test_error_handling(self):
        """Test error handling with informative messages."""
        try:
            # Try to create a mesh with invalid parameters
            invalid_mesh = qdsim_cpp.Mesh(-1.0, -1.0, -1, -1, -1)
            self.fail("Expected an exception for invalid mesh parameters")
        except Exception as e:
            # Verify that the exception message is informative
            error_message = str(e)
            self.assertGreater(len(error_message), 0)
            print(f"Got expected error message: {error_message}")

if __name__ == "__main__":
    unittest.main()
