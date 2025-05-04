#!/usr/bin/env python3
"""
Example script for running a square potential simulation with P2 elements.

This script demonstrates how to set up and run a quantum dot simulation with
a square potential well and quadratic (P2) finite elements, which provide a
good balance between accuracy and computational efficiency.

Author: Dr. Mazharuddin Mohammed
"""

from qdsim import Simulator, Config

config = Config()
config.potential_type = "square"
config.element_order = 2
sim = Simulator(config)
eigenvalues, eigenvectors = sim.run(num_eigenvalues=5)
print("Eigenvalues (eV):", eigenvalues / 1.602e-19)