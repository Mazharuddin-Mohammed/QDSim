#!/usr/bin/env python3
"""
Example script for plotting error estimators with P3 elements.

This script demonstrates how to visualize the error estimator field for
a quantum dot simulation with cubic (P3) finite elements, which is used
to guide adaptive mesh refinement.

Author: Dr. Mazharuddin Mohammed
"""

from qdsim import Simulator, Config
from qdsim.visualization import plot_error_estimator
import matplotlib.pyplot as plt

config = Config()
config.element_order = 3
sim = Simulator(config)
_, eigenvectors = sim.run(num_eigenvalues=1)
fig, ax = plt.subplots()
plot_error_estimator(ax, sim.mesh, eigenvectors[0], config.adaptive_threshold)
plt.show()