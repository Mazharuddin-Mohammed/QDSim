#!/usr/bin/env python3
"""
Example script for visualizing error estimators in quantum dot simulations.

This script demonstrates how to compute and visualize the error estimator field
for a quantum dot simulation, which is used to guide adaptive mesh refinement
by identifying regions where the solution has high gradients or discontinuities.

Author: Dr. Mazharuddin Mohammed
"""

from qdsim import Simulator, Config
from qdsim.visualization import plot_error_estimator
import matplotlib.pyplot as plt

config = Config()
sim = Simulator(config)
_, eigenvectors = sim.run(num_eigenvalues=1)
fig, ax = plt.subplots()
plot_error_estimator(ax, sim.mesh, eigenvectors[0], config.adaptive_threshold)
plt.show()