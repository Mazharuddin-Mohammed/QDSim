# run_gaussian_p3.py
from qdsim import Simulator, Config
from qdsim.visualization import plot_wavefunction
import matplotlib.pyplot as plt

config = Config()
config.potential_type = "gaussian"
config.element_order = 3
config.nx = 20  # Coarser initial mesh for P3
config.ny = 20
sim = Simulator(config)
eigenvalues, eigenvectors = sim.run(num_eigenvalues=5, max_refinements=2)

# Visualize first eigenstate
fig, ax = plt.subplots()
plot_wavefunction(ax, sim.mesh, eigenvectors[0])
plt.show()