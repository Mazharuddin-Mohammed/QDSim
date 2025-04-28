from qdsim import Simulator, Config
from qdsim.visualization import plot_error_estimator
import matplotlib.pyplot as plt

config = Config()
sim = Simulator(config)
_, eigenvectors = sim.run(num_eigenvalues=1)
fig, ax = plt.subplots()
plot_error_estimator(ax, sim.mesh, eigenvectors[0], config.adaptive_threshold)
plt.show()