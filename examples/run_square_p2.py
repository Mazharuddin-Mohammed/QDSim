# run_square_p2.py
from qdsim import Simulator, Config

config = Config()
config.potential_type = "square"
config.element_order = 2
sim = Simulator(config)
eigenvalues, eigenvectors = sim.run(num_eigenvalues=5)
print("Eigenvalues (eV):", eigenvalues / 1.602e-19)