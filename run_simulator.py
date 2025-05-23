from qdsim import Simulator, Config
from qdsim.visualization import plot_energy_shift
import numpy as np
import matplotlib.pyplot as plt

def main():
    config = Config()
    config.element_order = 3
    config.qd_material = "InAs"
    config.matrix_material = "GaAs"
    V_r_values = np.linspace(0, 2, 10)  # Reverse bias from 0 to 2 V
    energies = []
    linewidths = []

    for V_r in V_r_values:
        config.V_r = V_r
        sim = Simulator(config)
        eigenvalues, _ = sim.run(num_eigenvalues=1)
        energies.append(np.real(eigenvalues[0]) / 1.602e-19)  # eV
        linewidths.append(-2 * np.imag(eigenvalues[0]) / 1.602e-19)  # eV

    fig, ax = plt.subplots()
    plot_energy_shift(ax, V_r_values, energies, linewidths)
    plt.savefig('energy_shift.png')
    print("Plot saved to energy_shift.png")
    # Uncomment to show the plot interactively
    # plt.show()

if __name__ == "__main__":
    main()