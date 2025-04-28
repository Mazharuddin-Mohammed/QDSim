import numpy as np
import matplotlib.pyplot as plt

def plot_wavefunction(ax, mesh, eigenvector):
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())
    psi = np.abs(eigenvector) ** 2
    ax.tricontourf(nodes[:, 0], nodes[:, 1], elements, psi, cmap='viridis')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Wavefunction Density')

def plot_electric_field(ax, mesh, poisson_solver):
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())
    E = np.array([poisson_solver.get_electric_field(x, y) for x, y in nodes])
    ax.quiver(nodes[:, 0], nodes[:, 1], E[:, 0], E[:, 1], scale=1e6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Electric Field (V/m)')

def plot_potential(ax, mesh, poisson_solver):
    nodes = np.array(mesh.get_nodes())
    elements = np.array(mesh.get_elements())
    phi = poisson_solver.get_potential()
    ax.tricontourf(nodes[:, 0], nodes[:, 1], elements, phi, cmap='viridis')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Electrostatic Potential (V)')

def plot_energy_shift(ax, V_r_values, energies, linewidths):
    ax.plot(V_r_values, energies, 'o-', label='Energy (eV)')
    ax2 = ax.twinx()
    ax2.plot(V_r_values, linewidths, 's-', color='red', label='Linewidth (eV)')
    ax.set_xlabel('Reverse Bias (V)')
    ax.set_ylabel('Energy (eV)')
    ax2.set_ylabel('Linewidth (eV)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')