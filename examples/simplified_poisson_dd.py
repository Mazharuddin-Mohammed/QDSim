"""
Simplified Poisson-Drift-Diffusion solver example.

This script demonstrates a simplified implementation of the Poisson-Drift-Diffusion
solver using the existing QDSim classes.

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Add the frontend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import QDSim
from frontend.qdsim import Mesh, PoissonSolver

class SimplifiedPoissonDriftDiffusionSolver:
    """
    A simplified implementation of the Poisson-Drift-Diffusion solver.
    """

    def __init__(self, mesh, epsilon_r, doping_profile):
        """
        Initialize the solver.

        Args:
            mesh: The mesh to use for the simulation
            epsilon_r: Function that returns the relative permittivity at a given position
            doping_profile: Function that returns the doping profile at a given position
        """
        self.mesh = mesh
        self.epsilon_r = epsilon_r
        self.doping_profile = doping_profile

        # Constants
        self.q = 1.602e-19  # Elementary charge (C)
        self.epsilon_0 = 8.85418782e-14  # Vacuum permittivity (F/cm)
        self.kT = 0.0259  # Thermal voltage at 300K (eV)

        # Initialize vectors
        self.phi = np.zeros(mesh.get_num_nodes())
        self.n = np.zeros(mesh.get_num_nodes())
        self.p = np.zeros(mesh.get_num_nodes())
        self.E_field = [np.zeros(2) for _ in range(mesh.get_num_nodes())]

        # Material properties
        self.N_c = 4.7e17  # Effective density of states in conduction band (cm^-3)
        self.N_v = 7.0e18  # Effective density of states in valence band (cm^-3)
        self.E_g = 1.424  # Band gap (eV)

        # Initialize carrier concentrations
        self._initialize_carrier_concentrations()

    def _initialize_carrier_concentrations(self):
        """
        Initialize carrier concentrations based on doping profile.
        """
        # Calculate intrinsic carrier concentration
        ni = np.sqrt(self.N_c * self.N_v) * np.exp(-self.E_g / (2.0 * self.kT))

        # Initialize carrier concentrations at each node
        for i in range(self.mesh.get_num_nodes()):
            x = self.mesh.get_nodes()[i][0]
            y = self.mesh.get_nodes()[i][1]

            # Get doping at this position
            doping = self.doping_profile(x, y)

            # Initialize carrier concentrations based on doping
            if doping > 0:
                # n-type region
                self.n[i] = doping
                self.p[i] = ni * ni / doping
            elif doping < 0:
                # p-type region
                self.p[i] = -doping
                self.n[i] = ni * ni / self.p[i]
            else:
                # Intrinsic region
                self.n[i] = ni
                self.p[i] = ni

            # Ensure minimum carrier concentrations for numerical stability
            self.n[i] = max(self.n[i], 1e5)
            self.p[i] = max(self.p[i], 1e5)

    def solve(self, V_p, V_n):
        """
        Solve the Poisson equation.

        Args:
            V_p: Voltage applied to the p-contact
            V_n: Voltage applied to the n-contact
        """
        # Define the charge density function
        def rho(x, y, n_vec, p_vec):
            # Find the nearest node
            min_dist = float('inf')
            nearest_node = 0

            for i in range(self.mesh.get_num_nodes()):
                node = self.mesh.get_nodes()[i]
                dist = (node[0] - x)**2 + (node[1] - y)**2

                if dist < min_dist:
                    min_dist = dist
                    nearest_node = i

            # Get doping at this position
            doping = self.doping_profile(x, y)

            # Compute charge density: rho = q * (p - n + N_D - N_A)
            return self.q * (self.p[nearest_node] - self.n[nearest_node] + doping)

        # Create a PoissonSolver with the epsilon_r and rho functions
        poisson_solver = PoissonSolver(self.mesh, self.epsilon_r, rho)

        # Solve the Poisson equation
        print("Solving Poisson equation...")
        start_time = time.time()

        # We need to manually set the boundary conditions
        # Create the stiffness matrix and right-hand side vector
        n_nodes = self.mesh.get_num_nodes()
        A = lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        # Assemble the system
        for e in range(self.mesh.get_num_elements()):
            element = self.mesh.get_elements()[e]
            nodes = [self.mesh.get_nodes()[j] for j in element]

            # Calculate element area
            x1, y1 = nodes[0]
            x2, y2 = nodes[1]
            x3, y3 = nodes[2]

            # Compute the derivatives of the shape functions
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

            dN1_dx = (y2 - y3) / (2.0 * area)
            dN1_dy = (x3 - x2) / (2.0 * area)
            dN2_dx = (y3 - y1) / (2.0 * area)
            dN2_dy = (x1 - x3) / (2.0 * area)
            dN3_dx = (y1 - y2) / (2.0 * area)
            dN3_dy = (x2 - x1) / (2.0 * area)

            # Get the center of the element
            xc = (x1 + x2 + x3) / 3.0
            yc = (y1 + y2 + y3) / 3.0

            # Get the permittivity at the center
            eps = self.epsilon_r(xc, yc) * self.epsilon_0

            # Compute the element stiffness matrix
            K_e = np.zeros((3, 3))

            K_e[0, 0] = eps * area * (dN1_dx * dN1_dx + dN1_dy * dN1_dy)
            K_e[0, 1] = eps * area * (dN1_dx * dN2_dx + dN1_dy * dN2_dy)
            K_e[0, 2] = eps * area * (dN1_dx * dN3_dx + dN1_dy * dN3_dy)
            K_e[1, 0] = K_e[0, 1]
            K_e[1, 1] = eps * area * (dN2_dx * dN2_dx + dN2_dy * dN2_dy)
            K_e[1, 2] = eps * area * (dN2_dx * dN3_dx + dN2_dy * dN3_dy)
            K_e[2, 0] = K_e[0, 2]
            K_e[2, 1] = K_e[1, 2]
            K_e[2, 2] = eps * area * (dN3_dx * dN3_dx + dN3_dy * dN3_dy)

            # Get the charge density at the center
            rho_val = rho(xc, yc, self.n, self.p)

            # Compute the element right-hand side vector
            f_e = np.zeros(3)

            f_e[0] = -rho_val * area / 3.0
            f_e[1] = -rho_val * area / 3.0
            f_e[2] = -rho_val * area / 3.0

            # Assemble the element stiffness matrix and right-hand side vector
            for i in range(3):
                for j in range(3):
                    A[element[i], element[j]] += K_e[i, j]
                b[element[i]] += f_e[i]

        # Apply Dirichlet boundary conditions
        for i in range(n_nodes):
            x = self.mesh.get_nodes()[i][0]
            y = self.mesh.get_nodes()[i][1]

            # Left boundary (p-type)
            if x < 1e-6:
                A[i, :] = 0.0
                A[i, i] = 1.0
                b[i] = V_p
            # Right boundary (n-type)
            elif x > self.mesh.get_lx() - 1e-6:
                A[i, :] = 0.0
                A[i, i] = 1.0
                b[i] = V_n

        # Convert to CSR format for efficient solving
        A = A.tocsr()

        # Solve the system
        self.phi = spsolve(A, b)

        end_time = time.time()
        print(f"Poisson solve time: {end_time - start_time:.2f} seconds")

        # Compute the electric field
        self._compute_electric_field()

        # Update carrier concentrations
        self._update_carrier_concentrations()

        return self.phi, self.n, self.p, self.E_field

    def _compute_electric_field(self):
        """
        Compute the electric field from the potential.
        """
        # Compute the electric field at each node
        for i in range(self.mesh.get_num_nodes()):
            # Find all elements containing this node
            node_elements = []
            for e in range(self.mesh.get_num_elements()):
                element = self.mesh.get_elements()[e]
                if i in element:
                    node_elements.append(e)

            # Compute the average electric field from all elements containing this node
            E_avg = np.zeros(2)

            for e in node_elements:
                element = self.mesh.get_elements()[e]
                nodes = [self.mesh.get_nodes()[j] for j in element]

                # Calculate element area
                x1, y1 = nodes[0]
                x2, y2 = nodes[1]
                x3, y3 = nodes[2]

                # Compute the derivatives of the shape functions
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

                dN1_dx = (y2 - y3) / (2.0 * area)
                dN1_dy = (x3 - x2) / (2.0 * area)
                dN2_dx = (y3 - y1) / (2.0 * area)
                dN2_dy = (x1 - x3) / (2.0 * area)
                dN3_dx = (y1 - y2) / (2.0 * area)
                dN3_dy = (x2 - x1) / (2.0 * area)

                # Compute the gradient of the potential
                dphi_dx = self.phi[element[0]] * dN1_dx + self.phi[element[1]] * dN2_dx + self.phi[element[2]] * dN3_dx
                dphi_dy = self.phi[element[0]] * dN1_dy + self.phi[element[1]] * dN2_dy + self.phi[element[2]] * dN3_dy

                # The electric field is the negative gradient of the potential
                E_elem = np.array([-dphi_dx, -dphi_dy])

                # Add to the average
                E_avg += E_elem

            # Compute the average
            if node_elements:
                E_avg /= len(node_elements)

            # Store the electric field
            self.E_field[i] = E_avg

    def _update_carrier_concentrations(self):
        """
        Update carrier concentrations based on the potential.
        """
        # Calculate intrinsic carrier concentration
        ni = np.sqrt(self.N_c * self.N_v) * np.exp(-self.E_g / (2.0 * self.kT))

        # Update carrier concentrations at each node
        for i in range(self.mesh.get_num_nodes()):
            x = self.mesh.get_nodes()[i][0]
            y = self.mesh.get_nodes()[i][1]

            # Get doping at this position
            doping = self.doping_profile(x, y)

            # Update carrier concentrations based on potential and doping
            if doping > 0:
                # n-type region
                self.n[i] = self.N_c * np.exp((self.phi[i]) / self.kT)
                self.p[i] = self.N_v * np.exp(-(self.phi[i] + self.E_g) / self.kT)
            elif doping < 0:
                # p-type region
                self.n[i] = self.N_c * np.exp((self.phi[i]) / self.kT)
                self.p[i] = self.N_v * np.exp(-(self.phi[i] + self.E_g) / self.kT)
            else:
                # Intrinsic region
                self.n[i] = ni
                self.p[i] = ni

            # Ensure minimum carrier concentrations for numerical stability
            self.n[i] = max(self.n[i], 1e5)
            self.p[i] = max(self.p[i], 1e5)

    def get_potential(self):
        """
        Get the computed electrostatic potential.

        Returns:
            The electrostatic potential vector
        """
        return self.phi

    def get_electron_concentration(self):
        """
        Get the computed electron concentration.

        Returns:
            The electron concentration vector
        """
        return self.n

    def get_hole_concentration(self):
        """
        Get the computed hole concentration.

        Returns:
            The hole concentration vector
        """
        return self.p

    def get_electric_field(self):
        """
        Get the computed electric field.

        Returns:
            The electric field vectors
        """
        return self.E_field


def run_simulation(bias_voltage=0.0, mesh_size=51):
    """
    Run a simulation of a P-N junction diode.

    Args:
        bias_voltage: The bias voltage applied to the diode
        mesh_size: The size of the mesh (number of nodes in each dimension)

    Returns:
        solver: The SimplifiedPoissonDriftDiffusionSolver object
        mesh: The mesh
    """
    # Create a mesh for the P-N junction diode
    Lx = 200.0  # nm
    Ly = 100.0  # nm
    mesh = Mesh(Lx, Ly, mesh_size, mesh_size // 2, 1)  # Linear elements

    # Define the doping profile
    def doping_profile(x, y):
        # P-N junction at x = Lx/2
        if x < Lx / 2.0:
            return -1e17  # P-type (acceptors)
        else:
            return 1e17  # N-type (donors)

    # Define the relative permittivity
    def epsilon_r(x, y):
        return 12.9  # GaAs

    # Create the solver
    solver = SimplifiedPoissonDriftDiffusionSolver(mesh, epsilon_r, doping_profile)

    # Solve the Poisson equation
    print(f"Solving for bias voltage = {bias_voltage} V")
    start_time = time.time()
    phi, n, p, E_field = solver.solve(0.0, bias_voltage)
    end_time = time.time()
    print(f"Solution time: {end_time - start_time:.2f} seconds")

    return solver, mesh


def plot_results(solver, mesh, bias_voltage=0.0):
    """
    Plot the results of the simulation.

    Args:
        solver: The SimplifiedPoissonDriftDiffusionSolver object
        mesh: The mesh
        bias_voltage: The bias voltage applied to the diode
    """
    # Get the results
    potential = solver.get_potential()
    n = solver.get_electron_concentration()
    p = solver.get_hole_concentration()
    E_field = solver.get_electric_field()

    # Create a grid for plotting
    Lx = mesh.get_lx()
    Ly = mesh.get_ly()
    nx, ny = 200, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Interpolate the results onto the grid
    potential_grid = np.zeros((ny, nx))
    n_grid = np.zeros((ny, nx))
    p_grid = np.zeros((ny, nx))
    Ex_grid = np.zeros((ny, nx))
    Ey_grid = np.zeros((ny, nx))

    # Simple nearest-neighbor interpolation
    for i in range(nx):
        for j in range(ny):
            # Find the nearest node
            min_dist = float('inf')
            nearest_node = 0

            for k in range(mesh.get_num_nodes()):
                node = mesh.get_nodes()[k]
                dist = (node[0] - x[i])**2 + (node[1] - y[j])**2

                if dist < min_dist:
                    min_dist = dist
                    nearest_node = k

            # Interpolate the results
            potential_grid[j, i] = potential[nearest_node]
            n_grid[j, i] = n[nearest_node]
            p_grid[j, i] = p[nearest_node]
            Ex_grid[j, i] = E_field[nearest_node][0]
            Ey_grid[j, i] = E_field[nearest_node][1]

    # Create the figure
    fig = plt.figure(figsize=(15, 10))

    # Plot the potential
    ax1 = fig.add_subplot(231)
    im1 = ax1.imshow(potential_grid, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    ax1.set_title('Electrostatic Potential (V)')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1)

    # Plot the electron concentration
    ax2 = fig.add_subplot(232)
    im2 = ax2.imshow(np.log10(n_grid), extent=[0, Lx, 0, Ly], origin='lower', cmap='plasma')
    ax2.set_title('Electron Concentration (log10(cm^-3))')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(im2, ax=ax2)

    # Plot the hole concentration
    ax3 = fig.add_subplot(233)
    im3 = ax3.imshow(np.log10(p_grid), extent=[0, Lx, 0, Ly], origin='lower', cmap='plasma')
    ax3.set_title('Hole Concentration (log10(cm^-3))')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3)

    # Plot the electric field
    ax4 = fig.add_subplot(234)
    E_mag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    im4 = ax4.imshow(E_mag, extent=[0, Lx, 0, Ly], origin='lower', cmap='inferno')
    ax4.set_title('Electric Field Magnitude (V/cm)')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')
    plt.colorbar(im4, ax=ax4)

    # Plot the electric field vectors
    ax5 = fig.add_subplot(235)
    skip = 5
    ax5.quiver(X[::skip, ::skip], Y[::skip, ::skip], Ex_grid[::skip, ::skip], Ey_grid[::skip, ::skip],
              scale=50, width=0.002)
    ax5.set_title('Electric Field Vectors')
    ax5.set_xlabel('x (nm)')
    ax5.set_ylabel('y (nm)')

    # Plot the potential in 3D
    ax6 = fig.add_subplot(236, projection='3d')
    surf = ax6.plot_surface(X, Y, potential_grid, cmap='viridis', linewidth=0, antialiased=False)
    ax6.set_title('Electrostatic Potential (3D)')
    ax6.set_xlabel('x (nm)')
    ax6.set_ylabel('y (nm)')
    ax6.set_zlabel('V (V)')
    ax6.view_init(30, 45)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'simplified_pn_junction_bias_{bias_voltage:.1f}.png')
    plt.close()


def main():
    """
    Main function.
    """
    # Run simulations for different bias voltages
    bias_voltages = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for bias in bias_voltages:
        # Run the simulation
        solver, mesh = run_simulation(bias_voltage=bias, mesh_size=31)

        # Plot the results
        plot_results(solver, mesh, bias_voltage=bias)

        print(f"Results saved to simplified_pn_junction_bias_{bias:.1f}.png")


if __name__ == "__main__":
    main()
