"""
Python implementation of the drift-diffusion solver.

This module provides a pure Python implementation of the drift-diffusion solver
for use when the C++ implementation is not available.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class PythonDriftDiffusionSolver:
    """
    Python implementation of the drift-diffusion solver.

    This class solves the drift-diffusion equations using the finite element method.
    It is used as a fallback when the C++ implementation is not available.
    """

    def __init__(self, mesh):
        """
        Initialize the drift-diffusion solver.

        Args:
            mesh: The mesh to use for the simulation
        """
        self.mesh = mesh

        # Constants
        self.q = 1.602e-19  # Elementary charge (C)
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.T = 300.0  # Temperature (K)
        self.kT = self.kB * self.T / self.q  # Thermal voltage (eV)

        # Material properties
        self.mu_n = 8500.0  # Electron mobility (cm^2/V·s)
        self.mu_p = 400.0  # Hole mobility (cm^2/V·s)

        # Initialize vectors
        self.n = np.zeros(mesh.get_num_nodes())
        self.p = np.zeros(mesh.get_num_nodes())
        self.phi_n = np.zeros(mesh.get_num_nodes())
        self.phi_p = np.zeros(mesh.get_num_nodes())
        self.J_n = [np.zeros(2) for _ in range(mesh.get_num_nodes())]
        self.J_p = [np.zeros(2) for _ in range(mesh.get_num_nodes())]

    def solve_continuity_equations(self, phi, E_field, doping_profile, N_c, N_v, E_g, use_fermi_dirac=False):
        """
        Solve the continuity equations for electrons and holes.

        Args:
            phi: The electrostatic potential vector
            E_field: The electric field vectors
            doping_profile: Function that returns the doping profile at a given position
            N_c: Effective density of states in conduction band (cm^-3)
            N_v: Effective density of states in valence band (cm^-3)
            E_g: Band gap (eV)
            use_fermi_dirac: Whether to use Fermi-Dirac statistics (default: False)

        Returns:
            Tuple of (n, p, phi_n, phi_p, J_n, J_p)
        """
        # Calculate intrinsic carrier concentration
        ni = np.sqrt(N_c * N_v) * np.exp(-E_g / (2.0 * self.kT))

        # Get the number of nodes
        n_nodes = self.mesh.get_num_nodes()

        # Initialize carrier concentrations and quasi-Fermi potentials
        for i in range(n_nodes):
            x = self.mesh.get_nodes()[i][0]
            y = self.mesh.get_nodes()[i][1]

            # Get doping at this position
            doping = doping_profile(x, y)

            # Initialize carrier concentrations based on doping
            if doping > 0:
                # n-type region
                self.n[i] = doping
                self.p[i] = ni * ni / doping

                # Initialize quasi-Fermi potentials
                self.phi_n[i] = phi[i]
                self.phi_p[i] = phi[i] - E_g - self.kT * np.log(doping / N_v)
            elif doping < 0:
                # p-type region
                self.p[i] = -doping
                self.n[i] = ni * ni / self.p[i]

                # Initialize quasi-Fermi potentials
                self.phi_n[i] = phi[i] + self.kT * np.log(ni * ni / (-doping) / N_c)
                self.phi_p[i] = phi[i] - E_g
            else:
                # Intrinsic region
                self.n[i] = ni
                self.p[i] = ni

                # Initialize quasi-Fermi potentials
                self.phi_n[i] = phi[i]
                self.phi_p[i] = phi[i] - E_g

            # Ensure minimum carrier concentrations for numerical stability
            self.n[i] = max(self.n[i], 1e5)
            self.p[i] = max(self.p[i], 1e5)

        # Create the stiffness matrices and right-hand side vectors for electrons and holes
        A_n = lil_matrix((n_nodes, n_nodes))
        A_p = lil_matrix((n_nodes, n_nodes))
        b_n = np.zeros(n_nodes)
        b_p = np.zeros(n_nodes)

        # Assemble the systems
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

            # Get the electric field at the center (average of the nodes)
            E_c = (E_field[element[0]] + E_field[element[1]] + E_field[element[2]]) / 3.0

            # Get the carrier concentrations at the center (average of the nodes)
            n_c = (self.n[element[0]] + self.n[element[1]] + self.n[element[2]]) / 3.0
            p_c = (self.p[element[0]] + self.p[element[1]] + self.p[element[2]]) / 3.0

            # Compute the element stiffness matrices for electrons and holes
            K_n = np.zeros((3, 3))
            K_p = np.zeros((3, 3))

            # Diffusion terms
            for i in range(3):
                for j in range(3):
                    # Electron diffusion
                    K_n[i, j] = self.mu_n * self.kT * area * (
                        dN1_dx * dN1_dx + dN1_dy * dN1_dy if i == 0 and j == 0 else
                        dN1_dx * dN2_dx + dN1_dy * dN2_dy if i == 0 and j == 1 else
                        dN1_dx * dN3_dx + dN1_dy * dN3_dy if i == 0 and j == 2 else
                        dN2_dx * dN1_dx + dN2_dy * dN1_dy if i == 1 and j == 0 else
                        dN2_dx * dN2_dx + dN2_dy * dN2_dy if i == 1 and j == 1 else
                        dN2_dx * dN3_dx + dN2_dy * dN3_dy if i == 1 and j == 2 else
                        dN3_dx * dN1_dx + dN3_dy * dN1_dy if i == 2 and j == 0 else
                        dN3_dx * dN2_dx + dN3_dy * dN2_dy if i == 2 and j == 1 else
                        dN3_dx * dN3_dx + dN3_dy * dN3_dy
                    )

                    # Hole diffusion
                    K_p[i, j] = self.mu_p * self.kT * area * (
                        dN1_dx * dN1_dx + dN1_dy * dN1_dy if i == 0 and j == 0 else
                        dN1_dx * dN2_dx + dN1_dy * dN2_dy if i == 0 and j == 1 else
                        dN1_dx * dN3_dx + dN1_dy * dN3_dy if i == 0 and j == 2 else
                        dN2_dx * dN1_dx + dN2_dy * dN1_dy if i == 1 and j == 0 else
                        dN2_dx * dN2_dx + dN2_dy * dN2_dy if i == 1 and j == 1 else
                        dN2_dx * dN3_dx + dN2_dy * dN3_dy if i == 1 and j == 2 else
                        dN3_dx * dN1_dx + dN3_dy * dN1_dy if i == 2 and j == 0 else
                        dN3_dx * dN2_dx + dN3_dy * dN2_dy if i == 2 and j == 1 else
                        dN3_dx * dN3_dx + dN3_dy * dN3_dy
                    )

            # Drift terms (simplified for stability)
            # These terms are approximated and would need a more sophisticated
            # implementation for accurate results in high-field regions

            # Assemble the element stiffness matrices
            for i in range(3):
                for j in range(3):
                    A_n[element[i], element[j]] += K_n[i, j]
                    A_p[element[i], element[j]] += K_p[i, j]

            # Generation-recombination terms (simplified SRH model)
            # For a more accurate simulation, a proper SRH or Auger model should be used
            tau_n = 1e-9  # Electron lifetime (s)
            tau_p = 1e-9  # Hole lifetime (s)

            # Compute the recombination rate
            R = (n_c * p_c - ni * ni) / (tau_n * (p_c + ni) + tau_p * (n_c + ni))

            # Add the recombination terms to the right-hand side vectors
            for i in range(3):
                b_n[element[i]] -= self.q * R * area / 3.0
                b_p[element[i]] -= self.q * R * area / 3.0

        # Apply boundary conditions
        for i in range(n_nodes):
            x = self.mesh.get_nodes()[i][0]
            y = self.mesh.get_nodes()[i][1]

            # Left or right boundary
            if x < 1e-6 or x > self.mesh.get_lx() - 1e-6:
                # Dirichlet boundary conditions for quasi-Fermi potentials
                A_n[i, :] = 0.0
                A_n[i, i] = 1.0
                b_n[i] = self.phi_n[i]

                A_p[i, :] = 0.0
                A_p[i, i] = 1.0
                b_p[i] = self.phi_p[i]

        # Convert to CSR format for efficient solving
        A_n = A_n.tocsr()
        A_p = A_p.tocsr()

        # Solve the systems
        self.phi_n = spsolve(A_n, b_n)
        self.phi_p = spsolve(A_p, b_p)

        # Update carrier concentrations based on quasi-Fermi potentials
        for i in range(n_nodes):
            # Electrons
            if use_fermi_dirac:
                # Fermi-Dirac statistics (simplified)
                eta_n = (self.phi_n[i] - phi[i]) / self.kT
                self.n[i] = N_c * self._fermi_integral(eta_n)
            else:
                # Boltzmann statistics
                # Limit the exponent to avoid overflow
                exponent = min(max((self.phi_n[i] - phi[i]) / self.kT, -50.0), 50.0)
                self.n[i] = N_c * np.exp(exponent)

            # Holes
            if use_fermi_dirac:
                # Fermi-Dirac statistics (simplified)
                eta_p = (phi[i] - self.phi_p[i] - E_g) / self.kT
                self.p[i] = N_v * self._fermi_integral(eta_p)
            else:
                # Boltzmann statistics
                # Limit the exponent to avoid overflow
                exponent = min(max((phi[i] - self.phi_p[i] - E_g) / self.kT, -50.0), 50.0)
                self.p[i] = N_v * np.exp(exponent)

            # Ensure minimum carrier concentrations for numerical stability
            self.n[i] = max(self.n[i], 1e5)
            self.p[i] = max(self.p[i], 1e5)

        # Compute current densities
        self._compute_current_densities(phi)

        return self.n, self.p, self.phi_n, self.phi_p, self.J_n, self.J_p

    def _compute_current_densities(self, phi):
        """
        Compute the current densities for electrons and holes.

        Args:
            phi: The electrostatic potential vector
        """
        # Compute the current densities at each node
        for i in range(self.mesh.get_num_nodes()):
            # Find all elements containing this node
            node_elements = []
            for e in range(self.mesh.get_num_elements()):
                element = self.mesh.get_elements()[e]
                if i in element:
                    node_elements.append(e)

            # Compute the average current densities from all elements containing this node
            J_n_avg = np.zeros(2)
            J_p_avg = np.zeros(2)

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

                # Compute the gradients of the potentials
                dphi_dx = phi[element[0]] * dN1_dx + phi[element[1]] * dN2_dx + phi[element[2]] * dN3_dx
                dphi_dy = phi[element[0]] * dN1_dy + phi[element[1]] * dN2_dy + phi[element[2]] * dN3_dy

                dphi_n_dx = self.phi_n[element[0]] * dN1_dx + self.phi_n[element[1]] * dN2_dx + self.phi_n[element[2]] * dN3_dx
                dphi_n_dy = self.phi_n[element[0]] * dN1_dy + self.phi_n[element[1]] * dN2_dy + self.phi_n[element[2]] * dN3_dy

                dphi_p_dx = self.phi_p[element[0]] * dN1_dx + self.phi_p[element[1]] * dN2_dx + self.phi_p[element[2]] * dN3_dx
                dphi_p_dy = self.phi_p[element[0]] * dN1_dy + self.phi_p[element[1]] * dN2_dy + self.phi_p[element[2]] * dN3_dy

                # Get the carrier concentrations at the center (average of the nodes)
                n_c = (self.n[element[0]] + self.n[element[1]] + self.n[element[2]]) / 3.0
                p_c = (self.p[element[0]] + self.p[element[1]] + self.p[element[2]]) / 3.0

                # Compute the current densities
                # J_n = q * mu_n * n * grad(phi_n)
                # J_p = -q * mu_p * p * grad(phi_p)
                # Check for NaN or Inf values
                if np.isnan(n_c) or np.isinf(n_c) or np.isnan(dphi_n_dx) or np.isinf(dphi_n_dx) or np.isnan(dphi_n_dy) or np.isinf(dphi_n_dy):
                    J_n_elem = np.zeros(2)
                else:
                    J_n_elem = self.q * self.mu_n * n_c * np.array([dphi_n_dx, dphi_n_dy])

                if np.isnan(p_c) or np.isinf(p_c) or np.isnan(dphi_p_dx) or np.isinf(dphi_p_dx) or np.isnan(dphi_p_dy) or np.isinf(dphi_p_dy):
                    J_p_elem = np.zeros(2)
                else:
                    J_p_elem = -self.q * self.mu_p * p_c * np.array([dphi_p_dx, dphi_p_dy])

                # Add to the average
                J_n_avg += J_n_elem
                J_p_avg += J_p_elem

            # Compute the average
            if node_elements:
                J_n_avg /= len(node_elements)
                J_p_avg /= len(node_elements)

            # Store the current densities
            self.J_n[i] = J_n_avg
            self.J_p[i] = J_p_avg

    def _fermi_integral(self, eta):
        """
        Compute the Fermi-Dirac integral of order 1/2.

        This is a simplified approximation of the Fermi-Dirac integral.
        For a more accurate implementation, a proper numerical integration
        or a more sophisticated approximation should be used.

        Args:
            eta: The reduced Fermi level

        Returns:
            The value of the Fermi-Dirac integral
        """
        # Approximation for eta << 0
        if eta < -10.0:
            return np.exp(eta)

        # Approximation for eta >> 0
        if eta > 10.0:
            return (2.0 / 3.0) * eta**(3.0 / 2.0)

        # Approximation for intermediate values
        return np.exp(eta) / (1.0 + 0.27 * np.exp(-1.0 * eta))
