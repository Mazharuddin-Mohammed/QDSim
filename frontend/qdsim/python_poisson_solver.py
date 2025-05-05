"""
Python implementation of the Poisson solver.

This module provides a pure Python implementation of the Poisson solver
for use when the C++ implementation is not available.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Import Thomas algorithm for tridiagonal systems
from .thomas_solver import thomas_algorithm

class PythonPoissonSolver:
    """
    Python implementation of the Poisson solver.

    This class solves the Poisson equation using the finite element method.
    It is used as a fallback when the C++ implementation is not available.
    """

    def __init__(self, mesh):
        """
        Initialize the Poisson solver.

        Args:
            mesh: The mesh to use for the simulation
        """
        self.mesh = mesh

    def solve(self, epsilon_r, rho, V_left, V_right, use_thomas=True):
        """
        Solve the Poisson equation.

        Args:
            epsilon_r: Function that returns the relative permittivity at a given position
            rho: Function that returns the charge density at a given position
            V_left: Voltage at the left boundary
            V_right: Voltage at the right boundary
            use_thomas: Whether to use the Thomas algorithm for 1D problems

        Returns:
            The solution vector (electrostatic potential)
        """
        # Constants
        epsilon_0 = 8.85418782e-14  # Vacuum permittivity (F/cm)

        # Get the number of nodes
        n_nodes = self.mesh.get_num_nodes()

        # Check if this is a 1D problem (all nodes have the same y-coordinate)
        is_1d = True
        y0 = self.mesh.get_nodes()[0][1]
        for i in range(1, n_nodes):
            if abs(self.mesh.get_nodes()[i][1] - y0) > 1e-6:
                is_1d = False
                break

        # For 1D problems, we can use the Thomas algorithm
        if is_1d and use_thomas:
            print("Using Thomas algorithm for 1D problem")

            # Get the x-coordinates of the nodes
            x_coords = np.array([self.mesh.get_nodes()[i][0] for i in range(n_nodes)])

            # Sort the nodes by x-coordinate
            sorted_indices = np.argsort(x_coords)
            x_sorted = x_coords[sorted_indices]

            # Create the tridiagonal system
            a = np.zeros(n_nodes)  # Lower diagonal
            b = np.zeros(n_nodes)  # Main diagonal
            c = np.zeros(n_nodes)  # Upper diagonal
            d = np.zeros(n_nodes)  # Right-hand side

            # Fill the tridiagonal system
            for i in range(1, n_nodes - 1):
                # Get the node and its neighbors
                x_i = x_sorted[i]
                x_im1 = x_sorted[i - 1]
                x_ip1 = x_sorted[i + 1]

                # Calculate the element lengths
                h_im1 = x_i - x_im1
                h_i = x_ip1 - x_i

                # Get the permittivity at this node
                eps_i = epsilon_r(x_i, y0) * epsilon_0

                # Get the charge density at this node
                rho_i = rho(x_i, y0)

                # Fill the tridiagonal system
                a[i] = -eps_i / h_im1
                b[i] = eps_i * (1.0 / h_im1 + 1.0 / h_i)
                c[i] = -eps_i / h_i
                d[i] = -rho_i

            # Apply boundary conditions
            b[0] = 1.0
            c[0] = 0.0
            d[0] = V_left

            a[n_nodes - 1] = 0.0
            b[n_nodes - 1] = 1.0
            d[n_nodes - 1] = V_right

            # Solve the tridiagonal system using the Thomas algorithm
            phi_sorted = thomas_algorithm(a, b, c, d)

            # Reorder the solution to match the original node ordering
            phi = np.zeros(n_nodes)
            for i in range(n_nodes):
                phi[sorted_indices[i]] = phi_sorted[i]

            return phi

        # For 2D problems, use the standard FEM approach
        # Create the stiffness matrix and right-hand side vector
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
            eps = epsilon_r(xc, yc) * epsilon_0

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
            rho_val = rho(xc, yc)

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

            # Left boundary
            if x < 1e-6:
                A[i, :] = 0.0
                A[i, i] = 1.0
                b[i] = V_left
            # Right boundary
            elif x > self.mesh.get_lx() - 1e-6:
                A[i, :] = 0.0
                A[i, i] = 1.0
                b[i] = V_right

        # Convert to CSR format for efficient solving
        A = A.tocsr()

        # Solve the system
        phi = spsolve(A, b)

        return phi

    def compute_electric_field(self, phi):
        """
        Compute the electric field from the potential.

        Args:
            phi: The electrostatic potential vector

        Returns:
            The electric field vectors
        """
        # Initialize the electric field
        E_field = [np.zeros(2) for _ in range(self.mesh.get_num_nodes())]

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
                dphi_dx = phi[element[0]] * dN1_dx + phi[element[1]] * dN2_dx + phi[element[2]] * dN3_dx
                dphi_dy = phi[element[0]] * dN1_dy + phi[element[1]] * dN2_dy + phi[element[2]] * dN3_dy

                # The electric field is the negative gradient of the potential
                E_elem = np.array([-dphi_dx, -dphi_dy])

                # Add to the average
                E_avg += E_elem

            # Compute the average
            if node_elements:
                E_avg /= len(node_elements)

            # Store the electric field
            E_field[i] = E_avg

        return E_field
