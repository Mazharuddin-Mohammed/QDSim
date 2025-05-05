"""
Python implementation of the Schrodinger solver.

This module provides a pure Python implementation of the Schrodinger solver
for use when the C++ implementation is not available.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# Import GPU fallback module
from .gpu_fallback import (
    is_gpu_available, cp, cp_lil_matrix, cp_csr_matrix, cp_eigsh,
    to_gpu, to_cpu, solve_generalized_eigenvalue_problem
)

class PythonSchrodingerSolver:
    """
    Python implementation of the Schrodinger solver.

    This class solves the Schrodinger equation using the finite element method.
    It is used as a fallback when the C++ implementation is not available.
    """

    def __init__(self, mesh, potential_function, use_gpu=False):
        """
        Initialize the Schrodinger solver.

        Args:
            mesh: The mesh to use for the simulation
            potential_function: Function that returns the potential at a given position
            use_gpu: Whether to use GPU acceleration if available
        """
        self.mesh = mesh
        self.potential_function = potential_function

        # Constants
        self.hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
        self.m0 = 9.10938356e-31  # Electron mass (kg)
        self.q = 1.602176634e-19  # Elementary charge (C)

        # Effective mass (in units of m0)
        self.m_eff = 0.067  # GaAs

        # Conversion factor from J to eV
        self.J_to_eV = 1.0 / self.q

        # Conversion factor for the Hamiltonian
        # hbar^2 / (2 * m_eff * m0) in units of eV·nm^2
        self.hbar_factor = self.hbar**2 / (2.0 * self.m_eff * self.m0) * self.J_to_eV * 1e18

        # GPU acceleration
        self.use_gpu = use_gpu and is_gpu_available()

        if self.use_gpu:
            print("GPU acceleration enabled for SchrodingerSolver")
        else:
            print("GPU acceleration not available or disabled for SchrodingerSolver")

    def solve(self, num_states=5):
        """
        Solve the Schrodinger equation.

        Args:
            num_states: Number of eigenstates to compute
                       Note: Using a smaller number of states for the Python implementation
                       for better performance.

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        print("Solving Schrodinger equation using Python implementation")
        print("This may take a few minutes...")

        # Get the number of nodes
        n_nodes = self.mesh.get_num_nodes()

        # Limit the number of states for the Python implementation
        num_states = min(num_states, 5)

        # Create the Hamiltonian matrix and overlap matrix
        if self.use_gpu:
            H = cp_lil_matrix((n_nodes, n_nodes))
            S = cp_lil_matrix((n_nodes, n_nodes))

            # Get all elements and nodes at once for faster processing
            elements = to_gpu(np.array([self.mesh.get_elements()[e] for e in range(self.mesh.get_num_elements())]))
            nodes = to_gpu(np.array(self.mesh.get_nodes()))
        else:
            H = lil_matrix((n_nodes, n_nodes))
            S = lil_matrix((n_nodes, n_nodes))

            # Get all elements and nodes at once for faster processing
            elements = np.array([self.mesh.get_elements()[e] for e in range(self.mesh.get_num_elements())])
            nodes = np.array(self.mesh.get_nodes())

        # Assemble the matrices
        print("Assembling matrices...")
        for e in range(self.mesh.get_num_elements()):
            if e % 100 == 0:
                print(f"  Processing element {e}/{self.mesh.get_num_elements()}")

            element = elements[e]
            node1, node2, node3 = nodes[element[0]], nodes[element[1]], nodes[element[2]]

            # Calculate element area
            x1, y1 = node1
            x2, y2 = node2
            x3, y3 = node3

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

            # Get the potential at the center
            V = self.potential_function(xc, yc)

            # Compute the element matrices
            T_e = np.zeros((3, 3))  # Kinetic energy
            V_e = np.zeros((3, 3))  # Potential energy
            M_e = np.zeros((3, 3))  # Mass matrix

            # Kinetic energy matrix
            T_e[0, 0] = self.hbar_factor * area * (dN1_dx * dN1_dx + dN1_dy * dN1_dy)
            T_e[0, 1] = self.hbar_factor * area * (dN1_dx * dN2_dx + dN1_dy * dN2_dy)
            T_e[0, 2] = self.hbar_factor * area * (dN1_dx * dN3_dx + dN1_dy * dN3_dy)
            T_e[1, 0] = T_e[0, 1]
            T_e[1, 1] = self.hbar_factor * area * (dN2_dx * dN2_dx + dN2_dy * dN2_dy)
            T_e[1, 2] = self.hbar_factor * area * (dN2_dx * dN3_dx + dN2_dy * dN3_dy)
            T_e[2, 0] = T_e[0, 2]
            T_e[2, 1] = T_e[1, 2]
            T_e[2, 2] = self.hbar_factor * area * (dN3_dx * dN3_dx + dN3_dy * dN3_dy)

            # Potential energy matrix
            V_e[0, 0] = V * area / 6.0
            V_e[0, 1] = V * area / 12.0
            V_e[0, 2] = V * area / 12.0
            V_e[1, 0] = V_e[0, 1]
            V_e[1, 1] = V * area / 6.0
            V_e[1, 2] = V * area / 12.0
            V_e[2, 0] = V_e[0, 2]
            V_e[2, 1] = V_e[1, 2]
            V_e[2, 2] = V * area / 6.0

            # Mass matrix
            M_e[0, 0] = area / 6.0
            M_e[0, 1] = area / 12.0
            M_e[0, 2] = area / 12.0
            M_e[1, 0] = M_e[0, 1]
            M_e[1, 1] = area / 6.0
            M_e[1, 2] = area / 12.0
            M_e[2, 0] = M_e[0, 2]
            M_e[2, 1] = M_e[1, 2]
            M_e[2, 2] = area / 6.0

            # Assemble the element matrices
            for i in range(3):
                for j in range(3):
                    H[element[i], element[j]] += T_e[i, j] + V_e[i, j]
                    S[element[i], element[j]] += M_e[i, j]

        # Convert to CSR format for efficient solving
        print("Converting matrices to CSR format...")
        if self.use_gpu:
            H = H.tocsr()
            S = S.tocsr()

            # Solve the generalized eigenvalue problem on GPU
            print(f"Solving eigenvalue problem for {num_states} states on GPU...")
            eigenvalues, eigenvectors = solve_generalized_eigenvalue_problem(
                H, S, k=min(num_states, n_nodes-2), sigma=0.0, which='LM'
            )
            print("Eigenvalue problem solved successfully")
        else:
            H = H.tocsr()
            S = S.tocsr()

            # Solve the generalized eigenvalue problem on CPU
            print(f"Solving eigenvalue problem for {num_states} states on CPU...")
            eigenvalues, eigenvectors = eigsh(H, k=min(num_states, n_nodes-2), M=S, sigma=0.0, which='LM')

        print("Eigenvalue problem solved successfully")
        return eigenvalues, eigenvectors.T
