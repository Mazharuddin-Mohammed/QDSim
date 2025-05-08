"""
Python implementation of the SchrodingerSolver class.

This is a fallback implementation that is used when the C++ implementation is not available.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import warnings

class SchrodingerSolver:
    """
    Python implementation of the SchrodingerSolver class.

    This class solves the Schrödinger equation using the finite element method.
    It is a fallback implementation that is used when the C++ implementation is not available.
    """

    def __init__(self, mesh, m_star, V, use_gpu=False):
        """
        Initialize the SchrodingerSolver.

        Args:
            mesh: The mesh on which to solve the Schrödinger equation
            m_star: Function that returns the effective mass at a given position
            V: Function that returns the potential at a given position
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.mesh = mesh
        self.m_star = m_star
        self.V = V
        self.use_gpu = use_gpu

        if use_gpu:
            warnings.warn("GPU acceleration is not available in the Python implementation.")

        # Initialize matrices
        self.H = None
        self.M = None
        self.eigenvalues = None
        self.eigenvectors = None

        # Assemble matrices
        self.assemble_matrices()

    def assemble_matrices(self):
        """
        Assemble the Hamiltonian and mass matrices.
        """
        # Get mesh data
        num_nodes = self.mesh.get_num_nodes()
        num_elements = self.mesh.get_num_elements()
        nodes = self.mesh.get_nodes()
        elements = self.mesh.get_elements()

        # Initialize matrices
        self.H = lil_matrix((num_nodes, num_nodes), dtype=np.complex128)
        self.M = lil_matrix((num_nodes, num_nodes), dtype=np.complex128)

        # Assemble matrices element by element
        for e in range(num_elements):
            # Get element nodes
            element = elements[e]
            n1 = element[0]
            n2 = element[1]
            n3 = element[2]

            # Get node coordinates
            x1 = nodes[n1][0]
            y1 = nodes[n1][1]
            x2 = nodes[n2][0]
            y2 = nodes[n2][1]
            x3 = nodes[n3][0]
            y3 = nodes[n3][1]

            # Calculate element area
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

            # Calculate shape function gradients
            b1 = (y2 - y3) / (2.0 * area)
            c1 = (x3 - x2) / (2.0 * area)
            b2 = (y3 - y1) / (2.0 * area)
            c2 = (x1 - x3) / (2.0 * area)
            b3 = (y1 - y2) / (2.0 * area)
            c3 = (x2 - x1) / (2.0 * area)

            # Calculate element centroid
            xc = (x1 + x2 + x3) / 3.0
            yc = (y1 + y2 + y3) / 3.0

            # Get effective mass and potential at centroid
            m = self.m_star(xc, yc)
            V_val = self.V(xc, yc)

            # Assemble element matrices
            for i in range(3):
                for j in range(3):
                    # Get global node indices
                    ni = element[i]
                    nj = element[j]

                    # Calculate gradients of shape functions
                    if i == 0:
                        dNi_dx = b1
                        dNi_dy = c1
                    elif i == 1:
                        dNi_dx = b2
                        dNi_dy = c2
                    else:
                        dNi_dx = b3
                        dNi_dy = c3

                    if j == 0:
                        dNj_dx = b1
                        dNj_dy = c1
                    elif j == 1:
                        dNj_dx = b2
                        dNj_dy = c2
                    else:
                        dNj_dx = b3
                        dNj_dy = c3

                    # Calculate shape functions at centroid
                    Ni = 1.0 / 3.0
                    Nj = 1.0 / 3.0

                    # Calculate Hamiltonian matrix element
                    hbar = 6.582119569e-16  # eV·s
                    kinetic_term = (hbar * hbar / (2.0 * m)) * (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * area
                    potential_term = V_val * Ni * Nj * area
                    H_ij = kinetic_term + potential_term

                    # Calculate mass matrix element
                    M_ij = Ni * Nj * area

                    # Add to global matrices
                    self.H[ni, nj] += H_ij
                    self.M[ni, nj] += M_ij

        # Convert to CSR format for efficient computation
        self.H = self.H.tocsr()
        self.M = self.M.tocsr()

    def solve(self, num_eigenvalues=10):
        """
        Solve the Schrödinger equation.

        Args:
            num_eigenvalues: The number of eigenvalues to compute

        Returns:
            A pair containing the eigenvalues and eigenvectors
        """
        # Check if the matrices are assembled
        if self.H is None or self.M is None:
            raise RuntimeError("Matrices not assembled")

        # Check if the number of eigenvalues is valid
        if num_eigenvalues <= 0 or num_eigenvalues > self.H.shape[0]:
            raise ValueError("Invalid number of eigenvalues")

        # Solve the generalized eigenvalue problem
        try:
            # Convert to CSR format for better compatibility with eigsh
            H_csr = self.H.tocsr()
            M_csr = self.M.tocsr()

            # Make sure the matrices are real (discard imaginary part if any)
            H_real = H_csr.real
            M_real = M_csr.real

            # Use a simpler solver configuration first
            self.eigenvalues, eigvecs = eigsh(H_real, k=num_eigenvalues, M=M_real, which='SM')

            # Store eigenvectors as a list
            self.eigenvectors = [eigvecs[:, i] for i in range(eigvecs.shape[1])]

            # Convert eigenvectors to complex if needed
            if np.iscomplexobj(self.H) or np.iscomplexobj(self.M):
                self.eigenvectors = [v.astype(np.complex128) for v in self.eigenvectors]
        except Exception as e:
            # Try with a different solver configuration
            try:
                # Use a shift-invert mode with a small shift
                self.eigenvalues, eigvecs = eigsh(H_real, k=num_eigenvalues, M=M_real,
                                                 sigma=0.1, which='LM', mode='cayley',
                                                 maxiter=10000, tol=1e-5)

                # Store eigenvectors as a list
                self.eigenvectors = [eigvecs[:, i] for i in range(eigvecs.shape[1])]

                # Convert eigenvectors to complex if needed
                if np.iscomplexobj(self.H) or np.iscomplexobj(self.M):
                    self.eigenvectors = [v.astype(np.complex128) for v in self.eigenvectors]
            except Exception as e2:
                # Try one more approach with lobpcg
                try:
                    from scipy.sparse.linalg import lobpcg

                    # Create random initial vectors
                    X = np.random.rand(H_real.shape[0], num_eigenvalues)

                    # Solve with LOBPCG
                    eigenvalues, eigenvectors = lobpcg(H_real, X, M=M_real, largest=False, maxiter=1000, tol=1e-5)

                    # Store results
                    self.eigenvalues = eigenvalues
                    self.eigenvectors = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]

                    # Convert eigenvectors to complex if needed
                    if np.iscomplexobj(self.H) or np.iscomplexobj(self.M):
                        self.eigenvectors = [v.astype(np.complex128) for v in self.eigenvectors]
                except Exception as e3:
                    raise RuntimeError(f"Failed to solve eigenvalue problem: {e}, {e2}, {e3}")

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = np.array(self.eigenvalues)[idx]

        # Reorder eigenvectors according to sorted eigenvalues
        sorted_eigenvectors = []
        for i in idx:
            if i < len(self.eigenvectors):
                sorted_eigenvectors.append(self.eigenvectors[i])

        self.eigenvectors = sorted_eigenvectors

        return self.eigenvalues.tolist(), self.eigenvectors

    def get_eigenvalues(self):
        """
        Get the eigenvalues.

        Returns:
            The eigenvalues
        """
        if self.eigenvalues is None:
            raise RuntimeError("Eigenvalues not computed")

        return self.eigenvalues.tolist()

    def get_eigenvectors(self):
        """
        Get the eigenvectors.

        Returns:
            The eigenvectors
        """
        if self.eigenvectors is None:
            raise RuntimeError("Eigenvectors not computed")

        return [self.eigenvectors[:, i] for i in range(self.eigenvalues.size)]

    def get_H(self):
        """
        Get the Hamiltonian matrix.

        Returns:
            The Hamiltonian matrix
        """
        return self.H

    def get_M(self):
        """
        Get the mass matrix.

        Returns:
            The mass matrix
        """
        return self.M

    def get_mesh(self):
        """
        Get the mesh.

        Returns:
            The mesh
        """
        return self.mesh

    def is_gpu_enabled(self):
        """
        Check if GPU acceleration is enabled.

        Returns:
            False (GPU acceleration is not available in the Python implementation)
        """
        return False
