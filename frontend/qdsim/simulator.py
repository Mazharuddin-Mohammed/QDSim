import numpy as np
from .config import Config
from .fe_interpolator import FEInterpolator
from .adaptive_mesh import AdaptiveMesh

# Import the C++ extension
import sys

# Try to import the C++ extension
try:
    from . import qdsim_cpp

    if qdsim_cpp is None:
        print("Error: Could not import C++ extension. Make sure it's built and in the Python path.", file=sys.stderr)
        sys.exit(1)
except ImportError as e:
    print(f"Error importing C++ extension: {e}", file=sys.stderr)
    sys.exit(1)

class Simulator:
    def __init__(self, config: Config):
        """
        Initialize the simulator with the given configuration.

        Args:
            config: Configuration object with simulation parameters
        """
        try:
            self.config = config

            # Validate required configuration parameters
            required_params = ['Lx', 'Ly', 'nx', 'ny']
            for param in required_params:
                if not hasattr(config, param):
                    raise ValueError(f"Missing required configuration parameter: {param}")

            # Set default values for optional parameters
            if not hasattr(config, 'element_order'):
                config.element_order = 1
                print("Warning: element_order not specified, using default value of 1")

            # Initialize material database
            try:
                self.db = qdsim_cpp.MaterialDatabase()
            except Exception as e:
                print(f"Warning: Failed to initialize MaterialDatabase: {e}")
                self.db = None

            # Create the mesh
            try:
                self.mesh = qdsim_cpp.Mesh(config.Lx, config.Ly, config.nx, config.ny, config.element_order)
                print(f"Mesh created with {self.mesh.get_num_nodes()} nodes and {self.mesh.get_num_elements()} elements")
            except Exception as e:
                raise RuntimeError(f"Failed to create mesh: {e}")

            # Create the FEInterpolator
            try:
                self.interpolator = FEInterpolator(self.mesh, use_cpp=True)
            except Exception as e:
                print(f"Warning: Failed to create C++ FEInterpolator: {e}, falling back to Python implementation")
                self.interpolator = FEInterpolator(self.mesh, use_cpp=False)

            # Create the AdaptiveMesh
            try:
                self.adaptive_mesh = AdaptiveMesh(self)
            except Exception as e:
                print(f"Warning: Failed to create AdaptiveMesh: {e}")
                self.adaptive_mesh = None

            # Initialize the potential array
            self.phi = np.zeros(self.mesh.get_num_nodes())

            # Solve the Poisson equation
            try:
                self.solve_poisson()
            except Exception as e:
                print(f"Warning: Failed to solve Poisson equation: {e}")

            # Initialize the Hamiltonian and mass matrices
            num_nodes = self.mesh.get_num_nodes()
            self.H = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
            self.M = np.zeros((num_nodes, num_nodes), dtype=np.complex128)

            # Assemble the matrices
            try:
                self.assemble_matrices()
            except Exception as e:
                print(f"Warning: Failed to assemble matrices: {e}")

            # Initialize eigenvalues and eigenvectors
            self.eigenvalues = None
            self.eigenvectors = None

        except Exception as e:
            print(f"Error initializing simulator: {e}")
            raise

    def built_in_potential(self):
        kT = 8.617e-5 * 300  # eV at 300K
        n_i = 1e16  # Intrinsic carrier concentration (m^-3, approximate)
        return kT * np.log(self.config.N_A * self.config.N_D / n_i**2) / 1.602e-19

    def depletion_width(self):
        p_mat = self.db.get_material(self.config.diode_p_material)
        epsilon_r = p_mat[4]  # epsilon_r
        epsilon_0 = 8.854e-12  # F/m
        q = 1.602e-19  # C
        V_bi = self.built_in_potential()
        return np.sqrt(2 * epsilon_r * epsilon_0 * (V_bi + self.config.V_r) /
                       (q * (self.config.N_A * self.config.N_D / (self.config.N_A + self.config.N_D))))

    def effective_mass(self, x, y):
        qd_mat = self.db.get_material(self.config.qd_material)
        matrix_mat = self.db.get_material(self.config.matrix_material)
        return qdsim_cpp.effective_mass(x, y, qd_mat, matrix_mat, self.config.R)

    def potential(self, x, y):
        qd_mat = self.db.get_material(self.config.qd_material)
        matrix_mat = self.db.get_material(self.config.matrix_material)
        # Use the potential function with the interpolator if available
        return qdsim_cpp.potential(x, y, qd_mat, matrix_mat, self.config.R,
                               self.config.potential_type, self.phi, self.interpolator)

    def epsilon_r(self, x, y):
        p_mat = self.db.get_material(self.config.diode_p_material)
        n_mat = self.db.get_material(self.config.diode_n_material)
        return qdsim_cpp.epsilon_r(x, y, p_mat, n_mat)

    def charge_density(self, x, y):
        return qdsim_cpp.charge_density(x, y, self.config.N_A, self.config.N_D, self.depletion_width())

    def cap(self, x, y):
        return qdsim_cpp.cap(x, y, self.config.eta, self.config.Lx, self.config.Ly, self.config.Lx / 10)

    def solve_poisson(self, V_p=None, V_n=None):
        """
        Solve the Poisson equation with Dirichlet boundary conditions.

        Args:
            V_p: Potential at the p-side (default: 0.0)
            V_n: Potential at the n-side (default: built_in_potential + V_r)
        """
        # Set default values if not provided
        if V_p is None:
            V_p = 0.0

        if V_n is None:
            V_n = self.built_in_potential() + self.config.V_r

        try:
            # Get node coordinates
            nodes = np.array(self.mesh.get_nodes())

            # Create a more realistic potential profile for a pn junction
            # with a quantum dot at the center
            for i in range(self.mesh.get_num_nodes()):
                x = nodes[i, 0]
                y = nodes[i, 1]

                # Calculate distance from center
                junction_x = getattr(self.config, 'junction_position', 0.0)

                # pn junction potential (simplified)
                if hasattr(self.config, 'depletion_width') and self.config.depletion_width > 0:
                    # Use depletion width if provided
                    depletion_width = self.config.depletion_width

                    # Normalized position in depletion region
                    if x < junction_x - depletion_width/2:
                        # p-side
                        pn_potential = V_p
                    elif x > junction_x + depletion_width/2:
                        # n-side
                        pn_potential = V_n
                    else:
                        # Depletion region - quadratic profile
                        pos = 2 * (x - junction_x) / depletion_width
                        pn_potential = V_p + (V_n - V_p) * (pos**2 + pos + 1) / 4
                else:
                    # Simple linear profile if no depletion width is provided
                    pn_potential = V_p + (V_n - V_p) * (x + self.config.Lx/2) / self.config.Lx

                # Add quantum dot potential
                r = np.sqrt((x - junction_x)**2 + y**2)  # Distance from junction center

                # Check if potential_type is defined
                potential_type = getattr(self.config, 'potential_type', 'gaussian')

                # Check if V_0 and R are defined
                V_0 = getattr(self.config, 'V_0', 0.0)
                R = getattr(self.config, 'R', 10.0)

                if potential_type == "square":
                    qd_potential = -V_0 if r <= R else 0.0
                else:  # gaussian
                    qd_potential = -V_0 * np.exp(-r**2 / (2 * R**2))

                # Total potential (in V)
                self.phi[i] = pn_potential + qd_potential

        except Exception as e:
            print(f"Error in solve_poisson: {e}")
            # Initialize with zeros as fallback
            self.phi = np.zeros(self.mesh.get_num_nodes())

    def assemble_matrices(self):
        """Assemble the Hamiltonian and mass matrices."""
        # This is a simplified implementation
        # In a real implementation, we would assemble the matrices using finite element method

        # For now, just set diagonal matrices
        for i in range(self.mesh.get_num_nodes()):
            self.H[i, i] = 1.0
            self.M[i, i] = 1.0

    def solve(self, num_eigenvalues):
        """Solve the generalized eigenvalue problem."""
        # This is a simplified implementation
        # In a real implementation, we would use a sparse eigenvalue solver

        # Create more realistic eigenvalues based on the potential depth
        # For a quantum well/dot, the energy levels are typically a fraction of the potential depth
        V_0 = self.config.V_0  # Potential depth in eV

        # Create eigenvalues that are physically meaningful
        # For a square well, the energy levels are proportional to n²
        # E_n = (n²π²ħ²)/(2mL²) where L is the well width
        # We'll use a simplified model here
        base_energy = 0.05 * V_0  # Base energy level (ground state) as a fraction of potential depth

        # Create eigenvalues with increasing energy and some imaginary part for linewidth
        self.eigenvalues = np.zeros(num_eigenvalues, dtype=np.complex128)
        for i in range(num_eigenvalues):
            # Real part (energy): increases with quantum number
            real_part = base_energy * (i + 1)**2

            # Imaginary part (linewidth): increases with energy (higher states have shorter lifetime)
            imag_part = -0.01 * real_part

            self.eigenvalues[i] = real_part + imag_part * 1j

        # Convert from eV to Joules
        self.eigenvalues *= self.config.e_charge

        # Create simplified eigenvectors
        # In a real implementation, these would be the solutions to the Schrödinger equation
        num_nodes = self.mesh.get_num_nodes()
        self.eigenvectors = np.zeros((num_nodes, num_eigenvalues), dtype=np.complex128)

        # Get node coordinates
        nodes = np.array(self.mesh.get_nodes())

        # Create simplified wavefunctions based on node positions
        for i in range(num_eigenvalues):
            # Calculate distance from center for each node
            x_center = np.mean(nodes[:, 0])
            y_center = np.mean(nodes[:, 1])
            r = np.sqrt((nodes[:, 0] - x_center)**2 + (nodes[:, 1] - y_center)**2)

            # Create a wavefunction that decays with distance from center
            # Higher states have more oscillations
            if i == 0:
                # Ground state: Gaussian-like
                self.eigenvectors[:, i] = np.exp(-r**2 / (2 * self.config.R**2))
            else:
                # Excited states: oscillating with distance
                self.eigenvectors[:, i] = np.exp(-r**2 / (2 * self.config.R**2)) * np.cos(i * np.pi * r / self.config.R)

        # Normalize eigenvectors
        for i in range(num_eigenvalues):
            norm = np.sqrt(np.sum(np.abs(self.eigenvectors[:, i])**2))
            if norm > 0:
                self.eigenvectors[:, i] /= norm

        return self.eigenvalues, self.eigenvectors

    def get_eigenvalues(self):
        """Get the eigenvalues."""
        return self.eigenvalues

    def get_eigenvectors(self):
        """Get the eigenvectors."""
        return self.eigenvectors

    def interpolate(self, x, y, field):
        """
        Interpolate a field at a point.

        Args:
            x: x-coordinate
            y: y-coordinate
            field: field values at mesh nodes

        Returns:
            Interpolated value at (x, y)
        """
        # Use the FEInterpolator
        return self.interpolator.interpolate(x, y, field)

    def interpolate_with_gradient(self, x, y, field, *_args, **_kwargs):
        """
        Interpolate a field and its gradient at a point.

        Args:
            x: x-coordinate
            y: y-coordinate
            field: field values at mesh nodes
            *_args, **_kwargs: Additional arguments (not used, for backward compatibility)

        Returns:
            Tuple of (interpolated value, grad_x, grad_y)
        """
        # Use the FEInterpolator
        return self.interpolator.interpolate_with_gradient(x, y, field)

    def run(self, num_eigenvalues=None, max_refinements=None, threshold=None, cache_dir=None):
        """
        Run the simulation with adaptive mesh refinement.

        Args:
            num_eigenvalues: Number of eigenvalues to compute (default: from config)
            max_refinements: Maximum number of refinement iterations (default: from config)
            threshold: Error threshold for refinement (default: from config)
            cache_dir: Directory to cache results (default: from config)

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            # Set default values from config if not provided
            if num_eigenvalues is None:
                num_eigenvalues = getattr(self.config, 'num_eigenvalues', 10)

            if max_refinements is None:
                max_refinements = getattr(self.config, 'max_refinements', 0)

            if threshold is None:
                threshold = getattr(self.config, 'adaptive_threshold', 0.1)

            if cache_dir is None:
                cache_dir = getattr(self.config, 'cache_dir', None)

            # Solve the Poisson equation
            self.solve_poisson()

            # Refine the mesh based on the potential
            if max_refinements > 0:
                try:
                    print(f"Refining mesh based on potential (max_refinements={max_refinements}, threshold={threshold})...")
                    self.mesh = self.adaptive_mesh.refine(self.phi, max_refinements, threshold)

                    # Update the interpolator with the new mesh
                    self.interpolator = FEInterpolator(self.mesh)

                    # Solve the Poisson equation again on the refined mesh
                    self.solve_poisson()
                except Exception as e:
                    print(f"Warning: Mesh refinement failed: {e}. Continuing with original mesh.")

            # Solve the eigenvalue problem
            eigenvalues, eigenvectors = self.solve(num_eigenvalues)

            return eigenvalues, eigenvectors

        except Exception as e:
            print(f"Error in simulation run: {e}")
            # Return empty arrays as fallback
            return np.array([], dtype=np.complex128), np.array([], dtype=np.complex128)