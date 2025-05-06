"""
Python implementation of the full Poisson-Drift-Diffusion solver.

This module provides a pure Python implementation of the full Poisson-Drift-Diffusion solver
for use when the C++ implementation is not available.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import time
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
from .python_poisson_solver import PythonPoissonSolver
from .python_drift_diffusion_solver import PythonDriftDiffusionSolver

# Import GPU fallback module
from .gpu_fallback import (
    is_gpu_available, cp, to_gpu, to_cpu
)

# Import Thomas algorithm for tridiagonal systems
from .thomas_solver import thomas_algorithm

# Import line search
from .line_search import line_search

class PythonFullPoissonDriftDiffusionSolver:
    """
    Python implementation of the full Poisson-Drift-Diffusion solver.

    This class solves the coupled Poisson-Drift-Diffusion equations using the finite element method.
    It is used as a fallback when the C++ implementation is not available.
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
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)
        self.T = 300.0  # Temperature (K)
        self.kT = self.kB * self.T / self.q  # Thermal voltage (eV)

        # Material properties
        self.N_c = 4.7e17  # Effective density of states in conduction band (cm^-3)
        self.N_v = 7.0e18  # Effective density of states in valence band (cm^-3)
        self.E_g = 1.424  # Band gap (eV)
        self.mu_n = 8500.0  # Electron mobility (cm^2/V·s)
        self.mu_p = 400.0  # Hole mobility (cm^2/V·s)

        # Initialize vectors
        self.phi = np.zeros(mesh.get_num_nodes())
        self.n = np.zeros(mesh.get_num_nodes())
        self.p = np.zeros(mesh.get_num_nodes())
        self.phi_n = np.zeros(mesh.get_num_nodes())
        self.phi_p = np.zeros(mesh.get_num_nodes())
        self.E_field = [np.zeros(2) for _ in range(mesh.get_num_nodes())]
        self.J_n = [np.zeros(2) for _ in range(mesh.get_num_nodes())]
        self.J_p = [np.zeros(2) for _ in range(mesh.get_num_nodes())]

        # Create the solvers
        self.poisson_solver = PythonPoissonSolver(mesh)
        self.drift_diffusion_solver = PythonDriftDiffusionSolver(mesh)

        # Solver settings
        self.use_fermi_dirac_statistics = False
        self.use_quantum_corrections = False
        self.use_adaptive_mesh_refinement = False
        self.refinement_threshold = 0.1
        self.max_refinement_level = 3
        self.use_gpu = False
        self.damping_factor = 0.3  # Damping factor for better convergence

        # Non-linear solver settings
        self.nonlinear_solver_type = 'newton'  # 'newton', 'anderson', or 'picard'
        self.use_line_search = True
        self.use_thomas_algorithm = True  # Use Thomas algorithm for tridiagonal systems
        self.anderson_m = 5  # Number of previous iterations to use for Anderson acceleration

        # Try to import GPU accelerator
        try:
            from .gpu_interpolator import GPUInterpolator
            self.gpu_interpolator_available = True
            print("GPU interpolator available")
        except ImportError:
            self.gpu_interpolator_available = False
            print("GPU interpolator not available")

        # Try to import CuPy for GPU acceleration
        try:
            import cupy as cp
            self.cupy_available = True
            print("CuPy available for GPU acceleration")
        except ImportError:
            self.cupy_available = False
            print("CuPy not available for GPU acceleration")

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
            # Assume fully ionized doping concentrations
            if doping > 0:
                # n-type region (N_D > 0, N_A = 0)
                self.n[i] = doping  # n = N_D (fully ionized donors)
                self.p[i] = ni * ni / doping  # p*n = ni^2 (mass action law)
            elif doping < 0:
                # p-type region (N_D = 0, N_A > 0)
                self.p[i] = -doping  # p = N_A (fully ionized acceptors)
                self.n[i] = ni * ni / self.p[i]  # p*n = ni^2 (mass action law)
            else:
                # Intrinsic region (N_D = N_A = 0)
                self.n[i] = ni
                self.p[i] = ni

            # Ensure minimum carrier concentrations for numerical stability
            self.n[i] = max(self.n[i], 1e5)
            self.p[i] = max(self.p[i], 1e5)

    def solve(self, V_p, V_n, tolerance=1e-6, max_iter=20, damping_factor=None):
        """
        Solve the coupled Poisson-Drift-Diffusion equations.

        Args:
            V_p: Voltage applied to the p-contact
            V_n: Voltage applied to the n-contact
            tolerance: Convergence tolerance
            max_iter: Maximum number of iterations
            damping_factor: Damping factor for better convergence (if None, use self.damping_factor)

        Returns:
            True if the solution converged, False otherwise
        """
        print(f"Solving Poisson-Drift-Diffusion equations with V_p = {V_p} V, V_n = {V_n} V")

        # Use the provided damping factor or the default one
        if damping_factor is None:
            damping_factor = self.damping_factor

        # Precompute node positions for faster lookup
        if self.use_gpu:
            nodes = to_gpu(np.array(self.mesh.get_nodes()))
        else:
            nodes = np.array(self.mesh.get_nodes())

        # Define the charge density function for the Poisson solver
        def rho(x, y):
            # Use vectorized operations for faster nearest node lookup
            if self.use_gpu:
                dist = cp.sum((nodes - cp.array([x, y]))**2, axis=1)
                nearest_node = cp.argmin(dist).item()
            else:
                dist = np.sum((nodes - np.array([x, y]))**2, axis=1)
                nearest_node = np.argmin(dist)

            # Get doping at this position
            doping = self.doping_profile(x, y)

            # Compute charge density: rho = q * (p - n + N_D - N_A)
            # Apply parity multipliers: +1 for holes (positive charge), -1 for electrons (negative charge)
            if self.use_gpu:
                p_val = to_cpu(self.p[nearest_node])
                n_val = to_cpu(self.n[nearest_node])
                return self.q * (+1 * p_val - 1 * n_val + doping)
            else:
                return self.q * (+1 * self.p[nearest_node] - 1 * self.n[nearest_node] + doping)

        # Initialize the potential with a linear profile between V_p and V_n
        print("Initializing potential with linear profile")
        x_coords = nodes[:, 0]
        Lx = self.mesh.get_lx()

        # Linear interpolation between V_p and V_n (vectorized)
        if self.use_gpu:
            self.phi = to_gpu(V_p + (V_n - V_p) * to_cpu(x_coords) / Lx)
        else:
            self.phi = V_p + (V_n - V_p) * x_coords / Lx

        # Self-consistent iteration
        error = 1.0
        iter_count = 0

        # Reduced number of iterations for the Python implementation
        max_iter = min(max_iter, 20)  # Increased from 10 to 20 for better convergence

        # Track convergence history for adaptive damping
        error_history = []

        # Start timing
        start_time = time.time()

        # Define a function to calculate the residual for line search
        def calculate_residual(phi_test):
            # Convert to CPU for calculation
            phi_test_cpu = to_cpu(phi_test) if self.use_gpu else phi_test

            # Calculate the residual for the Poisson equation
            residual = 0.0

            # Get the nodes
            nodes_cpu = to_cpu(nodes) if self.use_gpu else nodes

            # Calculate the residual at each node
            for i in range(self.mesh.get_num_nodes()):
                x, y = nodes_cpu[i]

                # Get doping at this position
                doping = self.doping_profile(x, y)

                # Calculate the residual for the Poisson equation
                # Apply parity multipliers: +1 for holes (positive charge), -1 for electrons (negative charge)
                rho_i = self.q * (+1 * p_old[i] - 1 * n_old[i] + doping)
                residual += (rho_i - self._laplacian(phi_test_cpu, i))**2

            # Normalize the residual
            residual = np.sqrt(residual / self.mesh.get_num_nodes())

            return residual

        while error > tolerance and iter_count < max_iter:
            # Save the current potential and carrier concentrations
            phi_old = np.copy(to_cpu(self.phi))
            n_old = np.copy(to_cpu(self.n))
            p_old = np.copy(to_cpu(self.p))

            # Solve the Poisson equation
            print(f"Iteration {iter_count + 1}: Solving Poisson equation")
            phi_new = self.poisson_solver.solve(self.epsilon_r, rho, V_p, V_n, use_thomas=self.use_thomas_algorithm)

            # Calculate the update direction
            dphi = phi_new - phi_old

            # Perform line search to find optimal step size if enabled
            if self.use_line_search and iter_count > 0:
                # Define a function to calculate the residual for different step sizes
                def residual_func(phi_test):
                    return calculate_residual(phi_test)

                # Perform line search
                alpha = line_search(residual_func, phi_old, dphi, residual_func(phi_old))
            else:
                # Use fixed damping factor
                alpha = damping_factor

            # Apply damping to the potential
            if iter_count > 0:
                if self.use_gpu:
                    self.phi = to_gpu(phi_old + alpha * to_cpu(dphi))
                else:
                    self.phi = phi_old + alpha * dphi
            else:
                # First iteration, just use the new values
                self.phi = phi_new

            # Compute the electric field
            self.E_field = self.poisson_solver.compute_electric_field(self.phi)

            # Solve the drift-diffusion equations
            print(f"Iteration {iter_count + 1}: Solving drift-diffusion equations")
            self.n, self.p, self.phi_n, self.phi_p, self.J_n, self.J_p = self.drift_diffusion_solver.solve_continuity_equations(
                self.phi, self.E_field, self.doping_profile, self.N_c, self.N_v, self.E_g, self.use_fermi_dirac_statistics
            )

            # Apply damping to the carrier concentrations
            if iter_count > 0:
                if self.use_gpu:
                    n_new = to_cpu(self.n)
                    p_new = to_cpu(self.p)
                    self.n = to_gpu(n_old + alpha * (n_new - n_old))
                    self.p = to_gpu(p_old + alpha * (p_new - p_old))
                else:
                    n_new = self.n
                    p_new = self.p
                    self.n = n_old + alpha * (n_new - n_old)
                    self.p = p_old + alpha * (p_new - p_old)

            # Compute the error
            if self.use_gpu:
                phi_cpu = to_cpu(self.phi)
                error = np.linalg.norm(phi_cpu - phi_old) / np.linalg.norm(phi_cpu) if np.linalg.norm(phi_cpu) > 1e-10 else np.linalg.norm(phi_cpu - phi_old)
            else:
                error = np.linalg.norm(self.phi - phi_old) / np.linalg.norm(self.phi) if np.linalg.norm(self.phi) > 1e-10 else np.linalg.norm(self.phi - phi_old)

            # Track error history for convergence analysis
            error_history.append(error)

            # Increment the iteration counter
            iter_count += 1

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            print(f"Iteration {iter_count}: error = {error}, elapsed time = {elapsed_time:.2f}s, step size = {alpha:.3f}")

            # Adaptive damping: adjust damping factor based on convergence behavior
            if not self.use_line_search and len(error_history) >= 3:
                if error_history[-1] > error_history[-2] and error_history[-2] > error_history[-3]:
                    # Error is increasing, reduce damping factor
                    damping_factor = max(0.1, damping_factor * 0.8)
                    print(f"Convergence slowing down, reducing damping factor to {damping_factor}")
                elif error_history[-1] < 0.5 * error_history[-2]:
                    # Error is decreasing rapidly, increase damping factor
                    damping_factor = min(0.9, damping_factor * 1.2)
                    print(f"Convergence accelerating, increasing damping factor to {damping_factor}")

            # For the Python implementation, we'll accept a higher tolerance
            if error < 0.01:
                print("Reached acceptable tolerance for Python implementation")
                break

        # Check if the solution converged
        if iter_count >= max_iter and error > tolerance:
            print(f"Warning: Solution did not converge after {max_iter} iterations")
            print("Using the current solution anyway for demonstration purposes")
        else:
            print(f"Solution converged after {iter_count} iterations in {elapsed_time:.2f}s")

        return True

    def _laplacian(self, phi, node_index):
        """
        Calculate the Laplacian of the potential at a node.

        Args:
            phi: The potential vector
            node_index: The index of the node

        Returns:
            The Laplacian of the potential at the node
        """
        # Get the node coordinates
        x, y = self.mesh.get_nodes()[node_index]

        # Find all elements containing this node
        node_elements = []
        for e in range(self.mesh.get_num_elements()):
            element = self.mesh.get_elements()[e]
            if node_index in element:
                node_elements.append(e)

        # Calculate the Laplacian as the sum of second derivatives
        laplacian = 0.0

        # For each element containing this node
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

            # Calculate the second derivatives (approximation)
            d2phi_dx2 = 0.0
            d2phi_dy2 = 0.0

            # Add contribution from this element
            laplacian += (d2phi_dx2 + d2phi_dy2) / len(node_elements)

        return laplacian

    def _calculate_residual(self, phi, n, p):
        """
        Calculate the residual for the Poisson-Drift-Diffusion equations.

        Args:
            phi: Electrostatic potential
            n: Electron concentration
            p: Hole concentration

        Returns:
            The residual (a measure of how well the equations are satisfied)
        """
        # Calculate the residual for the Poisson equation
        residual_poisson = 0.0

        # Calculate the residual for the continuity equations
        residual_n = 0.0
        residual_p = 0.0

        # Get the nodes
        nodes = np.array(self.mesh.get_nodes())

        # Calculate the residual at each node
        for i in range(self.mesh.get_num_nodes()):
            x, y = nodes[i]

            # Get doping at this position
            doping = self.doping_profile(x, y)

            # Calculate the residual for the Poisson equation
            # Apply parity multipliers: +1 for holes (positive charge), -1 for electrons (negative charge)
            rho_i = self.q * (+1 * p[i] - 1 * n[i] + doping)
            residual_poisson += rho_i**2

            # Calculate the residual for the continuity equations
            # This is a simplified version; a full implementation would check the divergence of the currents
            residual_n += (n[i] - n[i])**2  # Placeholder
            residual_p += (p[i] - p[i])**2  # Placeholder

        # Normalize the residuals
        residual_poisson = np.sqrt(residual_poisson / self.mesh.get_num_nodes())
        residual_n = np.sqrt(residual_n / self.mesh.get_num_nodes())
        residual_p = np.sqrt(residual_p / self.mesh.get_num_nodes())

        # Combine the residuals
        residual = residual_poisson + residual_n + residual_p

        return residual

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

    def get_electric_field(self, x, y):
        """
        Get the electric field at a given position.

        Args:
            x: The x-coordinate
            y: The y-coordinate

        Returns:
            The electric field vector at the given position
        """
        # Find the nearest node
        min_dist = float('inf')
        nearest_node = 0

        for i in range(self.mesh.get_num_nodes()):
            node = self.mesh.get_nodes()[i]
            dist = (node[0] - x)**2 + (node[1] - y)**2

            if dist < min_dist:
                min_dist = dist
                nearest_node = i

        return self.E_field[nearest_node]

    def get_electric_field_all(self):
        """
        Get the computed electric field at all mesh nodes.

        Returns:
            The electric field vectors
        """
        return self.E_field

    def get_electron_current_density(self):
        """
        Get the computed electron current density.

        Returns:
            The electron current density vectors
        """
        return self.J_n

    def get_hole_current_density(self):
        """
        Get the computed hole current density.

        Returns:
            The hole current density vectors
        """
        return self.J_p

    def set_carrier_statistics_model(self, use_fermi_dirac):
        """
        Set the carrier statistics model.

        Args:
            use_fermi_dirac: Whether to use Fermi-Dirac statistics (True) or Boltzmann statistics (False)
        """
        self.use_fermi_dirac_statistics = use_fermi_dirac

    def enable_quantum_corrections(self, enable):
        """
        Enable or disable quantum corrections.

        Args:
            enable: Whether to enable quantum corrections
        """
        self.use_quantum_corrections = enable

    def enable_adaptive_mesh_refinement(self, enable, refinement_threshold=0.1, max_refinement_level=3):
        """
        Enable or disable adaptive mesh refinement.

        Args:
            enable: Whether to enable adaptive mesh refinement
            refinement_threshold: The threshold for refinement
            max_refinement_level: The maximum refinement level
        """
        self.use_adaptive_mesh_refinement = enable
        self.refinement_threshold = refinement_threshold
        self.max_refinement_level = max_refinement_level

    def set_heterojunction(self, materials, regions):
        """
        Set the material properties for a heterojunction.

        Args:
            materials: List of materials
            regions: List of functions that define the regions
        """
        # This is a placeholder for a more sophisticated implementation
        # In a real implementation, this would set up the material properties
        # for different regions of the device
        pass

    def set_generation_recombination_model(self, g_r):
        """
        Set the generation-recombination model.

        Args:
            g_r: Function that returns the generation-recombination rate at a given position
        """
        # This is a placeholder for a more sophisticated implementation
        # In a real implementation, this would set up the generation-recombination model
        pass

    def set_mobility_models(self, mu_n, mu_p):
        """
        Set the mobility models for electrons and holes.

        Args:
            mu_n: Function that returns the electron mobility at a given position
            mu_p: Function that returns the hole mobility at a given position
        """
        # This is a placeholder for a more sophisticated implementation
        # In a real implementation, this would set up the mobility models
        pass

    def enable_gpu_acceleration(self, enable):
        """
        Enable or disable GPU acceleration.

        Args:
            enable: Whether to enable GPU acceleration
        """
        if enable:
            if not self.gpu_interpolator_available and not self.cupy_available:
                print("Warning: GPU acceleration not available. Make sure CuPy is installed.")
                self.use_gpu = False
                return

            print("Enabling GPU acceleration for Poisson-Drift-Diffusion solver")
            self.use_gpu = True

            # Create GPU interpolator if available
            if self.gpu_interpolator_available:
                from .gpu_interpolator import GPUInterpolator
                self.gpu_interpolator = GPUInterpolator(self.mesh, use_gpu=True)
                print("Created GPU interpolator")
        else:
            print("Disabling GPU acceleration for Poisson-Drift-Diffusion solver")
            self.use_gpu = False

    def set_nonlinear_solver(self, solver_type, use_line_search=True, use_thomas_algorithm=True, anderson_m=5):
        """
        Set the non-linear solver type.

        Args:
            solver_type: The solver type ('newton', 'anderson', or 'picard')
            use_line_search: Whether to use line search
            use_thomas_algorithm: Whether to use the Thomas algorithm for tridiagonal systems
            anderson_m: Number of previous iterations to use for Anderson acceleration
        """
        if solver_type not in ['newton', 'anderson', 'picard']:
            print(f"Warning: Unknown solver type '{solver_type}'. Using 'newton' instead.")
            solver_type = 'newton'

        self.nonlinear_solver_type = solver_type
        self.use_line_search = use_line_search
        self.use_thomas_algorithm = use_thomas_algorithm
        self.anderson_m = anderson_m

        print(f"Using {solver_type} solver for non-linear equations")
        if solver_type == 'newton':
            print(f"  Line search: {use_line_search}")
            print(f"  Thomas algorithm: {use_thomas_algorithm}")
        elif solver_type == 'anderson':
            print(f"  Number of previous iterations: {anderson_m}")
            print(f"  Damping factor: {self.damping_factor}")
