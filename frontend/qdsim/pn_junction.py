"""
P-N Junction Module for QDSim.

This module provides classes and functions for modeling P-N junctions
with physically accurate potentials based on doping concentrations,
quasi-Fermi levels, and self-consistent solutions.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import scipy.constants as const
from scipy.optimize import fsolve

class PNJunction:
    """
    Class for modeling P-N junctions with physically accurate potentials.

    This class calculates the electrostatic potential, carrier concentrations,
    and quasi-Fermi levels in a P-N junction based on doping concentrations,
    material parameters, and applied bias.
    """

    def __init__(self, config):
        """
        Initialize the P-N junction model.

        Args:
            config: Configuration object with P-N junction parameters
        """
        # Physical constants
        self.q = config.e_charge  # Elementary charge (C)
        self.k_B = config.k_B  # Boltzmann constant (J/K)
        self.epsilon_0 = config.epsilon_0  # Vacuum permittivity (F/m)
        self.T = config.T  # Temperature (K)

        # Material parameters
        self.epsilon_r = config.epsilon_r  # Relative permittivity
        self.N_A = config.N_A  # Acceptor concentration (m^-3)
        self.N_D = config.N_D  # Donor concentration (m^-3)
        self.E_g = config.E_g  # Band gap (J)
        self.chi = config.chi  # Electron affinity (J)
        self.n_i = self.calculate_intrinsic_carrier_concentration(config)  # Intrinsic carrier concentration (m^-3)

        # Junction parameters
        self.junction_position = config.junction_position  # Junction position (m)
        self.V_bi = self.calculate_built_in_potential()  # Built-in potential (V)
        self.V_r = config.V_r  # Reverse bias (V)
        self.V_total = self.V_bi + self.V_r  # Total potential across the junction (V)

        # Calculate depletion width
        self.W = self.calculate_depletion_width()  # Total depletion width (m)
        self.W_p = self.W * self.N_D / (self.N_A + self.N_D)  # P-side depletion width (m)
        self.W_n = self.W * self.N_A / (self.N_A + self.N_D)  # N-side depletion width (m)

        # Quasi-Fermi levels
        self.E_F_p = -self.k_B * self.T * np.log(self.N_A / self.n_i)  # Quasi-Fermi level in p-region (J)
        self.E_F_n = self.k_B * self.T * np.log(self.N_D / self.n_i)  # Quasi-Fermi level in n-region (J)

        # Potential profile parameters
        self.transition_width = 2e-9  # Width of the smooth transition region (m)

    def calculate_intrinsic_carrier_concentration(self, config):
        """
        Calculate the intrinsic carrier concentration.

        Args:
            config: Configuration object with material parameters

        Returns:
            Intrinsic carrier concentration (m^-3)
        """
        # Effective density of states
        m_star = config.m_star  # Effective mass
        m_0 = config.m_e  # Electron mass
        h = 6.626e-34  # Planck constant (J·s)

        N_c = 2 * (2 * np.pi * m_star * self.k_B * self.T / h**2)**(3/2)  # Conduction band
        N_v = 2 * (2 * np.pi * m_star * self.k_B * self.T / h**2)**(3/2)  # Valence band

        # Intrinsic carrier concentration
        n_i = np.sqrt(N_c * N_v) * np.exp(-self.E_g / (2 * self.k_B * self.T))

        return n_i

    def calculate_built_in_potential(self):
        """
        Calculate the built-in potential of the P-N junction.

        Returns:
            Built-in potential (V)
        """
        # Built-in potential from doping concentrations
        V_bi = (self.k_B * self.T / self.q) * np.log(self.N_A * self.N_D / self.n_i**2)

        return V_bi

    def calculate_depletion_width(self):
        """
        Calculate the depletion width of the P-N junction.

        Returns:
            Depletion width (m)
        """
        # Depletion width from depletion approximation
        W = np.sqrt(2 * self.epsilon_0 * self.epsilon_r * self.V_total / self.q *
                   (1/self.N_A + 1/self.N_D))

        return W

    def potential(self, x, y):
        """
        Calculate the electrostatic potential at a given position.

        This function calculates the electrostatic potential using a physically
        accurate model for a P-N junction in reverse bias, which has a step-like
        potential profile at the junction interface.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Electrostatic potential (J)
        """
        # Distance from junction
        d = x - self.junction_position

        # Step-like potential profile for reverse bias
        if d < -self.W_p:
            # P-side outside depletion region
            V = 0
        elif d > self.W_n:
            # N-side outside depletion region
            V = -self.V_total
        else:
            # Inside depletion region
            # For reverse bias, we have a step-like potential at the junction
            # with a small transition region for numerical stability
            transition_width = 1e-9  # 1 nm transition width

            if d < -transition_width:
                # P-side depletion region
                V = 0
            elif d > transition_width:
                # N-side depletion region
                V = -self.V_total
            else:
                # Smooth transition at the junction interface
                alpha = (d + transition_width) / (2 * transition_width)
                V = -self.V_total * alpha

        # Convert from V to J
        return V * self.q

    def electron_concentration(self, x, y):
        """
        Calculate the electron concentration at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Electron concentration (m^-3)
        """
        # Potential in V
        V = self.potential(x, y) / self.q

        # Distance from junction
        d = x - self.junction_position

        if d < 0:
            # P-side
            n = self.n_i**2 / self.N_A * np.exp(self.q * V / (self.k_B * self.T))
        else:
            # N-side
            n = self.N_D * np.exp(-(self.V_total + V) * self.q / (self.k_B * self.T))

        return n

    def hole_concentration(self, x, y):
        """
        Calculate the hole concentration at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Hole concentration (m^-3)
        """
        # Potential in V
        V = self.potential(x, y) / self.q

        # Distance from junction
        d = x - self.junction_position

        if d < 0:
            # P-side
            p = self.N_A * np.exp(-V * self.q / (self.k_B * self.T))
        else:
            # N-side
            p = self.n_i**2 / self.N_D * np.exp((self.V_total + V) * self.q / (self.k_B * self.T))

        return p

    def charge_density(self, x, y):
        """
        Calculate the charge density at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Charge density (C/m^3)
        """
        # Distance from junction
        d = x - self.junction_position

        if d < -self.W_p or d > self.W_n:
            # Outside depletion region - charge neutrality
            rho = 0
        elif d < 0:
            # P-side depletion region - ionized acceptors
            rho = -self.q * self.N_A
        else:
            # N-side depletion region - ionized donors
            rho = self.q * self.N_D

        return rho

    def electric_field(self, x, y):
        """
        Calculate the electric field at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Electric field vector (V/m)
        """
        # Distance from junction
        d = x - self.junction_position

        if d < -self.W_p or d > self.W_n:
            # Outside depletion region
            E_x = 0
        elif d < 0:
            # P-side depletion region
            E_x = self.q * self.N_A * (d + self.W_p) / (self.epsilon_0 * self.epsilon_r)
        else:
            # N-side depletion region
            E_x = -self.q * self.N_D * (self.W_n - d) / (self.epsilon_0 * self.epsilon_r)

        # No field in y-direction
        E_y = 0

        return np.array([E_x, E_y])

    def conduction_band_edge(self, x, y):
        """
        Calculate the conduction band edge at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Conduction band edge (J)
        """
        # Potential in J
        V = self.potential(x, y)

        # Conduction band edge
        E_c = -self.chi - V

        return E_c

    def valence_band_edge(self, x, y):
        """
        Calculate the valence band edge at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Valence band edge (J)
        """
        # Conduction band edge
        E_c = self.conduction_band_edge(x, y)

        # Valence band edge
        E_v = E_c - self.E_g

        return E_v

    def quasi_fermi_level_electrons(self, x, y):
        """
        Calculate the quasi-Fermi level for electrons at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Quasi-Fermi level for electrons (J)
        """
        # Distance from junction
        d = x - self.junction_position

        # Smooth transition between p and n regions
        if d < -self.W_p:
            # P-side outside depletion region
            E_Fn = self.E_F_p
        elif d > self.W_n:
            # N-side outside depletion region
            E_Fn = self.E_F_n
        else:
            # Inside depletion region - linear interpolation
            alpha = (d + self.W_p) / (self.W_p + self.W_n)
            E_Fn = self.E_F_p + alpha * (self.E_F_n - self.E_F_p)

        return E_Fn

    def quasi_fermi_level_holes(self, x, y):
        """
        Calculate the quasi-Fermi level for holes at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Quasi-Fermi level for holes (J)
        """
        # Distance from junction
        d = x - self.junction_position

        # Smooth transition between p and n regions
        if d < -self.W_p:
            # P-side outside depletion region
            E_Fp = self.E_F_p
        elif d > self.W_n:
            # N-side outside depletion region
            E_Fp = self.E_F_n
        else:
            # Inside depletion region - linear interpolation
            alpha = (d + self.W_p) / (self.W_p + self.W_n)
            E_Fp = self.E_F_p + alpha * (self.E_F_n - self.E_F_p)

        return E_Fp

    def update_bias(self, V_r):
        """
        Update the reverse bias and recalculate dependent parameters.

        Args:
            V_r: Reverse bias (V)
        """
        self.V_r = V_r
        self.V_total = self.V_bi + self.V_r
        self.W = self.calculate_depletion_width()
        self.W_p = self.W * self.N_D / (self.N_A + self.N_D)
        self.W_n = self.W * self.N_A / (self.N_A + self.N_D)

    def update_doping(self, N_A, N_D):
        """
        Update the doping concentrations and recalculate dependent parameters.

        Args:
            N_A: Acceptor concentration (m^-3)
            N_D: Donor concentration (m^-3)
        """
        self.N_A = N_A
        self.N_D = N_D
        self.V_bi = self.calculate_built_in_potential()
        self.V_total = self.V_bi + self.V_r
        self.W = self.calculate_depletion_width()
        self.W_p = self.W * self.N_D / (self.N_A + self.N_D)
        self.W_n = self.W * self.N_A / (self.N_A + self.N_D)
        self.E_F_p = -self.k_B * self.T * np.log(self.N_A / self.n_i)
        self.E_F_n = self.k_B * self.T * np.log(self.N_D / self.n_i)


class SelfConsistentPNJunction(PNJunction):
    """
    Class for self-consistent P-N junction modeling.

    This class extends the PNJunction class with self-consistent
    solution of the Poisson equation using a numerical solver.
    """

    def __init__(self, config):
        """
        Initialize the self-consistent P-N junction model.

        Args:
            config: Configuration object with P-N junction parameters
        """
        super().__init__(config)

        # Import the Poisson solver
        from qdsim.poisson_solver import PoissonSolver1D

        # Discretization parameters
        self.nx = 1001  # Number of points in x-direction
        self.x_min = config.junction_position - 3 * self.W_p  # Minimum x-coordinate
        self.x_max = config.junction_position + 3 * self.W_n  # Maximum x-coordinate

        # Initialize the Poisson solver
        self.poisson_solver = PoissonSolver1D(self.x_min, self.x_max, self.nx, self.epsilon_r)

        # Set boundary conditions
        self.poisson_solver.set_boundary_conditions(0, -self.V_total)

        # Initialize potential and carrier concentrations
        self.x_grid = self.poisson_solver.x
        self.V_grid = np.zeros(self.nx)
        self.E_grid = np.zeros(self.nx)

        # Solve Poisson equation with fixed charge density (depletion approximation)
        self.solve_poisson_depletion()

    def solve_poisson_depletion(self):
        """
        Solve the Poisson equation using the depletion approximation.

        This method calculates the potential by solving Poisson's equation
        with a fixed charge density based on the depletion approximation.
        """
        # Calculate charge density using depletion approximation
        rho = np.zeros(self.nx)

        for i, x in enumerate(self.x_grid):
            # Distance from junction
            d = x - self.junction_position

            # Charge density in depletion approximation
            if d < -self.W_p:
                # P-side outside depletion region (charge neutrality)
                rho[i] = 0
            elif d > self.W_n:
                # N-side outside depletion region (charge neutrality)
                rho[i] = 0
            elif d < 0:
                # P-side depletion region (ionized acceptors)
                rho[i] = -self.q * self.N_A
            else:
                # N-side depletion region (ionized donors)
                rho[i] = self.q * self.N_D

        # Set charge density in Poisson solver
        self.poisson_solver.set_charge_density(rho)

        # Solve Poisson equation
        self.V_grid = self.poisson_solver.solve()

        # Calculate electric field
        self.E_grid = self.poisson_solver.get_electric_field()

        print(f"Solved Poisson equation with depletion approximation")

    def potential_interpolated(self, x, y):
        """
        Interpolate the potential at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Interpolated potential (J)
        """
        # Check if x is within the grid
        if x < self.x_min or x > self.x_max:
            # Use analytical solution outside the grid
            return super().potential(x, y)

        # Find the grid indices
        i = int((x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1))
        i = max(0, min(i, self.nx - 2))

        # Linear interpolation
        dx = self.x_grid[i+1] - self.x_grid[i]
        alpha = (x - self.x_grid[i]) / dx
        V = (1 - alpha) * self.V_grid[i] + alpha * self.V_grid[i+1]

        # Convert from V to J
        return V * self.q

    def electron_concentration(self, x, y):
        """
        Calculate the electron concentration at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Electron concentration (m^-3)
        """
        # Get the potential in V
        V = self.potential_interpolated(x, y) / self.q

        # Distance from junction
        d = x - self.junction_position

        # Electron concentration
        if d < -self.W_p:
            # P-side outside depletion region
            n = self.n_i**2 / self.N_A
        elif d > self.W_n:
            # N-side outside depletion region
            n = self.N_D
        elif d < 0:
            # P-side depletion region
            n = self.n_i**2 / self.N_A * np.exp(self.q * V / (self.k_B * self.T))
        else:
            # N-side depletion region
            n = self.N_D * np.exp(-(self.V_total + V) * self.q / (self.k_B * self.T))

        return n

    def hole_concentration(self, x, y):
        """
        Calculate the hole concentration at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Hole concentration (m^-3)
        """
        # Get the potential in V
        V = self.potential_interpolated(x, y) / self.q

        # Distance from junction
        d = x - self.junction_position

        # Hole concentration
        if d < -self.W_p:
            # P-side outside depletion region
            p = self.N_A
        elif d > self.W_n:
            # N-side outside depletion region
            p = self.n_i**2 / self.N_D
        elif d < 0:
            # P-side depletion region
            p = self.N_A * np.exp(-V * self.q / (self.k_B * self.T))
        else:
            # N-side depletion region
            p = self.n_i**2 / self.N_D * np.exp((self.V_total + V) * self.q / (self.k_B * self.T))

        return p

    def electric_field_interpolated(self, x, y):
        """
        Interpolate the electric field at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Interpolated electric field (V/m)
        """
        # Check if x is within the grid
        if x < self.x_min or x > self.x_max:
            # Use analytical solution outside the grid
            return super().electric_field(x, y)[0]

        # Find the grid indices
        i = int((x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1))
        i = max(0, min(i, self.nx - 2))

        # Linear interpolation
        dx = self.x_grid[i+1] - self.x_grid[i]
        alpha = (x - self.x_grid[i]) / dx
        E = (1 - alpha) * self.E_grid[i] + alpha * self.E_grid[i+1]

        return np.array([E, 0])



    def electron_concentration_interpolated(self, x, y):
        """
        Interpolate the electron concentration at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Interpolated electron concentration (m^-3)
        """
        # Check if x is within the grid
        if x < self.x_min or x > self.x_max:
            # Use analytical solution outside the grid
            return super().electron_concentration(x, y)

        # Find the grid indices
        i = int((x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1))
        i = max(0, min(i, self.nx - 2))

        # Linear interpolation
        dx = self.x_grid[i+1] - self.x_grid[i]
        alpha = (x - self.x_grid[i]) / dx
        n = (1 - alpha) * self.n_grid[i] + alpha * self.n_grid[i+1]

        return n

    def hole_concentration_interpolated(self, x, y):
        """
        Interpolate the hole concentration at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Interpolated hole concentration (m^-3)
        """
        # Check if x is within the grid
        if x < self.x_min or x > self.x_max:
            # Use analytical solution outside the grid
            return super().hole_concentration(x, y)

        # Find the grid indices
        i = int((x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1))
        i = max(0, min(i, self.nx - 2))

        # Linear interpolation
        dx = self.x_grid[i+1] - self.x_grid[i]
        alpha = (x - self.x_grid[i]) / dx
        p = (1 - alpha) * self.p_grid[i] + alpha * self.p_grid[i+1]

        return p

    def electric_field_interpolated(self, x, y):
        """
        Interpolate the electric field at a given position.

        Args:
            x: x-coordinate (m)
            y: y-coordinate (m)

        Returns:
            Interpolated electric field (V/m)
        """
        # Check if x is within the grid
        if x < self.x_min or x > self.x_max:
            # Use analytical solution outside the grid
            return super().electric_field(x, y)[0]

        # Find the grid indices
        i = int((x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1))
        i = max(0, min(i, self.nx - 2))

        # Linear interpolation
        dx = self.x_grid[i+1] - self.x_grid[i]
        alpha = (x - self.x_grid[i]) / dx
        E = (1 - alpha) * self.E_grid[i] + alpha * self.E_grid[i+1]

        return np.array([E, 0])
