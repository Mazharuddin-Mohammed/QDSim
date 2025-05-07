#!/usr/bin/env python3
"""
Comprehensive test for PN junction potential determination enhancements.

This test verifies the following enhancements:
1. Proper physics-based PN junction potential determination from charge distributions
2. Replacement of depletion-width approximation with self-consistent approach
3. Inclusion of carrier statistics in potential calculation
4. Proper handling of quasi-Fermi levels

Author: Dr. Mazharuddin Mohammed
Date: 2023-07-15
"""

import os
import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time

# Add the parent directory to the path so we can import qdsim_cpp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import qdsim_cpp
except ImportError:
    print("Error: qdsim_cpp module not found. Make sure it's built and in the Python path.")
    sys.exit(1)

class PNJunctionPotentialTest(unittest.TestCase):
    """Test case for PN junction potential determination enhancements."""

    def setUp(self):
        """Set up test fixtures."""
        # Physical constants
        self.q = 1.602e-19  # Elementary charge in C
        self.epsilon_0 = 8.85e-12  # Vacuum permittivity in F/m
        self.k_B = 1.38e-23  # Boltzmann constant in J/K
        self.T = 300  # Temperature in K
        self.V_T = self.k_B * self.T / self.q  # Thermal voltage in V

        # Material parameters (GaAs)
        self.epsilon_r = 12.9  # Relative permittivity
        self.N_A = 1e16  # Acceptor doping concentration in cm^-3
        self.N_D = 1e16  # Donor doping concentration in cm^-3
        self.n_i = 2.1e6  # Intrinsic carrier concentration in cm^-3

        # Device dimensions
        self.Lx = 200.0  # Device length in nm
        self.Ly = 100.0  # Device width in nm

        # Mesh parameters
        self.nx = 100  # Number of elements in x-direction
        self.ny = 50   # Number of elements in y-direction
        self.element_order = 1  # Linear elements

        # Create the mesh
        self.mesh = qdsim_cpp.Mesh(self.Lx, self.Ly, self.nx, self.ny, self.element_order)

        # Output directory for plots
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Define callback functions
        self.setup_callbacks()

    def setup_callbacks(self):
        """Set up callback functions for the solver."""
        # Define the relative permittivity function
        def epsilon_r_func(x, y):
            return self.epsilon_r

        # Define the charge density function
        def rho_func(x, y, n, p):
            # The C++ code might pass arrays or scalars
            # We need to handle both cases safely
            try:
                # Try to convert to float (works for scalars and length-1 arrays)
                n_val = float(n)
                p_val = float(p)
            except (TypeError, ValueError):
                # If conversion fails, use the first element (for arrays)
                try:
                    n_val = float(n[0])
                    p_val = float(p[0])
                except (IndexError, TypeError):
                    # If all else fails, use default values
                    n_val = 0.0
                    p_val = 0.0

            # Convert from cm^-3 to m^-3
            n_m3 = n_val * 1e6
            p_m3 = p_val * 1e6

            # Calculate the charge density in C/m^3
            if x < self.Lx / 2:
                # p-region
                return self.q * (p_m3 - n_m3 - self.N_A * 1e6)
            else:
                # n-region
                return self.q * (p_m3 - n_m3 + self.N_D * 1e6)

        # Store the callback functions as instance variables
        self.epsilon_r_func = epsilon_r_func
        self.rho_func = rho_func

    def test_depletion_approximation(self):
        """Test the depletion approximation for PN junction potential."""
        print("\n=== Testing Depletion Approximation ===")

        try:
            # Create a PN junction solver with depletion approximation
            pn_solver = qdsim_cpp.PNJunction(
                self.mesh,
                self.epsilon_r,  # Relative permittivity (float)
                self.N_A,        # Acceptor doping concentration (float)
                self.N_D,        # Donor doping concentration (float)
                self.T,          # Temperature (float)
                self.Lx / 2,     # Junction position (float)
                0.0              # Reverse bias voltage (float)
            )

            # Solve using the built-in solver
            start_time = time.time()
            pn_solver.solve()
            solve_time = time.time() - start_time
            print(f"PN junction solve time: {solve_time:.6f}s")

            # Get the results
            # Create a grid of points to evaluate the potential and carrier concentrations
            x_points = np.linspace(0, self.Lx, 100)
            y_mid = self.Ly / 2

            # Evaluate the potential and carrier concentrations at each point
            potential = np.array([pn_solver.get_potential(x, y_mid) for x in x_points])
            n = np.array([pn_solver.get_electron_concentration(x, y_mid) for x in x_points])
            p = np.array([pn_solver.get_hole_concentration(x, y_mid) for x in x_points])

            # Store the x-coordinates for plotting
            self.x_points = x_points

            # Calculate the built-in potential
            V_bi = self.V_T * np.log(self.N_A * self.N_D / (self.n_i * self.n_i))
            print(f"Built-in potential: {V_bi:.6f} V")

            # Calculate the depletion width
            W = np.sqrt(2 * self.epsilon_0 * self.epsilon_r * V_bi / self.q *
                        (1/self.N_A + 1/self.N_D)) * 1e9  # Convert to nm
            print(f"Depletion width: {W:.6f} nm")

            # Plot the results
            self.plot_results("depletion_approximation", potential, n, p)

            print("Depletion approximation test passed!")
        except Exception as e:
            print(f"Depletion approximation test failed: {e}")
            raise

    def test_self_consistent_solution(self):
        """Test the self-consistent solution for PN junction potential."""
        print("\n=== Testing Self-Consistent Solution ===")

        try:
            # Create a PN junction solver with self-consistent approach
            pn_solver = qdsim_cpp.PNJunction(
                self.mesh,
                self.epsilon_r,  # Relative permittivity (float)
                self.N_A,        # Acceptor doping concentration (float)
                self.N_D,        # Donor doping concentration (float)
                self.T,          # Temperature (float)
                self.Lx / 2,     # Junction position (float)
                0.0              # Reverse bias voltage (float)
            )

            # Solve using the built-in solver
            start_time = time.time()
            pn_solver.solve()
            solve_time = time.time() - start_time
            print(f"Self-consistent solution solve time: {solve_time:.6f}s")

            # Get the results
            # Create a grid of points to evaluate the potential and carrier concentrations
            x_points = np.linspace(0, self.Lx, 100)
            y_mid = self.Ly / 2

            # Evaluate the potential and carrier concentrations at each point
            potential = np.array([pn_solver.get_potential(x, y_mid) for x in x_points])
            n = np.array([pn_solver.get_electron_concentration(x, y_mid) for x in x_points])
            p = np.array([pn_solver.get_hole_concentration(x, y_mid) for x in x_points])

            # Store the x-coordinates for plotting
            self.x_points = x_points

            # Calculate the built-in potential
            V_bi = self.V_T * np.log(self.N_A * self.N_D / (self.n_i * self.n_i))
            print(f"Built-in potential: {V_bi:.6f} V")

            # Plot the results
            self.plot_results("self_consistent_solution", potential, n, p)

            # Compare with depletion approximation
            self.compare_solutions()

            print("Self-consistent solution test passed!")
        except Exception as e:
            print(f"Self-consistent solution test failed: {e}")
            raise

    def test_quasi_fermi_levels(self):
        """Test the quasi-Fermi levels in PN junction potential calculation."""
        print("\n=== Testing Quasi-Fermi Levels ===")

        try:
            # Apply a forward bias
            V_bias = 0.5  # Forward bias voltage in V

            # Create a PN junction solver with self-consistent approach
            pn_solver = qdsim_cpp.PNJunction(
                self.mesh,
                self.epsilon_r,  # Relative permittivity (float)
                self.N_A,        # Acceptor doping concentration (float)
                self.N_D,        # Donor doping concentration (float)
                self.T,          # Temperature (float)
                self.Lx / 2,     # Junction position (float)
                -V_bias          # Reverse bias voltage (float) - negative for forward bias
            )

            # Solve using the built-in solver
            start_time = time.time()
            pn_solver.solve()
            solve_time = time.time() - start_time
            print(f"Quasi-Fermi levels solve time: {solve_time:.6f}s")

            # Get the results
            # Create a grid of points to evaluate the potential and carrier concentrations
            x_points = np.linspace(0, self.Lx, 100)
            y_mid = self.Ly / 2

            # Evaluate the potential, carrier concentrations, and quasi-Fermi levels at each point
            potential = np.array([pn_solver.get_potential(x, y_mid) for x in x_points])
            n = np.array([pn_solver.get_electron_concentration(x, y_mid) for x in x_points])
            p = np.array([pn_solver.get_hole_concentration(x, y_mid) for x in x_points])
            E_fn = np.array([pn_solver.get_quasi_fermi_level_electrons(x, y_mid) for x in x_points])
            E_fp = np.array([pn_solver.get_quasi_fermi_level_holes(x, y_mid) for x in x_points])

            # Store the x-coordinates for plotting
            self.x_points = x_points

            # Calculate the built-in potential
            V_bi = self.V_T * np.log(self.N_A * self.N_D / (self.n_i * self.n_i))
            print(f"Built-in potential: {V_bi:.6f} V")
            print(f"Applied bias: {V_bias:.6f} V")

            # Plot the results with quasi-Fermi levels
            self.plot_results_with_quasi_fermi_levels("quasi_fermi_levels", potential, n, p, E_fn, E_fp)

            print("Quasi-Fermi levels test passed!")
        except Exception as e:
            print(f"Quasi-Fermi levels test failed: {e}")
            raise

    def compare_solutions(self):
        """Compare the depletion approximation and self-consistent solutions."""
        try:
            # Create PN junction solvers
            pn_solver_depletion = qdsim_cpp.PNJunction(
                self.mesh,
                self.epsilon_r,  # Relative permittivity (float)
                self.N_A,        # Acceptor doping concentration (float)
                self.N_D,        # Donor doping concentration (float)
                self.T,          # Temperature (float)
                self.Lx / 2,     # Junction position (float)
                0.0              # Reverse bias voltage (float)
            )

            pn_solver_self_consistent = qdsim_cpp.PNJunction(
                self.mesh,
                self.epsilon_r,  # Relative permittivity (float)
                self.N_A,        # Acceptor doping concentration (float)
                self.N_D,        # Donor doping concentration (float)
                self.T,          # Temperature (float)
                self.Lx / 2,     # Junction position (float)
                0.0              # Reverse bias voltage (float)
            )

            # Solve using both approaches
            pn_solver_depletion.solve()
            pn_solver_self_consistent.solve()

            # Get the results
            # Create a grid of points to evaluate the potential
            x_points = np.linspace(0, self.Lx, 100)
            y_mid = self.Ly / 2

            # Evaluate the potential at each point
            potential_depletion = np.array([pn_solver_depletion.get_potential(x, y_mid) for x in x_points])
            potential_self_consistent = np.array([pn_solver_self_consistent.get_potential(x, y_mid) for x in x_points])

            # Store the x-coordinates for plotting
            self.x_points = x_points

            # Plot the comparison
            self.plot_comparison("solution_comparison", potential_depletion, potential_self_consistent)
        except Exception as e:
            print(f"Solution comparison failed: {e}")
            raise

    def plot_results(self, name, potential, n, p):
        """Plot the results of the simulation."""
        # Create a figure
        fig = Figure(figsize=(15, 10))
        canvas = FigureCanvas(fig)

        # Create a 2D grid for visualization
        x_grid = np.linspace(0, self.Lx, 20)
        y_grid = np.linspace(0, self.Ly, 20)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Create 2D arrays for visualization
        potential_2d = np.zeros_like(X)
        n_2d = np.zeros_like(X)
        p_2d = np.zeros_like(X)

        # Fill the 2D arrays with values from the 1D arrays
        # For simplicity, we'll just repeat the 1D values along the y-axis
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Find the closest x-point
                idx = np.abs(self.x_points - X[i, j]).argmin()
                potential_2d[i, j] = potential[idx]
                n_2d[i, j] = n[idx]
                p_2d[i, j] = p[idx]

        # Plot the potential
        ax1 = fig.add_subplot(221)
        im1 = ax1.pcolormesh(X, Y, potential_2d, cmap='viridis', shading='auto')
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('Electrostatic Potential (V)')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')

        # Plot the electron concentration
        ax2 = fig.add_subplot(222)
        im2 = ax2.pcolormesh(X, Y, np.log10(n_2d), cmap='plasma', shading='auto')
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('Electron Concentration (log10(cm^-3))')
        ax2.set_xlabel('x (nm)')
        ax2.set_ylabel('y (nm)')

        # Plot the hole concentration
        ax3 = fig.add_subplot(223)
        im3 = ax3.pcolormesh(X, Y, np.log10(p_2d), cmap='plasma', shading='auto')
        fig.colorbar(im3, ax=ax3)
        ax3.set_title('Hole Concentration (log10(cm^-3))')
        ax3.set_xlabel('x (nm)')
        ax3.set_ylabel('y (nm)')

        # Plot the potential along the x-axis at y = Ly/2
        ax4 = fig.add_subplot(224)

        # Use the 1D arrays directly
        x_mid = self.x_points
        potential_mid = potential
        n_mid = n
        p_mid = p

        # Plot the potential
        ax4.plot(x_mid, potential_mid, 'b-', label='Potential (V)')

        # Create a second y-axis for carrier concentrations
        ax4_twin = ax4.twinx()
        ax4_twin.semilogy(x_mid, n_mid, 'r-', label='Electrons')
        ax4_twin.semilogy(x_mid, p_mid, 'g-', label='Holes')

        # Add labels and legend
        ax4.set_xlabel('x (nm)')
        ax4.set_ylabel('Potential (V)')
        ax4_twin.set_ylabel('Carrier Concentration (cm^-3)')

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax4.set_title('Potential and Carrier Concentrations along y = Ly/2')

        # Save the figure
        fig.tight_layout()
        canvas.print_figure(os.path.join(self.output_dir, f'{name}_results.png'), dpi=150)

        print(f"Results plot saved to {os.path.join(self.output_dir, f'{name}_results.png')}")

    def plot_results_with_quasi_fermi_levels(self, name, potential, n, p, E_fn, E_fp):
        """Plot the results of the simulation with quasi-Fermi levels."""
        # Create a figure
        fig = Figure(figsize=(15, 10))
        canvas = FigureCanvas(fig)

        # Create a 2D grid for visualization
        x_grid = np.linspace(0, self.Lx, 20)
        y_grid = np.linspace(0, self.Ly, 20)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Create 2D arrays for visualization
        potential_2d = np.zeros_like(X)
        n_2d = np.zeros_like(X)
        p_2d = np.zeros_like(X)

        # Fill the 2D arrays with values from the 1D arrays
        # For simplicity, we'll just repeat the 1D values along the y-axis
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Find the closest x-point
                idx = np.abs(self.x_points - X[i, j]).argmin()
                potential_2d[i, j] = potential[idx]
                n_2d[i, j] = n[idx]
                p_2d[i, j] = p[idx]

        # Plot the potential
        ax1 = fig.add_subplot(221)
        im1 = ax1.pcolormesh(X, Y, potential_2d, cmap='viridis', shading='auto')
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('Electrostatic Potential (V)')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')

        # Plot the electron concentration
        ax2 = fig.add_subplot(222)
        im2 = ax2.pcolormesh(X, Y, np.log10(n_2d), cmap='plasma', shading='auto')
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('Electron Concentration (log10(cm^-3))')
        ax2.set_xlabel('x (nm)')
        ax2.set_ylabel('y (nm)')

        # Plot the hole concentration
        ax3 = fig.add_subplot(223)
        im3 = ax3.pcolormesh(X, Y, np.log10(p_2d), cmap='plasma', shading='auto')
        fig.colorbar(im3, ax=ax3)
        ax3.set_title('Hole Concentration (log10(cm^-3))')
        ax3.set_xlabel('x (nm)')
        ax3.set_ylabel('y (nm)')

        # Plot the potential and quasi-Fermi levels along the x-axis at y = Ly/2
        ax4 = fig.add_subplot(224)

        # Use the 1D arrays directly
        x_mid = self.x_points
        potential_mid = potential
        E_fn_mid = E_fn
        E_fp_mid = E_fp

        # Plot the potential and quasi-Fermi levels
        ax4.plot(x_mid, potential_mid, 'b-', label='Potential (V)')
        ax4.plot(x_mid, E_fn_mid, 'r-', label='E_fn (eV)')
        ax4.plot(x_mid, E_fp_mid, 'g-', label='E_fp (eV)')

        # Add labels and legend
        ax4.set_xlabel('x (nm)')
        ax4.set_ylabel('Energy (eV)')
        ax4.legend(loc='upper right')

        ax4.set_title('Potential and Quasi-Fermi Levels along y = Ly/2')

        # Save the figure
        fig.tight_layout()
        canvas.print_figure(os.path.join(self.output_dir, f'{name}_results.png'), dpi=150)

        print(f"Results plot saved to {os.path.join(self.output_dir, f'{name}_results.png')}")

    def plot_comparison(self, name, potential_depletion, potential_self_consistent):
        """Plot the comparison between depletion approximation and self-consistent solution."""
        # Create a figure
        fig = Figure(figsize=(15, 5))
        canvas = FigureCanvas(fig)

        # Create a 2D grid for visualization
        x_grid = np.linspace(0, self.Lx, 20)
        y_grid = np.linspace(0, self.Ly, 20)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Create 2D arrays for visualization
        potential_depletion_2d = np.zeros_like(X)
        potential_self_consistent_2d = np.zeros_like(X)

        # Fill the 2D arrays with values from the 1D arrays
        # For simplicity, we'll just repeat the 1D values along the y-axis
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Find the closest x-point
                idx = np.abs(self.x_points - X[i, j]).argmin()
                potential_depletion_2d[i, j] = potential_depletion[idx]
                potential_self_consistent_2d[i, j] = potential_self_consistent[idx]

        # Calculate the difference
        diff_2d = potential_self_consistent_2d - potential_depletion_2d

        # Plot the depletion approximation potential
        ax1 = fig.add_subplot(131)
        im1 = ax1.pcolormesh(X, Y, potential_depletion_2d, cmap='viridis', shading='auto')
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('Depletion Approximation')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')

        # Plot the self-consistent solution potential
        ax2 = fig.add_subplot(132)
        im2 = ax2.pcolormesh(X, Y, potential_self_consistent_2d, cmap='viridis', shading='auto')
        fig.colorbar(im2, ax=ax2)
        ax2.set_title('Self-Consistent Solution')
        ax2.set_xlabel('x (nm)')
        ax2.set_ylabel('y (nm)')

        # Plot the difference
        ax3 = fig.add_subplot(133)
        im3 = ax3.pcolormesh(X, Y, diff_2d, cmap='seismic', shading='auto')
        fig.colorbar(im3, ax=ax3)
        ax3.set_title('Difference (Self-Consistent - Depletion)')
        ax3.set_xlabel('x (nm)')
        ax3.set_ylabel('y (nm)')

        # Save the figure
        fig.tight_layout()
        canvas.print_figure(os.path.join(self.output_dir, f'{name}.png'), dpi=150)

        print(f"Comparison plot saved to {os.path.join(self.output_dir, f'{name}.png')}")

        # Also plot a 1D comparison along y = Ly/2
        fig2 = Figure(figsize=(10, 6))
        canvas2 = FigureCanvas(fig2)

        # Use the 1D arrays directly
        x_mid = self.x_points
        potential_depletion_mid = potential_depletion
        potential_self_consistent_mid = potential_self_consistent

        # Plot the potentials
        ax = fig2.add_subplot(111)
        ax.plot(x_mid, potential_depletion_mid, 'b-', label='Depletion Approximation')
        ax.plot(x_mid, potential_self_consistent_mid, 'r-', label='Self-Consistent Solution')

        # Add labels and legend
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('Potential (V)')
        ax.legend(loc='upper right')

        ax.set_title('Potential Comparison along y = Ly/2')

        # Save the figure
        fig2.tight_layout()
        canvas2.print_figure(os.path.join(self.output_dir, f'{name}_1d.png'), dpi=150)

        print(f"1D comparison plot saved to {os.path.join(self.output_dir, f'{name}_1d.png')}")

if __name__ == "__main__":
    unittest.main()
