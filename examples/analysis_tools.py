#!/usr/bin/env python3
"""
Analysis Tools for QDSim

This script provides tools for analyzing simulation results.

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import qdsim_cpp as qdc
import sys
import os

class SimulationAnalyzer:
    """Tools for analyzing QDSim simulation results."""
    
    def __init__(self, mesh, potential, n_conc, p_conc):
        """Initialize the analyzer with simulation results."""
        self.mesh = mesh
        self.potential = potential
        self.n_conc = n_conc
        self.p_conc = p_conc
        
        # Create interpolators
        self.simple_mesh = qdc.create_simple_mesh(mesh)
        self.interpolator = qdc.SimpleInterpolator(self.simple_mesh)
        
        # Get mesh dimensions
        self.Lx = mesh.get_lx()
        self.Ly = mesh.get_ly()
        
        # Create grid for analysis
        self.nx = 201
        self.ny = 101
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize data arrays
        self.potential_grid = np.zeros((self.ny, self.nx))
        self.electron_conc = np.zeros((self.ny, self.nx))
        self.hole_conc = np.zeros((self.ny, self.nx))
        self.electric_field_x = np.zeros((self.ny, self.nx))
        self.electric_field_y = np.zeros((self.ny, self.nx))
        
        # Interpolate data onto grid
        self._interpolate_data()
    
    def _interpolate_data(self):
        """Interpolate data onto grid."""
        for i in range(self.ny):
            for j in range(self.nx):
                xi, yi = self.X[i, j], self.Y[i, j]
                
                try:
                    self.potential_grid[i, j] = self.interpolator.interpolate(xi, yi, self.potential)
                    self.electron_conc[i, j] = self.interpolator.interpolate(xi, yi, self.n_conc)
                    self.hole_conc[i, j] = self.interpolator.interpolate(xi, yi, self.p_conc)
                except:
                    pass
        
        # Calculate electric field
        for i in range(self.ny):
            for j in range(1, self.nx-1):
                self.electric_field_x[i, j] = -(self.potential_grid[i, j+1] - self.potential_grid[i, j-1]) / (self.x[j+1] - self.x[j-1])
            
        for i in range(1, self.ny-1):
            for j in range(self.nx):
                self.electric_field_y[i, j] = -(self.potential_grid[i+1, j] - self.potential_grid[i-1, j]) / (self.y[i+1] - self.y[i-1])
    
    def find_junction_position(self):
        """Find the position of the P-N junction."""
        # The junction is where the electric field is maximum
        mid_y = self.ny // 2
        E_field_x = self.electric_field_x[mid_y, :]
        
        # Find the peak of the electric field
        peaks, _ = find_peaks(np.abs(E_field_x), height=0.1*np.max(np.abs(E_field_x)))
        
        if len(peaks) > 0:
            # Return the position of the maximum peak
            max_peak = peaks[np.argmax(np.abs(E_field_x[peaks]))]
            return self.x[max_peak]
        else:
            # If no peak is found, return the middle of the domain
            return self.Lx / 2
    
    def calculate_depletion_width(self):
        """Calculate the depletion width of the P-N junction."""
        # The depletion width is the distance between the points where
        # the electric field drops to 10% of its maximum value
        mid_y = self.ny // 2
        E_field_x = self.electric_field_x[mid_y, :]
        
        # Find the junction position
        junction_pos = self.find_junction_position()
        junction_idx = np.argmin(np.abs(self.x - junction_pos))
        
        # Find the maximum electric field
        E_max = np.max(np.abs(E_field_x))
        
        # Find the points where the electric field drops to 10% of its maximum
        threshold = 0.1 * E_max
        
        # Find the left edge of the depletion region
        left_idx = junction_idx
        while left_idx > 0 and np.abs(E_field_x[left_idx]) > threshold:
            left_idx -= 1
        
        # Find the right edge of the depletion region
        right_idx = junction_idx
        while right_idx < len(E_field_x) - 1 and np.abs(E_field_x[right_idx]) > threshold:
            right_idx += 1
        
        # Calculate the depletion width
        depletion_width = self.x[right_idx] - self.x[left_idx]
        
        return depletion_width
    
    def calculate_built_in_potential(self):
        """Calculate the built-in potential of the P-N junction."""
        # The built-in potential is the difference between the potential
        # at the edges of the depletion region
        mid_y = self.ny // 2
        
        # Find the junction position
        junction_pos = self.find_junction_position()
        junction_idx = np.argmin(np.abs(self.x - junction_pos))
        
        # Find the maximum electric field
        E_field_x = self.electric_field_x[mid_y, :]
        E_max = np.max(np.abs(E_field_x))
        
        # Find the points where the electric field drops to 10% of its maximum
        threshold = 0.1 * E_max
        
        # Find the left edge of the depletion region
        left_idx = junction_idx
        while left_idx > 0 and np.abs(E_field_x[left_idx]) > threshold:
            left_idx -= 1
        
        # Find the right edge of the depletion region
        right_idx = junction_idx
        while right_idx < len(E_field_x) - 1 and np.abs(E_field_x[right_idx]) > threshold:
            right_idx += 1
        
        # Calculate the built-in potential
        V_bi = self.potential_grid[mid_y, right_idx] - self.potential_grid[mid_y, left_idx]
        
        return np.abs(V_bi)
    
    def calculate_carrier_densities(self):
        """Calculate the carrier densities in the P and N regions."""
        # Find the junction position
        junction_pos = self.find_junction_position()
        
        # Calculate the average carrier densities in the P and N regions
        mid_y = self.ny // 2
        p_region_mask = self.x < junction_pos
        n_region_mask = self.x > junction_pos
        
        # Average electron concentration in the N region
        n_N = np.mean(self.electron_conc[mid_y, n_region_mask])
        
        # Average hole concentration in the P region
        p_P = np.mean(self.hole_conc[mid_y, p_region_mask])
        
        # Average electron concentration in the P region
        n_P = np.mean(self.electron_conc[mid_y, p_region_mask])
        
        # Average hole concentration in the N region
        p_N = np.mean(self.hole_conc[mid_y, n_region_mask])
        
        return n_N, p_P, n_P, p_N
    
    def calculate_recombination_rate(self):
        """Calculate the recombination rate in the P-N junction."""
        # The recombination rate is proportional to n*p
        recombination_rate = self.electron_conc * self.hole_conc
        
        return recombination_rate
    
    def calculate_current_density(self):
        """Calculate the current density in the P-N junction."""
        # The current density is proportional to the gradient of the carrier concentrations
        # and the electric field
        q = 1.602e-19  # Elementary charge (C)
        
        # Calculate the gradient of the carrier concentrations
        dn_dx = np.zeros_like(self.electron_conc)
        dp_dx = np.zeros_like(self.hole_conc)
        
        for i in range(self.ny):
            for j in range(1, self.nx-1):
                dn_dx[i, j] = (self.electron_conc[i, j+1] - self.electron_conc[i, j-1]) / (self.x[j+1] - self.x[j-1])
                dp_dx[i, j] = (self.hole_conc[i, j+1] - self.hole_conc[i, j-1]) / (self.x[j+1] - self.x[j-1])
        
        # Calculate the current density
        # J = q * (mu_n * n * E + D_n * dn/dx - mu_p * p * E - D_p * dp/dx)
        # where mu_n and mu_p are the electron and hole mobilities
        # and D_n and D_p are the diffusion coefficients
        
        # For simplicity, we'll use constant mobilities and diffusion coefficients
        mu_n = 8500  # Electron mobility (cm^2/V·s)
        mu_p = 400   # Hole mobility (cm^2/V·s)
        
        # Einstein relation: D = mu * kT/q
        kT = 0.0259  # eV at 300K
        D_n = mu_n * kT
        D_p = mu_p * kT
        
        # Calculate the current density
        J_n = q * (mu_n * self.electron_conc * self.electric_field_x + D_n * dn_dx)
        J_p = q * (mu_p * self.hole_conc * self.electric_field_x - D_p * dp_dx)
        
        # Total current density
        J_total = J_n + J_p
        
        return J_total
    
    def plot_band_diagram(self):
        """Plot the band diagram of the P-N junction."""
        # Constants
        q = 1.602e-19  # Elementary charge (C)
        kT = 0.0259  # eV at 300K
        
        # Material parameters for GaAs
        E_g = 1.424  # Band gap (eV)
        chi = 4.07  # Electron affinity (eV)
        
        # Calculate the Fermi levels
        n_N, p_P, n_P, p_N = self.calculate_carrier_densities()
        
        # Effective density of states
        N_c = 4.7e17  # Conduction band (nm^-3)
        N_v = 7.0e18  # Valence band (nm^-3)
        
        # Calculate the Fermi levels
        E_F_n = kT * np.log(n_N / N_c)
        E_F_p = -E_g + kT * np.log(p_P / N_v)
        
        # Find the junction position
        junction_pos = self.find_junction_position()
        junction_idx = np.argmin(np.abs(self.x - junction_pos))
        
        # Calculate the band edges
        mid_y = self.ny // 2
        E_c = np.zeros_like(self.x)
        E_v = np.zeros_like(self.x)
        
        for j in range(self.nx):
            # Convert potential to energy
            V = self.potential_grid[mid_y, j]
            
            # Calculate the conduction band edge
            if self.x[j] < junction_pos:
                E_c[j] = -q * V - E_F_p
            else:
                E_c[j] = -q * V - E_F_n
            
            # Calculate the valence band edge
            E_v[j] = E_c[j] - E_g
        
        # Plot the band diagram
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, E_c, 'b-', linewidth=2, label='E_c')
        plt.plot(self.x, E_v, 'r-', linewidth=2, label='E_v')
        
        # Plot the Fermi levels
        E_F = np.zeros_like(self.x)
        for j in range(self.nx):
            E_F[j] = E_F_p if self.x[j] < junction_pos else E_F_n
        plt.plot(self.x, E_F, 'g--', linewidth=1, label='E_F')
        
        # Add labels and title
        plt.xlabel('Position (nm)')
        plt.ylabel('Energy (eV)')
        plt.title('Band Diagram')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add junction position annotation
        plt.axvline(x=junction_pos, color='k', linestyle='-')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def plot_electric_field(self):
        """Plot the electric field of the P-N junction."""
        # Plot the electric field
        mid_y = self.ny // 2
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.electric_field_x[mid_y, :], 'k-', linewidth=2)
        
        # Add labels and title
        plt.xlabel('Position (nm)')
        plt.ylabel('Electric Field (V/nm)')
        plt.title('Electric Field')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Find the junction position
        junction_pos = self.find_junction_position()
        
        # Add junction position annotation
        plt.axvline(x=junction_pos, color='k', linestyle='-')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def plot_carrier_concentrations(self):
        """Plot the carrier concentrations of the P-N junction."""
        # Plot the carrier concentrations
        mid_y = self.ny // 2
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.x, self.electron_conc[mid_y, :], 'b-', linewidth=2, label='n')
        plt.semilogy(self.x, self.hole_conc[mid_y, :], 'r-', linewidth=2, label='p')
        
        # Add labels and title
        plt.xlabel('Position (nm)')
        plt.ylabel('Carrier Concentration (nm^-3)')
        plt.title('Carrier Concentrations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Find the junction position
        junction_pos = self.find_junction_position()
        
        # Add junction position annotation
        plt.axvline(x=junction_pos, color='k', linestyle='-')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def plot_current_density(self):
        """Plot the current density of the P-N junction."""
        # Calculate the current density
        J_total = self.calculate_current_density()
        
        # Plot the current density
        mid_y = self.ny // 2
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, J_total[mid_y, :], 'k-', linewidth=2)
        
        # Add labels and title
        plt.xlabel('Position (nm)')
        plt.ylabel('Current Density (A/nm^2)')
        plt.title('Current Density')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Find the junction position
        junction_pos = self.find_junction_position()
        
        # Add junction position annotation
        plt.axvline(x=junction_pos, color='k', linestyle='-')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print a summary of the P-N junction properties."""
        # Find the junction position
        junction_pos = self.find_junction_position()
        
        # Calculate the depletion width
        depletion_width = self.calculate_depletion_width()
        
        # Calculate the built-in potential
        V_bi = self.calculate_built_in_potential()
        
        # Calculate the carrier densities
        n_N, p_P, n_P, p_N = self.calculate_carrier_densities()
        
        # Print the summary
        print("P-N Junction Summary")
        print("-------------------")
        print(f"Junction Position: {junction_pos:.2f} nm")
        print(f"Depletion Width: {depletion_width:.2f} nm")
        print(f"Built-in Potential: {V_bi:.4f} V")
        print(f"Electron Concentration in N-region: {n_N:.4e} nm^-3")
        print(f"Hole Concentration in P-region: {p_P:.4e} nm^-3")
        print(f"Electron Concentration in P-region: {n_P:.4e} nm^-3")
        print(f"Hole Concentration in N-region: {p_N:.4e} nm^-3")


def main():
    """Main function."""
    # Check if a file was specified
    if len(sys.argv) < 2:
        print("Usage: python analysis_tools.py <simulation_file.npz>")
        return
    
    # Load simulation results
    sim_file = sys.argv[1]
    if not os.path.exists(sim_file):
        print(f"Error: File {sim_file} not found")
        return
    
    try:
        data = np.load(sim_file)
        mesh = data['mesh'].item()
        potential = data['potential']
        n_conc = data['n_conc']
        p_conc = data['p_conc']
    except:
        print(f"Error: Could not load simulation results from {sim_file}")
        return
    
    # Create analyzer
    analyzer = SimulationAnalyzer(mesh, potential, n_conc, p_conc)
    
    # Print summary
    analyzer.print_summary()
    
    # Plot results
    analyzer.plot_band_diagram()
    analyzer.plot_electric_field()
    analyzer.plot_carrier_concentrations()
    analyzer.plot_current_density()


if __name__ == "__main__":
    main()
