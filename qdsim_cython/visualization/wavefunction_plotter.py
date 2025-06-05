#!/usr/bin/env python3
"""
Wavefunction Plotting and Visualization

This module provides comprehensive visualization capabilities for quantum
wavefunctions, probability densities, and device structures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, List, Optional, Dict, Any
import time

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WavefunctionPlotter:
    """
    Comprehensive wavefunction visualization class
    """
    
    def __init__(self, figsize=(12, 8), dpi=100):
        """Initialize the plotter"""
        self.figsize = figsize
        self.dpi = dpi
        self.color_schemes = {
            'quantum': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'plasma': plt.cm.plasma,
            'viridis': plt.cm.viridis,
            'coolwarm': plt.cm.coolwarm,
            'seismic': plt.cm.seismic
        }
        
        print("🎨 Wavefunction plotter initialized")
    
    def plot_wavefunction_2d(self, nodes_x, nodes_y, wavefunction, title="Wavefunction", 
                            save_path=None, show_colorbar=True, color_scheme='viridis'):
        """Plot 2D wavefunction"""
        
        print(f"🎨 Plotting 2D wavefunction: {title}")
        
        # Reshape wavefunction to 2D grid
        nx = len(np.unique(nodes_x))
        ny = len(np.unique(nodes_y))
        
        # Create regular grid
        x_unique = np.sort(np.unique(nodes_x))
        y_unique = np.sort(np.unique(nodes_y))
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Interpolate wavefunction onto grid
        from scipy.interpolate import griddata
        
        # Real part
        psi_real = np.real(wavefunction)
        Z_real = griddata((nodes_x, nodes_y), psi_real, (X, Y), method='cubic', fill_value=0)
        
        # Imaginary part (if complex)
        if np.any(np.imag(wavefunction) != 0):
            psi_imag = np.imag(wavefunction)
            Z_imag = griddata((nodes_x, nodes_y), psi_imag, (X, Y), method='cubic', fill_value=0)
            
            # Create subplot for complex wavefunction
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=self.dpi)
            
            # Real part
            im1 = ax1.contourf(X*1e9, Y*1e9, Z_real, levels=50, cmap=color_scheme)
            ax1.set_title(f'{title} - Real Part')
            ax1.set_xlabel('x (nm)')
            ax1.set_ylabel('y (nm)')
            if show_colorbar:
                plt.colorbar(im1, ax=ax1)
            
            # Imaginary part
            im2 = ax2.contourf(X*1e9, Y*1e9, Z_imag, levels=50, cmap=color_scheme)
            ax2.set_title(f'{title} - Imaginary Part')
            ax2.set_xlabel('x (nm)')
            ax2.set_ylabel('y (nm)')
            if show_colorbar:
                plt.colorbar(im2, ax=ax2)
            
            # Probability density
            Z_prob = np.abs(Z_real + 1j*Z_imag)**2
            im3 = ax3.contourf(X*1e9, Y*1e9, Z_prob, levels=50, cmap='hot')
            ax3.set_title(f'{title} - Probability Density')
            ax3.set_xlabel('x (nm)')
            ax3.set_ylabel('y (nm)')
            if show_colorbar:
                plt.colorbar(im3, ax=ax3)
        
        else:
            # Real wavefunction only
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
            
            # Wavefunction
            im1 = ax1.contourf(X*1e9, Y*1e9, Z_real, levels=50, cmap=color_scheme)
            ax1.set_title(f'{title} - Wavefunction')
            ax1.set_xlabel('x (nm)')
            ax1.set_ylabel('y (nm)')
            if show_colorbar:
                plt.colorbar(im1, ax=ax1)
            
            # Probability density
            Z_prob = Z_real**2
            im2 = ax2.contourf(X*1e9, Y*1e9, Z_prob, levels=50, cmap='hot')
            ax2.set_title(f'{title} - Probability Density')
            ax2.set_xlabel('x (nm)')
            ax2.set_ylabel('y (nm)')
            if show_colorbar:
                plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_wavefunction_3d(self, nodes_x, nodes_y, wavefunction, title="3D Wavefunction",
                            save_path=None, elevation=30, azimuth=45):
        """Plot 3D wavefunction surface"""
        
        print(f"🎨 Plotting 3D wavefunction: {title}")
        
        # Create regular grid
        x_unique = np.sort(np.unique(nodes_x))
        y_unique = np.sort(np.unique(nodes_y))
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Interpolate wavefunction
        from scipy.interpolate import griddata
        
        if np.any(np.imag(wavefunction) != 0):
            # Complex wavefunction - plot probability density
            psi_prob = np.abs(wavefunction)**2
            Z = griddata((nodes_x, nodes_y), psi_prob, (X, Y), method='cubic', fill_value=0)
            title_suffix = " - Probability Density"
        else:
            # Real wavefunction
            Z = griddata((nodes_x, nodes_y), np.real(wavefunction), (X, Y), method='cubic', fill_value=0)
            title_suffix = ""
        
        # Create 3D plot
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(X*1e9, Y*1e9, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Contour lines at bottom
        ax.contour(X*1e9, Y*1e9, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
        
        ax.set_title(f'{title}{title_suffix}')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('ψ' if title_suffix == "" else '|ψ|²')
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Add colorbar
        plt.colorbar(surf, ax=ax, shrink=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_energy_levels(self, eigenvalues, title="Energy Levels", save_path=None):
        """Plot energy level diagram"""
        
        print(f"🎨 Plotting energy levels: {title}")
        
        # Convert to eV
        EV_TO_J = 1.602176634e-19
        energies_eV = np.real(eigenvalues) / EV_TO_J
        
        # Check for complex eigenvalues (finite lifetimes)
        has_complex = np.any(np.imag(eigenvalues) != 0)
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # Plot energy levels
        for i, E in enumerate(energies_eV):
            color = 'red' if has_complex and abs(np.imag(eigenvalues[i])) > 1e-25 else 'blue'
            linestyle = '--' if has_complex and abs(np.imag(eigenvalues[i])) > 1e-25 else '-'
            
            ax.hlines(E, 0, 1, colors=color, linestyles=linestyle, linewidth=3)
            ax.text(1.05, E, f'E_{i+1} = {E:.3f} eV', va='center', fontsize=10)
            
            # Add lifetime information for complex eigenvalues
            if has_complex and abs(np.imag(eigenvalues[i])) > 1e-25:
                gamma = abs(np.imag(eigenvalues[i]))
                lifetime = 1.054571817e-34 / (2 * gamma) * 1e15  # in fs
                ax.text(1.05, E-0.01, f'τ = {lifetime:.1f} fs', va='center', 
                       fontsize=8, style='italic', color='red')
        
        ax.set_xlim(0, 1.5)
        ax.set_ylim(min(energies_eV) - 0.05, max(energies_eV) + 0.05)
        ax.set_ylabel('Energy (eV)')
        ax.set_title(title)
        ax.set_xticks([])
        
        # Add legend
        if has_complex:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', lw=3, label='Bound states'),
                Line2D([0], [0], color='red', lw=3, linestyle='--', label='Resonant states')
            ]
            ax.legend(handles=legend_elements)
        
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_device_structure(self, nodes_x, nodes_y, potential_func, m_star_func, 
                             title="Device Structure", save_path=None):
        """Plot device structure showing potential and effective mass"""
        
        print(f"🎨 Plotting device structure: {title}")
        
        # Create regular grid
        x_unique = np.sort(np.unique(nodes_x))
        y_unique = np.sort(np.unique(nodes_y))
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Calculate potential and effective mass
        potential = np.zeros_like(X)
        m_star = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                potential[i, j] = potential_func(X[i, j], Y[i, j])
                m_star[i, j] = m_star_func(X[i, j], Y[i, j])
        
        # Convert units
        EV_TO_J = 1.602176634e-19
        M_E = 9.1093837015e-31
        
        potential_eV = potential / EV_TO_J
        m_star_relative = m_star / M_E
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # Potential plot
        im1 = ax1.contourf(X*1e9, Y*1e9, potential_eV, levels=50, cmap='RdBu_r')
        ax1.set_title('Potential Energy')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Energy (eV)')
        
        # Effective mass plot
        im2 = ax2.contourf(X*1e9, Y*1e9, m_star_relative, levels=50, cmap='viridis')
        ax2.set_title('Effective Mass')
        ax2.set_xlabel('x (nm)')
        ax2.set_ylabel('y (nm)')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('m*/m₀')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_comprehensive_analysis(self, nodes_x, nodes_y, eigenvalues, eigenvectors,
                                   potential_func, m_star_func, title="Comprehensive Analysis",
                                   save_path=None):
        """Create comprehensive analysis plot with multiple panels"""
        
        print(f"🎨 Creating comprehensive analysis: {title}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Energy levels (top left)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_energy_levels_subplot(eigenvalues, ax1)
        
        # Device structure - potential (top middle)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_potential_subplot(nodes_x, nodes_y, potential_func, ax2)
        
        # Device structure - effective mass (top right)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_mass_subplot(nodes_x, nodes_y, m_star_func, ax3)
        
        # Wavefunction plots (bottom row)
        num_states = min(3, len(eigenvectors))
        for i in range(num_states):
            ax = plt.subplot(2, 3, 4 + i)
            self._plot_wavefunction_subplot(nodes_x, nodes_y, eigenvectors[i], 
                                          f'ψ_{i+1}', ax)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def _plot_energy_levels_subplot(self, eigenvalues, ax):
        """Helper function for energy levels subplot"""
        EV_TO_J = 1.602176634e-19
        energies_eV = np.real(eigenvalues) / EV_TO_J
        
        for i, E in enumerate(energies_eV):
            ax.hlines(E, 0, 1, colors='blue', linewidth=2)
            ax.text(1.05, E, f'E_{i+1}', va='center', fontsize=8)
        
        ax.set_xlim(0, 1.2)
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Energy Levels')
        ax.set_xticks([])
    
    def _plot_potential_subplot(self, nodes_x, nodes_y, potential_func, ax):
        """Helper function for potential subplot"""
        x_unique = np.sort(np.unique(nodes_x))
        y_unique = np.sort(np.unique(nodes_y))
        X, Y = np.meshgrid(x_unique, y_unique)
        
        potential = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                potential[i, j] = potential_func(X[i, j], Y[i, j])
        
        EV_TO_J = 1.602176634e-19
        im = ax.contourf(X*1e9, Y*1e9, potential/EV_TO_J, levels=20, cmap='RdBu_r')
        ax.set_title('Potential')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
    
    def _plot_mass_subplot(self, nodes_x, nodes_y, m_star_func, ax):
        """Helper function for effective mass subplot"""
        x_unique = np.sort(np.unique(nodes_x))
        y_unique = np.sort(np.unique(nodes_y))
        X, Y = np.meshgrid(x_unique, y_unique)
        
        m_star = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                m_star[i, j] = m_star_func(X[i, j], Y[i, j])
        
        M_E = 9.1093837015e-31
        im = ax.contourf(X*1e9, Y*1e9, m_star/M_E, levels=20, cmap='viridis')
        ax.set_title('Effective Mass')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
    
    def _plot_wavefunction_subplot(self, nodes_x, nodes_y, wavefunction, title, ax):
        """Helper function for wavefunction subplot"""
        x_unique = np.sort(np.unique(nodes_x))
        y_unique = np.sort(np.unique(nodes_y))
        X, Y = np.meshgrid(x_unique, y_unique)
        
        from scipy.interpolate import griddata
        
        if np.any(np.imag(wavefunction) != 0):
            psi_prob = np.abs(wavefunction)**2
            Z = griddata((nodes_x, nodes_y), psi_prob, (X, Y), method='cubic', fill_value=0)
        else:
            Z = griddata((nodes_x, nodes_y), np.real(wavefunction), (X, Y), method='cubic', fill_value=0)
        
        im = ax.contourf(X*1e9, Y*1e9, Z, levels=20, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')

def test_wavefunction_plotter():
    """Test the wavefunction plotter"""
    print("🎨 Testing Wavefunction Plotter")
    print("=" * 50)
    
    # Create test data
    x = np.linspace(0, 20e-9, 20)
    y = np.linspace(0, 15e-9, 15)
    X, Y = np.meshgrid(x, y)
    
    nodes_x = X.flatten()
    nodes_y = Y.flatten()
    
    # Create test wavefunction (Gaussian)
    x0, y0 = 10e-9, 7.5e-9
    sigma = 3e-9
    
    wavefunction = np.exp(-((nodes_x - x0)**2 + (nodes_y - y0)**2) / (2 * sigma**2))
    
    # Test eigenvalues
    eigenvalues = np.array([-0.1, -0.05, -0.02]) * 1.602176634e-19  # in Joules
    eigenvectors = [wavefunction, wavefunction * 0.8, wavefunction * 0.6]
    
    # Define test physics
    def potential_func(x, y):
        if 5e-9 < x < 15e-9:
            return -0.1 * 1.602176634e-19  # -100 meV well
        return 0.0
    
    def m_star_func(x, y):
        return 0.067 * 9.1093837015e-31
    
    # Create plotter
    plotter = WavefunctionPlotter()
    
    # Test 2D plot
    print("\n1. Testing 2D wavefunction plot...")
    fig1 = plotter.plot_wavefunction_2d(nodes_x, nodes_y, wavefunction, "Test Wavefunction 2D")
    
    # Test energy levels
    print("\n2. Testing energy level plot...")
    fig2 = plotter.plot_energy_levels(eigenvalues, "Test Energy Levels")
    
    # Test device structure
    print("\n3. Testing device structure plot...")
    fig3 = plotter.plot_device_structure(nodes_x, nodes_y, potential_func, m_star_func, "Test Device")
    
    print("\n✅ Wavefunction plotter test completed!")
    return True

if __name__ == "__main__":
    test_wavefunction_plotter()
