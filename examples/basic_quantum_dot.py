#!/usr/bin/env python3
"""
Basic Quantum Dot Simulation Example

This example demonstrates:
1. Setting up a quantum dot with parabolic confinement
2. Solving the Schrödinger equation
3. Analyzing energy levels and wavefunctions
4. Creating publication-quality visualizations

Run with: python examples/basic_quantum_dot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add QDSim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import qdsim_cython as qdsim
    from qdsim_cython.visualization.wavefunction_plotter import WavefunctionPlotter
except ImportError:
    print("❌ QDSim not found. Please install with: pip install -e .")
    print("   Or run: python setup.py build_ext --inplace")
    sys.exit(1)

def main():
    print("🔬 Basic Quantum Dot Simulation")
    print("=" * 50)
    
    # Physical constants
    HBAR = 1.054571817e-34  # J⋅s
    M_E = 9.1093837015e-31  # kg
    EV_TO_J = 1.602176634e-19  # J/eV
    
    # Material properties (GaAs)
    m_star = 0.067 * M_E  # Effective mass
    
    # Quantum dot parameters
    Lx, Ly = 20e-9, 15e-9  # Domain size
    x0, y0 = Lx/2, Ly/2    # Well center
    well_depth = 0.1 * EV_TO_J  # 100 meV
    
    # Calculate confinement frequency
    omega = np.sqrt(2 * well_depth / (m_star * (Lx/4)**2))
    
    print(f"Quantum dot parameters:")
    print(f"  Size: {Lx*1e9:.1f} × {Ly*1e9:.1f} nm")
    print(f"  Well depth: {well_depth/EV_TO_J*1000:.1f} meV")
    print(f"  Confinement frequency: {omega*1e12:.2f} THz")
    
    # Define material properties
    def m_star_func(x, y):
        """Effective mass function (constant for GaAs)"""
        return m_star
    
    def potential_func(x, y):
        """Parabolic quantum dot potential"""
        # Parabolic confinement
        r_squared = (x - x0)**2 + (y - y0)**2
        return 0.5 * m_star * omega**2 * r_squared
    
    # Create quantum solver
    print(f"\n🔧 Setting up quantum solver...")
    
    try:
        solver = qdsim.FixedOpenSystemSolver(
            nx=40, ny=30,           # Grid resolution
            Lx=Lx, Ly=Ly,          # Domain size
            m_star_func=m_star_func,
            potential_func=potential_func,
            use_open_boundaries=False  # Closed system for basic QD
        )
        print("✅ Solver created successfully")
    except Exception as e:
        print(f"❌ Failed to create solver: {e}")
        print("   Falling back to working example...")
        return create_fallback_example()
    
    # Solve the quantum system
    print(f"🚀 Solving Schrödinger equation...")
    try:
        num_states = 8
        eigenvals, eigenvecs = solver.solve(num_states)
        
        # Convert energies to eV
        energies_eV = np.real(eigenvals) / EV_TO_J
        
        print(f"✅ Found {len(eigenvals)} quantum states")
        print(f"\nEnergy levels:")
        for i, E in enumerate(energies_eV):
            print(f"  E_{i+1}: {E*1000:.2f} meV")
        
    except Exception as e:
        print(f"❌ Solver failed: {e}")
        return create_fallback_example()
    
    # Analytical comparison for 2D harmonic oscillator
    print(f"\n📊 Analytical comparison:")
    hbar_omega_eV = (HBAR * omega) / EV_TO_J
    print(f"  ℏω = {hbar_omega_eV*1000:.2f} meV")
    
    analytical_levels = []
    for nx in range(3):
        for ny in range(3):
            if nx + ny < num_states:
                E_analytical = hbar_omega_eV * (nx + ny + 1)
                analytical_levels.append(E_analytical)
    
    analytical_levels = sorted(analytical_levels)[:num_states]
    
    print(f"  Analytical vs Numerical:")
    for i, (E_num, E_ana) in enumerate(zip(energies_eV, analytical_levels)):
        error = abs(E_num - E_ana) / E_ana * 100
        print(f"    E_{i+1}: {E_num*1000:.2f} meV vs {E_ana*1000:.2f} meV (error: {error:.1f}%)")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    create_basic_visualizations(eigenvals, eigenvecs, solver)
    
    print(f"\n🎉 Basic quantum dot simulation completed successfully!")
    
    return {
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs,
        'energies_eV': energies_eV,
        'analytical_levels': analytical_levels,
        'solver': solver
    }

def create_fallback_example():
    """Create a fallback example using analytical solutions"""
    print("\n🔄 Creating fallback analytical example...")
    
    # Physical constants
    HBAR = 1.054571817e-34
    M_E = 9.1093837015e-31
    EV_TO_J = 1.602176634e-19
    
    # Parameters
    L = 10e-9  # Box size
    m_star = 0.067 * M_E
    
    # Analytical eigenvalues for 2D box
    eigenvals_analytical = []
    for nx in range(1, 4):
        for ny in range(1, 4):
            E = (nx**2 + ny**2) * np.pi**2 * HBAR**2 / (2 * m_star * L**2)
            eigenvals_analytical.append(E)
    
    eigenvals_analytical = sorted(eigenvals_analytical)[:5]
    energies_eV = np.array(eigenvals_analytical) / EV_TO_J
    
    print(f"✅ Analytical solution computed")
    print(f"Energy levels:")
    for i, E in enumerate(energies_eV):
        print(f"  E_{i+1}: {E*1000:.2f} meV")
    
    # Create simple visualization
    create_fallback_visualization(energies_eV)
    
    return {
        'eigenvalues': np.array(eigenvals_analytical),
        'energies_eV': energies_eV,
        'method': 'analytical'
    }

def create_basic_visualizations(eigenvals, eigenvecs, solver):
    """Create basic visualizations"""
    try:
        plotter = WavefunctionPlotter()
        
        # Energy level diagram
        fig1 = plotter.plot_energy_levels(eigenvals, "Quantum Dot Energy Levels")
        print("✅ Energy level plot created")
        
        # Plot first few wavefunctions
        for i in range(min(3, len(eigenvecs[0]))):
            title = f"Quantum Dot State {i+1}"
            fig = plotter.plot_wavefunction_2d(
                solver.nodes_x, solver.nodes_y, 
                eigenvecs[:, i], title
            )
        
        print("✅ Wavefunction plots created")
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("   Creating fallback plots...")
        create_fallback_visualization(np.real(eigenvals) / 1.602176634e-19)

def create_fallback_visualization(energies_eV):
    """Create fallback visualization using matplotlib only"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Energy level diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, E in enumerate(energies_eV):
            ax.hlines(E*1000, 0, 1, colors='blue', linewidth=3)
            ax.text(1.05, E*1000, f'E_{i+1} = {E*1000:.1f} meV', va='center')
        
        ax.set_xlim(0, 1.5)
        ax.set_ylim(min(energies_eV)*1000 - 5, max(energies_eV)*1000 + 5)
        ax.set_ylabel('Energy (meV)')
        ax.set_title('Quantum Dot Energy Levels')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_dot_energy_levels.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("✅ Fallback energy level plot created: quantum_dot_energy_levels.png")
        
    except Exception as e:
        print(f"⚠️  Even fallback visualization failed: {e}")

if __name__ == "__main__":
    results = main()
