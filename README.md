# QDSim - Advanced Quantum Dot Simulator

[![Documentation Status](https://readthedocs.org/projects/qdsimx/badge/?version=latest)](https://qdsimx.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/qdsim/qdsim/workflows/CI/badge.svg)](https://github.com/qdsim/qdsim/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A state-of-the-art quantum dot simulator for semiconductor nanostructures featuring advanced finite element methods, open quantum systems, GPU acceleration, and comprehensive visualization capabilities.

## 🚀 Key Features

### **Quantum Physics Engine**
- **Open Quantum Systems**: Complex eigenvalues with finite lifetimes using Complex Absorbing Potentials (CAP)
- **Self-Consistent Solvers**: Coupled Poisson-Schrödinger equations for realistic device physics
- **Advanced Eigensolvers**: Multiple algorithms (ARPACK, FEAST, Jacobi-Davidson) for different problem types
- **Dirac Delta Normalization**: Proper treatment of scattering states in open systems

### **High-Performance Computing**
- **Cython Backend**: Optimized C-level performance with Python accessibility
- **GPU Acceleration**: CUDA-based parallel computing with automatic CPU fallback
- **Memory Management**: Advanced memory optimization for large-scale simulations
- **Parallel Architecture**: MPI+OpenMP+CUDA hybrid parallelization

### **Advanced Numerical Methods**
- **Finite Element Method (FEM)**: Adaptive mesh refinement and error estimation
- **Complex Boundary Conditions**: Open boundaries for realistic device modeling
- **Material Interface Handling**: Smooth transitions between different semiconductor regions
- **Numerical Stability**: Conservative schemes and robust error handling

### **Comprehensive Visualization**
- **Interactive 3D Plotting**: Real-time visualization of wavefunctions and potentials
- **Energy Level Diagrams**: Complex eigenvalue analysis with lifetime information
- **Device Structure Visualization**: Material composition and potential landscapes
- **Publication-Quality Plots**: High-resolution figures for scientific publications

## 🎯 Applications

- **Quantum Dots in p-n Junctions**: Chromium QDs in InGaAs diodes under bias
- **Open Quantum Systems**: Electron injection and extraction in semiconductor devices
- **Resonant Tunneling**: Complex eigenvalue analysis of tunneling structures
- **Heterojunctions**: Band alignment and carrier confinement studies
- **Device Optimization**: Parameter sweeps and performance analysis

## 📦 Installation

### Prerequisites
```bash
# System dependencies
sudo apt-get install python3-dev python3-matplotlib python3-numpy python3-scipy

# Optional: CUDA for GPU acceleration
sudo apt-get install nvidia-cuda-toolkit
```

### Quick Install
```bash
git clone https://github.com/your-username/QDSim.git
cd QDSim
pip install -e .
```

### Development Install
```bash
git clone https://github.com/your-username/QDSim.git
cd QDSim
python3 -m venv qdsim_env
source qdsim_env/bin/activate
pip install -e .[dev]
```

## 🚀 Quick Start

### Basic Quantum Dot Simulation
```python
import qdsim
import numpy as np

# Define material properties
def m_star_func(x, y):
    return 0.067 * 9.1093837015e-31  # InGaAs effective mass

def potential_func(x, y):
    # Quantum well potential
    if 5e-9 < x < 15e-9:
        return -0.06 * 1.602176634e-19  # -60 meV well
    return 0.0

# Create open system solver
solver = qdsim.FixedOpenSystemSolver(
    nx=8, ny=6, 
    Lx=25e-9, Ly=20e-9,
    m_star_func=m_star_func,
    potential_func=potential_func,
    use_open_boundaries=True
)

# Apply open system physics
solver.apply_open_system_boundary_conditions()
solver.apply_dirac_delta_normalization()
solver.configure_device_specific_solver('quantum_well')

# Solve for eigenvalues and eigenvectors
eigenvals, eigenvecs = solver.solve(num_states=5)

# Analyze complex eigenvalues
for i, E in enumerate(eigenvals):
    E_eV = np.real(E) / 1.602176634e-19
    if np.imag(E) != 0:
        lifetime_fs = 1.054571817e-34 / (2 * abs(np.imag(E))) * 1e15
        print(f"E_{i+1}: {E_eV:.6f} eV, τ = {lifetime_fs:.1f} fs")
```

### Advanced Visualization
```python
from qdsim.visualization import WavefunctionPlotter

# Create plotter
plotter = WavefunctionPlotter()

# Plot energy levels with lifetime information
plotter.plot_energy_levels(eigenvals, "Open System Energy Levels")

# Plot 2D wavefunction
plotter.plot_wavefunction_2d(
    solver.nodes_x, solver.nodes_y, 
    eigenvecs[0], "Ground State Wavefunction"
)

# Comprehensive analysis plot
plotter.plot_comprehensive_analysis(
    solver.nodes_x, solver.nodes_y,
    eigenvals, eigenvecs,
    potential_func, m_star_func,
    "Complete Quantum System Analysis"
)
```

## 📚 Documentation

Comprehensive documentation is available at **[qdsimx.readthedocs.io](https://qdsimx.readthedocs.io)**

### Documentation Sections
- **[Installation Guide](https://qdsimx.readthedocs.io/en/latest/installation.html)**: Complete setup instructions for all platforms
- **[Quick Start](https://qdsimx.readthedocs.io/en/latest/quickstart.html)**: Get running in minutes with working examples
- **[Theory Guide](https://qdsimx.readthedocs.io/en/latest/theory/)**: Quantum mechanics formulation and numerical methods
- **[Cython Migration](https://qdsimx.readthedocs.io/en/latest/enhancements/cython_migration.html)**: Performance optimization through Cython backend
- **[Memory Management](https://qdsimx.readthedocs.io/en/latest/enhancements/memory_management.html)**: Advanced memory optimization techniques
- **[GPU Acceleration](https://qdsimx.readthedocs.io/en/latest/enhancements/gpu_acceleration.html)**: CUDA-based parallel computing
- **[Open Systems](https://qdsimx.readthedocs.io/en/latest/enhancements/open_systems.html)**: Complex eigenvalue theory and implementation
- **[User Guide](https://qdsimx.readthedocs.io/en/latest/user_guide/)**: Step-by-step tutorials and examples
- **[API Reference](https://qdsimx.readthedocs.io/en/latest/api/)**: Complete function and class documentation
- **[Developer Guide](https://qdsimx.readthedocs.io/en/latest/developer/)**: Contributing and extending QDSim

## 🔬 Scientific Background

QDSim implements state-of-the-art quantum mechanical simulations based on:

- **Time-Independent Schrödinger Equation**: `Ĥψ = Eψ` with complex boundary conditions
- **Finite Element Discretization**: Weak formulation with adaptive mesh refinement
- **Open System Theory**: Complex eigenvalue problems for finite lifetimes
- **Self-Consistent Field Theory**: Coupled Poisson-Schrödinger equations

For detailed theoretical background, see our [Theory Documentation](https://qdsimx.readthedocs.io/en/latest/theory/).

## 🚀 Recent Enhancements

QDSim has undergone major enhancements documented in chronological order:

1. **[Cython Migration](https://qdsimx.readthedocs.io/en/latest/enhancements/cython_migration.html)** - Complete backend migration to Cython for C-level performance
2. **[Memory Management](https://qdsimx.readthedocs.io/en/latest/enhancements/memory_management.html)** - Advanced memory optimization and RAII-based resource management
3. **[GPU Acceleration](https://qdsimx.readthedocs.io/en/latest/enhancements/gpu_acceleration.html)** - CUDA-based parallel computing with automatic fallback
4. **[Open System Implementation](https://qdsimx.readthedocs.io/en/latest/enhancements/open_systems.html)** - Complex eigenvalue theory and finite lifetime physics

Each enhancement includes theoretical formulation, implementation details, and validation results.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and guidelines
- Code style and testing requirements
- How to submit pull requests
- Reporting bugs and requesting features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Finite Element Methods**: Based on FEniCS and deal.II methodologies
- **Quantum Theory**: Following Griffiths and Sakurai quantum mechanics
- **Numerical Methods**: Inspired by Trefethen and Bau numerical linear algebra
- **Open Source Community**: Built with NumPy, SciPy, Matplotlib, and Cython

## 📞 Support

- **Documentation**: [qdsimx.readthedocs.io](https://qdsimx.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/QDSim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/QDSim/discussions)
- **Email**: mazharuddin.mohammed.official@gmail.com

---

**QDSim**: Advancing quantum device simulation through open-source collaboration.
