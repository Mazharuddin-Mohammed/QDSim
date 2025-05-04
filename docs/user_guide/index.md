# QDSim User Guide

## Introduction

QDSim is a 2D quantum dot simulator that solves the Schrödinger equation for quantum dots in semiconductor heterostructures. It uses the finite element method (FEM) to discretize the Schrödinger equation and compute the energy levels and wavefunctions of the quantum system.

This user guide provides comprehensive documentation for QDSim, including installation instructions, usage examples, and best practices.

## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)
8. [Examples](#examples)

## Installation

### Prerequisites

QDSim requires the following dependencies:

- Python 3.7 or higher
- C++ compiler with C++17 support (GCC 7+, Clang 5+, or MSVC 19.14+)
- CMake 3.10 or higher
- Eigen 3.3 or higher
- pybind11 2.6 or higher

Optional dependencies for advanced features:

- MPI for distributed computing
- CUDA for GPU acceleration
- OpenMP for shared-memory parallelism
- SLEPc for advanced eigensolvers

### Installing from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/username/qdsim.git
   cd qdsim
   ```

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Configure with CMake:
   ```bash
   cmake ..
   ```

   For advanced configurations:
   ```bash
   cmake -DUSE_MPI=ON -DUSE_CUDA=ON -DUSE_OPENMP=ON ..
   ```

4. Build the project:
   ```bash
   make
   ```

5. Install the Python package:
   ```bash
   make install
   ```

### Verifying the Installation

To verify that QDSim is installed correctly, run the following Python code:

```python
import qdsim
print(qdsim.__version__)
```

## Getting Started

### Basic Concepts

QDSim solves the time-independent Schrödinger equation:

\begin{equation}
\left[-\frac{\hbar^2}{2m^*}\nabla^2 + V(\mathbf{r})\right]\psi(\mathbf{r}) = E\psi(\mathbf{r})
\end{equation}

where:
- $\hbar$ is the reduced Planck constant
- $m^*$ is the effective mass of the electron
- $V(\mathbf{r})$ is the potential energy
- $\psi(\mathbf{r})$ is the wavefunction
- $E$ is the energy eigenvalue

The simulator discretizes this equation using the finite element method and solves the resulting generalized eigenvalue problem:

\begin{equation}
H\psi = EM\psi
\end{equation}

where $H$ is the Hamiltonian matrix, $M$ is the mass matrix, $\psi$ is the eigenvector, and $E$ is the eigenvalue.

### Coordinate System

QDSim uses a 2D Cartesian coordinate system with the following conventions:

- The origin (0, 0) is at the center of the simulation domain
- The x-axis is horizontal, with positive values to the right
- The y-axis is vertical, with positive values upward
- All spatial coordinates are in nanometers (nm)
- All energies are in electron volts (eV)

## Basic Usage

### Creating a Simulator

To create a simulator, import the `qdsim` module and create a `Simulator` object:

```python
import qdsim
import numpy as np
import matplotlib.pyplot as plt

# Create a simulator
simulator = qdsim.Simulator()
```

### Creating a Mesh

The first step is to create a mesh for the simulation domain:

```python
# Create a mesh with dimensions 100 nm x 100 nm and 101x101 mesh points
Lx = 100.0  # Domain size in nm
Ly = 100.0
nx = 101    # Number of mesh points
ny = 101
simulator.create_mesh(Lx, Ly, nx, ny)
```

### Setting the Potential

Next, define the potential energy function:

```python
# Define a harmonic oscillator potential
def harmonic_potential(x, y):
    # Parameters
    k = 0.1  # Spring constant in eV/nm²
    
    # Convert to J
    k_J = k * 1.602e-19  # Convert eV/nm² to J/m²
    
    # Calculate potential
    return k_J * (x**2 + y**2)

# Set the potential
simulator.set_potential(harmonic_potential)
```

### Setting the Material

Set the material properties:

```python
# Set the material to GaAs
simulator.set_material("GaAs")
```

### Solving the Schrödinger Equation

Solve the Schrödinger equation to find the energy levels and wavefunctions:

```python
# Solve for the first 10 eigenstates
simulator.solve(num_states=10)
```

### Retrieving Results

Retrieve the computed eigenvalues and eigenvectors:

```python
# Get eigenvalues
eigenvalues = simulator.get_eigenvalues()

# Convert eigenvalues to eV
eigenvalues_eV = np.array(eigenvalues) / 1.602e-19

# Print eigenvalues
print("Eigenvalues (eV):")
for i, e in enumerate(eigenvalues_eV):
    print(f"  E{i} = {e:.6f} eV")
```

### Visualizing Results

Visualize the potential and wavefunctions:

```python
# Plot potential
fig, ax = plt.subplots(figsize=(10, 8))
simulator.plot_potential(ax, convert_to_eV=True)
plt.savefig("potential.png", dpi=300, bbox_inches='tight')

# Plot ground state wavefunction
fig, ax = plt.subplots(figsize=(10, 8))
simulator.plot_wavefunction(ax, state_idx=0)
plt.savefig("wavefunction_0.png", dpi=300, bbox_inches='tight')

# Plot first excited state wavefunction
fig, ax = plt.subplots(figsize=(10, 8))
simulator.plot_wavefunction(ax, state_idx=1)
plt.savefig("wavefunction_1.png", dpi=300, bbox_inches='tight')
```

## Advanced Features

### Custom Materials

You can create custom materials with specific properties:

```python
# Create a custom material
custom_material = {
    "m_e": 0.1,           # Electron effective mass
    "m_h": 0.5,           # Hole effective mass
    "E_g": 1.5,           # Bandgap in eV
    "epsilon_r": 12.0,    # Dielectric constant
    "Delta_E_c": 0.7,     # Conduction band offset in eV
    "Delta_E_v": 0.3      # Valence band offset in eV
}

# Add the custom material to the database
simulator.add_material("CustomMaterial", custom_material)

# Use the custom material
simulator.set_material("CustomMaterial")
```

### Alloy Materials

You can create alloy materials with specific compositions:

```python
# Create an AlGaAs alloy with 30% Al
algaas = simulator.create_alloy("AlAs", "GaAs", 0.3, "Al0.3Ga0.7As")

# Use the alloy material
simulator.set_material("Al0.3Ga0.7As")
```

### Temperature-Dependent Properties

You can get material properties at specific temperatures:

```python
# Get GaAs properties at 77K (liquid nitrogen temperature)
gaas_77K = simulator.get_material_at_temperature("GaAs", 77.0)

# Print bandgap at 77K
print(f"GaAs bandgap at 77K: {gaas_77K['E_g']:.3f} eV")
```

### Spin-Orbit Coupling

You can include spin-orbit coupling effects:

```python
# Enable Rashba spin-orbit coupling
simulator.enable_spin_orbit_coupling(
    spin_orbit_type="rashba",
    rashba_parameter=0.05,  # eV·nm
    dresselhaus_parameter=0.0
)

# Solve with spin-orbit coupling
simulator.solve(num_states=10)
```

### Adaptive Mesh Refinement

You can use adaptive mesh refinement to improve accuracy in regions of interest:

```python
# Enable adaptive mesh refinement
simulator.enable_adaptive_refinement(True)

# Set refinement parameters
simulator.set_refinement_parameters(
    max_refinement_level=3,
    refinement_threshold=0.1,
    coarsening_threshold=0.01
)

# Refine the mesh
simulator.refine_mesh()
```

### Self-Consistent Calculations

You can perform self-consistent calculations that include the Poisson equation:

```python
# Enable self-consistent calculations
simulator.enable_self_consistent(True)

# Set self-consistent parameters
simulator.set_self_consistent_parameters(
    max_iterations=100,
    tolerance=1e-6,
    damping_factor=0.5
)

# Solve self-consistently
simulator.solve_self_consistent()
```

## Performance Optimization

### Mesh Optimization

The mesh resolution significantly affects the accuracy and performance of the simulation. Here are some guidelines:

- Use a finer mesh for regions with rapidly varying potentials
- Use a coarser mesh for regions with slowly varying potentials
- Use adaptive mesh refinement for optimal performance

### Parallel Computing

QDSim supports parallel computing for improved performance:

```python
# Enable MPI parallelism
simulator.enable_mpi(True)

# Enable OpenMP parallelism
simulator.enable_openmp(True, num_threads=4)

# Enable GPU acceleration
simulator.enable_gpu(True, device_id=0)
```

### Memory Optimization

For large simulations, you can use memory-efficient data structures:

```python
# Enable memory-efficient data structures
simulator.enable_memory_efficient(True)
```

## Troubleshooting

### Common Issues

#### Installation Issues

- **CMake not found**: Ensure that CMake is installed and in your PATH
- **Eigen not found**: Ensure that Eigen is installed and CMake can find it
- **Compilation errors**: Ensure that your compiler supports C++17

#### Runtime Issues

- **Memory errors**: Reduce the mesh size or enable memory-efficient data structures
- **Convergence issues**: Adjust the solver parameters or use a different solver
- **Numerical instabilities**: Check the potential function for discontinuities or very large values

### Error Handling

QDSim includes robust error handling with informative error messages:

```python
try:
    simulator.solve(num_states=10)
except Exception as e:
    print(f"Error: {e}")
    # Handle the error
```

### Logging

You can enable logging to get more information about the simulation:

```python
# Enable logging
simulator.enable_logging("qdsim.log", log_level="INFO")
```

## API Reference

For a complete API reference, see the [API Documentation](../api/index.html).

## Examples

For more examples, see the [Examples](../examples/index.html) section.
