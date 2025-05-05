# 2D Quantum Dot Simulator

This is a high-performance 2D Quantum Dot (QD) Simulator implemented in C++ and Python, designed to solve the time-independent Schrödinger equation for a quantum dot in a 2D domain. The simulator uses the Finite Element Method (FEM) with selectable linear (P1), quadratic (P2), or cubic (P3) basis functions, adaptive mesh refinement, MPI parallelization, delta normalization for continuum states, and visualization capabilities. It supports both square and Gaussian potential profiles, making it suitable for studying quantum confinement and scattering phenomena in semiconductor nanostructures.

## Table of Contents
1. [Features](#features)
2. [Physical Model and Equations](#physical-model-and-equations)
3. [Solution Approach](#solution-approach)
4. [Directory Structure](#directory-structure)
5. [Dependencies](#dependencies)
6. [Build Instructions](#build-instructions)
7. [Run Instructions](#run-instructions)
8. [Usage Examples](#usage-examples)
9. [Testing](#testing)
10. [Extensibility and Future Work](#extensibility-and-future-work)
11. [Contributing](#contributing)
12. [License](#license)

## Features
- **Finite Element Method (FEM)**:
  - Supports linear (P1, 3 nodes), quadratic (P2, 6 nodes), and cubic (P3, 10 nodes) triangular elements.
  - Higher-order basis functions for improved accuracy, especially for smooth potentials.
- **Adaptive Mesh Refinement**:
  - Dynamically refines the mesh based on wavefunction gradient error, focusing on regions near the QD.
  - Uses red refinement with MPI-parallelized node and element updates.
- **MPI Parallelization**:
  - Distributes matrix assembly and mesh refinement across multiple processes for scalability.
- **Delta Normalization**:
  - Normalizes continuum states using asymptotic amplitude, suitable for scattering problems.
- **Potential Models**:
  - Square potential: Sharp confinement for idealized QD.
  - Gaussian potential: Smooth confinement for realistic QD.
- **Complex Absorbing Potential (CAP)**:
  - Implements CAP to handle outgoing waves in open systems.
- **Full Poisson-Drift-Diffusion Solver**:
  - Solves the coupled Poisson-Drift-Diffusion equations for realistic device simulations.
  - Properly models carrier statistics and transport in semiconductor devices.
  - Self-consistent iteration scheme for accurate solutions.
  - Supports both Boltzmann and Fermi-Dirac statistics.
- **Visualization**:
  - Python-based visualization of wavefunction density, potential, carrier concentrations, and electric field.
  - Highlights P2 midpoints (blue) and P3 nodes (green) for higher-order elements.
  - 3D visualization of potentials and wavefunctions.
- **GPU Acceleration**:
  - Accelerates matrix operations and eigensolvers using GPU.
  - Supports higher-order elements for better accuracy.
- **Caching**:
  - Stores refined meshes to disk, reducing redundant computations.
- **GUI and CLI**:
  - Command-line interface (CLI) for batch simulations.
  - PySide6-based graphical user interface (GUI) for interactive use.
- **Testing**:
  - Comprehensive unit tests for backend (Catch2) and frontend (pytest).

## Physical Model and Equations

The simulator solves the **time-independent Schrödinger equation** in 2D for a quantum dot:

\[
-\frac{\hbar^2}{2} \nabla \cdot \left( \frac{1}{m^*(x,y)} \nabla \psi(x,y) \right) + V(x,y) \psi(x,y) + i \eta(x,y) \psi(x,y) = E \psi(x,y)
\]

Where:
- \( \psi(x,y) \): Wavefunction.
- \( E \): Energy eigenvalue (in Joules).
- \( \hbar \): Reduced Planck’s constant (\( 1.054 \times 10^{-34} \, \text{J·s} \)).
- \( m^*(x,y) \): Position-dependent effective mass (in kg).
- \( V(x,y) \): Potential energy (in Joules).
- \( \eta(x,y) \): Complex absorbing potential (CAP) for outgoing waves (in Joules).

### Effective Mass
The effective mass is modeled as:

\[
m^*(x,y) =
\begin{cases}
m_{\text{Cr}} & \text{if } \sqrt{x^2 + y^2} \leq R \\
m_{\text{AlGaAs}} & \text{otherwise}
\end{cases}
\]

Where:
- \( m_{\text{Cr}} \): Effective mass inside the QD (e.g., \( 0.1 m_e \), where \( m_e = 9.11 \times 10^{-31} \, \text{kg} \)).
- \( m_{\text{AlGaAs}} \): Effective mass in the barrier (e.g., \( 0.09 m_e \)).
- \( R \): QD radius (e.g., 10 nm).

### Potential
The potential \( V(x,y) \) supports two models:
1. **Square Potential**:
\[
V(x,y) =
\begin{cases}
0 & \text{if } \sqrt{x^2 + y^2} \leq R \\
V_0 & \text{if } \sqrt{x^2 + y^2} > R \text{ and } |x|, |y| \leq W/2 \\
V_{\text{bi}} & \text{otherwise}
\end{cases}
\]
2. **Gaussian Potential**:
\[
V(x,y) = V_0 \exp\left(-\frac{x^2 + y^2}{2 R^2}\right)
\]
Where:
- \( V_0 \): Potential height (e.g., \( 0.5 \, \text{eV} = 0.5 \times 1.602 \times 10^{-19} \, \text{J} \)).
- \( V_{\text{bi}} \): Barrier potential (e.g., \( 1.5 \, \text{eV} \)).
- \( W \): Device width (e.g., 100 nm).

### Complex Absorbing Potential (CAP)
The CAP is applied near domain boundaries to absorb outgoing waves:

\[
\eta(x,y) =
\begin{cases}
\eta_0 \left( \frac{|x| - L_x/2}{d} \right)^2 & \text{if } |x| > L_x/2 - d \\
\eta_0 \left( \frac{|y| - L_y/2}{d} \right)^2 & \text{if } |y| > L_y/2 - d \\
0 & \text{otherwise}
\end{cases}
\]

Where:
- \( \eta_0 \): CAP strength (e.g., \( 0.1 \, \text{eV} \)).
- \( L_x, L_y \): Domain dimensions (e.g., 100 nm).
- \( d \): CAP region width (typically \( L_x/10 \)).

### Poisson-Drift-Diffusion Model
The simulator also solves the coupled Poisson-Drift-Diffusion equations for realistic device simulations:

#### Poisson Equation
\[
\nabla \cdot (\epsilon_r \epsilon_0 \nabla \phi) = -\rho
\]

Where:
- \( \phi \): Electrostatic potential (in V).
- \( \epsilon_r \): Relative permittivity.
- \( \epsilon_0 \): Vacuum permittivity (\( 8.85 \times 10^{-14} \, \text{F/cm} \)).
- \( \rho \): Charge density (\( \rho = q(p - n + N_D - N_A) \)).
- \( q \): Elementary charge (\( 1.602 \times 10^{-19} \, \text{C} \)).
- \( n, p \): Electron and hole concentrations (in \( \text{cm}^{-3} \)).
- \( N_D, N_A \): Donor and acceptor concentrations (in \( \text{cm}^{-3} \)).

#### Drift-Diffusion Equations
\[
\nabla \cdot \mathbf{J}_n = q(R - G)
\]
\[
\nabla \cdot \mathbf{J}_p = -q(R - G)
\]

Where:
- \( \mathbf{J}_n, \mathbf{J}_p \): Electron and hole current densities.
- \( R \): Recombination rate.
- \( G \): Generation rate.

The current densities are given by:
\[
\mathbf{J}_n = q\mu_n n \nabla \phi_n
\]
\[
\mathbf{J}_p = -q\mu_p p \nabla \phi_p
\]

Where:
- \( \mu_n, \mu_p \): Electron and hole mobilities.
- \( \phi_n, \phi_p \): Quasi-Fermi potentials for electrons and holes.

#### Carrier Statistics
The carrier concentrations are related to the quasi-Fermi potentials by:
\[
n = N_c \mathcal{F}_{1/2}\left(\frac{\phi_n - \phi}{kT}\right)
\]
\[
p = N_v \mathcal{F}_{1/2}\left(\frac{\phi - \phi_p - E_g}{kT}\right)
\]

Where:
- \( N_c, N_v \): Effective densities of states in the conduction and valence bands.
- \( \mathcal{F}_{1/2} \): Fermi-Dirac integral of order 1/2.
- \( kT \): Thermal voltage (\( 0.0259 \, \text{eV} \) at 300K).
- \( E_g \): Band gap energy.

### Weak Form
The weak form of the Schrödinger equation, used for FEM, is:

\[
\int_{\Omega} \frac{\hbar^2}{2 m^*(x,y)} \nabla \psi \cdot \nabla v \, d\Omega + \int_{\Omega} (V(x,y) + i \eta(x,y)) \psi v \, d\Omega = E \int_{\Omega} \psi v \, d\Omega
\]

Where \( v \) is a test function, and \( \Omega \) is the 2D domain.

### Delta Normalization
For continuum states, the wavefunction is delta-normalized using the asymptotic amplitude:

\[
\psi(\mathbf{r}) \approx A \frac{e^{i k r}}{\sqrt{r}}, \quad k = \sqrt{\frac{2 m^* E}{\hbar^2}}
\]

The normalization constant \( A \) is computed in the far field, and the normalized wavefunction is:

\[
\psi_{\text{norm}}(\mathbf{r}) = \frac{\psi(\mathbf{r})}{A}
\]

## Solution Approach

The simulator employs the **Finite Element Method (FEM)** to discretize and solve the Schrödinger equation, with the following components:

### 1. Finite Element Discretization
- **Elements**:
  - **P1 (Linear)**: 3 nodes (vertices), linear basis functions: \( N_i = \lambda_i \).
  - **P2 (Quadratic)**: 6 nodes (3 vertices + 3 edge midpoints), basis functions:
    - Vertex: \( N_i = \lambda_i (2\lambda_i - 1) \)
    - Edge: \( N_{ij} = 4 \lambda_i \lambda_j \)
  - **P3 (Cubic)**: 10 nodes (3 vertices, 6 edge midpoints, 1 centroid), basis functions:
    - Vertex: \( N_i = \frac{1}{2} \lambda_i (3\lambda_i - 1)(3\lambda_i - 2) \)
    - Edge: \( N_{ij,k} = \frac{9}{2} \lambda_i \lambda_j (3\lambda_i - 1) \), \( N_{ij,l} = \frac{9}{2} \lambda_i \lambda_j (3\lambda_j - 1) \)
    - Centroid: \( N_c = 27 \lambda_1 \lambda_2 \lambda_3 \)
- **Quadrature**:
  - P1: 3-point Gaussian quadrature.
  - P2: 7-point Gaussian quadrature.
  - P3: 12-point Gaussian quadrature.
- **Matrix Assembly**:
  - Stiffness matrix: \( H_{ij} = \int_{\Omega} \frac{\hbar^2}{2 m^*} \nabla N_i \cdot \nabla N_j + (V + i \eta) N_i N_j \, d\Omega \)
  - Mass matrix: \( M_{ij} = \int_{\Omega} N_i N_j \, d\Omega \)
  - Assembled in parallel using MPI, with triplets synchronized across processes.

### 2. Adaptive Mesh Refinement
- **Error Estimator**:
  - Computes the gradient norm of the wavefunction: \( \|\nabla \psi\| \).
  - Elements with \( \|\nabla \psi\| > \text{threshold} \) are marked for refinement.
- **Red Refinement**:
  - Splits each marked triangle into four by adding midpoints to edges.
  - Updates P2 and P3 elements by generating new midpoints and centroids.
  - Parallelized with MPI to distribute refinement tasks.
- **Smoothing**:
  - Applies Laplacian smoothing to vertex nodes, preserving boundary nodes and higher-order nodes (midpoints, centroids).
- **Quality Check**:
  - Ensures triangle quality: \( Q = \frac{4\sqrt{3} \cdot \text{area}}{\sum \text{edge lengths}^2} > 0.1 \).
  - Verifies mesh conformity (no hanging nodes).

### 3. Eigenvalue Solver
- Uses the **Spectra** library to solve the generalized eigenvalue problem:
\[
H \psi = E M \psi
\]
- Computes the lowest \( n \) eigenvalues and eigenvectors, where \( n \) is user-specified.

### 4. Delta Normalization
- Extracts the asymptotic amplitude \( A \) from far-field wavefunction values.
- Normalizes the wavefunction: \( \psi_{\text{norm}} = \psi / A \).

### 5. Parallelization
- **MPI**:
  - Distributes element matrix computations across processes.
  - Synchronizes global matrices and refined mesh data.
  - Can be enabled/disabled at both compile time and runtime:
    - Compile time: Use `-DUSE_MPI=ON/OFF` with CMake
    - Runtime: Set `config.use_mpi = True/False` in Python
- **Caching**:
  - Stores refined meshes to disk, keyed by wavefunction norm, to avoid redundant refinements.

### 6. Visualization
- Plots wavefunction density (\( |\psi|^2 \)) using Matplotlib’s `tricontourf`.
- Highlights refined regions and higher-order nodes (P2 midpoints in blue, P3 nodes in green).
- Displays error estimator as a heatmap of gradient norms.

## Directory Structure



qdsim/
├── backend/
│   ├── include/
│   │   ├── mesh.h
│   │   ├── fem.h
│   │   ├── physics.h
│   │   ├── solver.h
│   │   ├── adaptive_mesh.h
│   │   ├── normalization.h
│   │   └── bindings.h
│   ├── src/
│   │   ├── mesh.cpp
│   │   ├── fem.cpp
│   │   ├── physics.cpp
│   │   ├── solver.cpp
│   │   ├── adaptive_mesh.cpp
│   │   ├── normalization.cpp
│   │   └── bindings.cpp
│   └── tests/
│       ├── test_mesh.cpp
│       ├── test_fem.cpp
│       ├── test_physics.cpp
│       ├── test_solver.cpp
│       ├── test_adaptive_mesh.cpp
│       └── test_normalization.cpp
├── frontend/
│   ├── qdsim/
│   │   ├── init.py
│   │   ├── config.py
│   │   ├── simulator.py
│   │   ├── visualization.py
│   │   └── gui.py
│   ├── tests/
│   │   ├── test_config.py
│   │   ├── test_simulator.py
│   │   └── test_visualization.py
│   ├── run_simulation.py
│   └── requirements.txt
├── CMakeLists.txt
└── README.md


## Dependencies
- **C++ Backend**:
  - C++17-compatible compiler (e.g., GCC 7+, Clang 5+)
  - CMake 3.10+
  - MPI (e.g., OpenMPI, MPICH)
  - Eigen3 (linear algebra library)
  - Spectra (eigenvalue solver)
  - Catch2 (testing framework)
- **Python Frontend**:
  - Python 3.8+
  - Pybind11 (C++/Python bindings)
  - NumPy (numerical computations)
  - Matplotlib (visualization)
  - PySide6 (GUI)
  - pytest (testing)

Install dependencies on Ubuntu:
```bash
sudo apt update
sudo apt install build-essential cmake libopenmpi-dev libeigen3-dev python3-dev python3-pip
pip install pybind11 numpy matplotlib pyside6 pytest
```

## Build Instructions

1. Create a build directory:
   ```bash
   mkdir -p build
   cd build
   ```

2. Configure and build the project:
   ```bash
   cmake ..
   make -j4
   ```

3. Install the Python package:
   ```bash
   cd ..
   pip install -e .
   ```

## Run Instructions

### Basic Run

```bash
python run_simulator.py
```

This will run the simulator and save the results to `energy_shift.png`.

### Chromium Quantum Dot in AlGaAs P-N Junction

This example demonstrates the simulation of a chromium quantum dot in an AlGaAs P-N junction. The quantum dot is positioned at the P-N junction interface, and the simulation shows the electrostatic potential, carrier concentrations, and quantum states.

```bash
python examples/python_quantum_dot_pn_junction.py
```

#### Simulation Parameters

- **Device Dimensions**: 200nm x 100nm
- **P-N Junction**: Located at x = 100nm
- **Quantum Dot**: Chromium quantum dot with 5nm radius and 0.5eV depth
- **Doping Concentrations**: 1e17 cm^-3 for both P and N regions
- **Bias Voltage**: 0.0V (equilibrium)

#### Results

The simulation produces the following results:

1. **Electrostatic Potential**: Shows the combined potential of the P-N junction and the quantum dot.
2. **Carrier Concentrations**: Shows the electron and hole concentrations in the device.
3. **Electric Field**: Shows the electric field distribution in the device.
4. **Quantum States**: Shows the energy levels and wavefunctions of the quantum dot.

The results are saved to `qd_pn_potentials_bias_0.0.png` and `qd_pn_carriers_bias_0.0.png`.

### Command-line Interface

QDSim now provides a comprehensive command-line interface for running simulations:

```bash
./qdsim_cli.py --lx 200 --ly 100 --nx 101 --ny 51 --qd-radius 10 --potential-depth 0.5 --potential-type gaussian --num-states 5 --save-plots --analyze
```

You can also use a configuration file:

```bash
./qdsim_cli.py --config config_samples/simple_qd.yaml --save-plots --analyze
```

For batch processing (parameter sweeps):

```bash
./qdsim_cli.py --batch config_samples/batch_qd.yaml --save-plots --analyze
```

Run `./qdsim_cli.py --help` for a complete list of options.

### Configuration Files

QDSim supports both JSON and YAML configuration files. Here's an example YAML configuration:

```yaml
# Domain size in nm
lx: 200
ly: 100

# Mesh parameters
nx: 101
ny: 51
element_order: 1

# Materials
qd_material: InAs
matrix_material: GaAs
diode_p_material: GaAs
diode_n_material: GaAs

# Quantum dot parameters
r: 10  # radius in nm
v_0: 0.5  # potential depth in eV
potential_type: gaussian

# Diode parameters
n_a: 1.0e+24  # acceptor concentration in m^-3
n_d: 1.0e+24  # donor concentration in m^-3
v_r: 0.0  # reverse bias in V

# Solver parameters
tolerance: 1.0e-6
max_iter: 100
use_mpi: false
```

### Batch Processing

For parameter sweeps, you can use a batch configuration file:

```yaml
base_config:
  # Base configuration parameters
  lx: 200
  ly: 100
  # ...

parameter_sweeps:
  - name: QD Radius Sweep
    parameter: r
    values: [5, 7.5, 10, 12.5, 15]

  - name: Potential Depth Sweep
    parameter: v_0
    values: [0.1, 0.2, 0.3, 0.4, 0.5]

  - name: Reverse Bias Sweep
    parameter: v_r
    values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
```

This will run multiple simulations with different parameter values and generate plots showing how the energy levels change with each parameter.

## Method Name Consistency

The C++ code uses both camelCase and snake_case for method names. For consistency, we've added both versions of the methods in the bindings:

- `getNumNodes()` and `get_num_nodes()`
- `getNumElements()` and `get_num_elements()`
- `getElementOrder()` and `get_element_order()`

## Python Bindings

The C++ code is exposed to Python using pybind11. The bindings are defined in `backend/include/bindings.h`.

Important headers for proper conversion:
- `<pybind11/pybind11.h>`: Core pybind11 functionality
- `<pybind11/eigen.h>`: Conversion of Eigen types
- `<pybind11/stl.h>`: Conversion of STL containers
- `<pybind11/complex.h>`: Conversion of complex numbers
- `<pybind11/functional.h>`: Conversion of function objects