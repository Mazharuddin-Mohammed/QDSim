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
- **Visualization**:
  - Python-based visualization of wavefunction density and error estimators using Matplotlib.
  - Highlights P2 midpoints (blue) and P3 nodes (green) for higher-order elements.
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