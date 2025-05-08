# QDSim Theory Documentation

## Introduction

This document provides a detailed explanation of the theoretical foundations and numerical methods used in QDSim. It covers the physics of quantum dots, the mathematical formulation of the Schrödinger equation, and the numerical techniques used to solve it.

## Table of Contents

1. [Quantum Dots](#quantum-dots)
2. [Schrödinger Equation](#schrödinger-equation)
3. [Finite Element Method](#finite-element-method)
4. [Eigenvalue Problem](#eigenvalue-problem)
5. [Material Properties](#material-properties)
6. [Spin-Orbit Coupling](#spin-orbit-coupling)
7. [Self-Consistent Calculations](#self-consistent-calculations)
8. [Numerical Methods](#numerical-methods)
9. [References](#references)

## Quantum Dots

### Definition and Properties

Quantum dots (QDs) are nanoscale semiconductor structures that confine electrons or holes in all three spatial dimensions. This confinement leads to discrete energy levels, similar to those in atoms, which is why quantum dots are sometimes called "artificial atoms."

The confinement in quantum dots is typically achieved through one of the following mechanisms:

1. **Electrostatic confinement**: Using electric fields from gate electrodes to create a potential well
2. **Strain-induced confinement**: Using strain fields to create a potential well
3. **Heterojunction confinement**: Using the band offsets between different semiconductor materials

### Types of Quantum Dots

QDSim can simulate various types of quantum dots, including:

1. **Electrostatically defined quantum dots**: Created by applying voltages to gate electrodes
2. **Self-assembled quantum dots**: Formed during epitaxial growth due to lattice mismatch
3. **Colloidal quantum dots**: Synthesized in solution using chemical methods
4. **Etched quantum dots**: Created by etching a quantum well structure

### Applications

Quantum dots have numerous applications, including:

1. **Quantum computing**: Quantum dots can be used as qubits for quantum information processing
2. **Single-photon sources**: Quantum dots can emit single photons on demand
3. **Quantum sensors**: Quantum dots can be used for high-sensitivity sensing
4. **Quantum memories**: Quantum dots can store quantum information
5. **Display technology**: Quantum dots can be used in displays for improved color reproduction
6. **Solar cells**: Quantum dots can be used in photovoltaic devices for improved efficiency

## Schrödinger Equation

### Time-Independent Schrödinger Equation

QDSim solves the time-independent Schrödinger equation for a particle in a potential:

\begin{equation}
\left[-\frac{\hbar^2}{2m^*}\nabla^2 + V(\mathbf{r})\right]\psi(\mathbf{r}) = E\psi(\mathbf{r})
\end{equation}

where:
- $\hbar$ is the reduced Planck constant
- $m^*$ is the effective mass of the electron or hole
- $V(\mathbf{r})$ is the potential energy
- $\psi(\mathbf{r})$ is the wavefunction
- $E$ is the energy eigenvalue

In 2D Cartesian coordinates, this equation becomes:

\begin{equation}
\left[-\frac{\hbar^2}{2m^*}\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right) + V(x, y)\right]\psi(x, y) = E\psi(x, y)
\end{equation}

### Effective Mass Approximation

QDSim uses the effective mass approximation, which replaces the free electron mass with an effective mass that accounts for the interaction of the electron with the crystal lattice. The effective mass is typically anisotropic and depends on the material:

\begin{equation}
\frac{1}{m^*} = \frac{1}{\hbar^2}\frac{\partial^2 E(\mathbf{k})}{\partial \mathbf{k}^2}
\end{equation}

where $E(\mathbf{k})$ is the energy-momentum dispersion relation of the material.

### Boundary Conditions

QDSim uses Dirichlet boundary conditions, which set the wavefunction to zero at the boundaries of the simulation domain:

\begin{equation}
\psi(\mathbf{r}) = 0 \quad \text{for} \quad \mathbf{r} \in \partial\Omega
\end{equation}

where $\partial\Omega$ is the boundary of the simulation domain $\Omega$.

## Finite Element Method

### Weak Formulation

To solve the Schrödinger equation numerically, QDSim uses the finite element method (FEM). The first step is to convert the strong form of the equation to a weak form by multiplying by a test function $v$ and integrating over the domain:

\begin{equation}
\int_\Omega \left[-\frac{\hbar^2}{2m^*}\nabla^2\psi + V\psi\right] v \, d\Omega = E \int_\Omega \psi v \, d\Omega
\end{equation}

Using integration by parts and applying the boundary conditions, we get:

\begin{equation}
\int_\Omega \frac{\hbar^2}{2m^*} \nabla\psi \cdot \nabla v \, d\Omega + \int_\Omega V\psi v \, d\Omega = E \int_\Omega \psi v \, d\Omega
\end{equation}

### Discretization

The domain $\Omega$ is discretized into a mesh of triangular elements. The wavefunction $\psi$ and test function $v$ are approximated using basis functions $\phi_i$:

\begin{equation}
\psi(\mathbf{r}) \approx \sum_{j=1}^{N} \psi_j \phi_j(\mathbf{r})
\end{equation}

\begin{equation}
v(\mathbf{r}) = \phi_i(\mathbf{r})
\end{equation}

where $N$ is the number of nodes in the mesh and $\psi_j$ are the coefficients to be determined.

Substituting these approximations into the weak form, we get:

\begin{equation}
\sum_{j=1}^{N} \psi_j \left[ \int_\Omega \frac{\hbar^2}{2m^*} \nabla\phi_j \cdot \nabla\phi_i \, d\Omega + \int_\Omega V\phi_j \phi_i \, d\Omega \right] = E \sum_{j=1}^{N} \psi_j \int_\Omega \phi_j \phi_i \, d\Omega
\end{equation}

This can be written in matrix form:

\begin{equation}
\mathbf{H} \boldsymbol{\psi} = E \mathbf{M} \boldsymbol{\psi}
\end{equation}

where:
- $\mathbf{H}$ is the Hamiltonian matrix with elements $H_{ij} = \int_\Omega \frac{\hbar^2}{2m^*} \nabla\phi_j \cdot \nabla\phi_i \, d\Omega + \int_\Omega V\phi_j \phi_i \, d\Omega$
- $\mathbf{M}$ is the mass matrix with elements $M_{ij} = \int_\Omega \phi_j \phi_i \, d\Omega$
- $\boldsymbol{\psi}$ is the vector of coefficients $\psi_j$

### Basis Functions

QDSim uses linear (P1) or quadratic (P2) Lagrange basis functions. For linear elements, the basis functions are:

\begin{equation}
\phi_i(\mathbf{r}) = 
\begin{cases}
1 & \text{at node } i \\
0 & \text{at all other nodes}
\end{cases}
\end{equation}

with linear interpolation between nodes.

For quadratic elements, additional nodes are placed at the midpoints of the edges, allowing for quadratic interpolation.

### Element Matrices

The integrals in the Hamiltonian and mass matrices are computed element by element. For each triangular element $e$, we compute:

\begin{equation}
H^e_{ij} = \int_{\Omega_e} \frac{\hbar^2}{2m^*} \nabla\phi_j \cdot \nabla\phi_i \, d\Omega + \int_{\Omega_e} V\phi_j \phi_i \, d\Omega
\end{equation}

\begin{equation}
M^e_{ij} = \int_{\Omega_e} \phi_j \phi_i \, d\Omega
\end{equation}

These element matrices are then assembled into the global matrices $\mathbf{H}$ and $\mathbf{M}$.

### Numerical Integration

The integrals are computed using numerical quadrature. For linear elements, a one-point quadrature rule is sufficient:

\begin{equation}
\int_{\Omega_e} f(\mathbf{r}) \, d\Omega \approx A_e f(\mathbf{r}_c)
\end{equation}

where $A_e$ is the area of the element and $\mathbf{r}_c$ is the centroid of the element.

For quadratic elements, a three-point quadrature rule is used:

\begin{equation}
\int_{\Omega_e} f(\mathbf{r}) \, d\Omega \approx \frac{A_e}{3} \sum_{i=1}^{3} f(\mathbf{r}_i)
\end{equation}

where $\mathbf{r}_i$ are the midpoints of the edges of the element.

## Eigenvalue Problem

### Generalized Eigenvalue Problem

The discretized Schrödinger equation leads to a generalized eigenvalue problem:

\begin{equation}
\mathbf{H} \boldsymbol{\psi} = E \mathbf{M} \boldsymbol{\psi}
\end{equation}

where $\mathbf{H}$ is the Hamiltonian matrix, $\mathbf{M}$ is the mass matrix, $\boldsymbol{\psi}$ is the eigenvector, and $E$ is the eigenvalue.

### Solution Methods

QDSim uses several methods to solve the generalized eigenvalue problem:

1. **Direct methods**: For small systems, QDSim uses direct methods such as the Cholesky decomposition:
   - Compute the Cholesky decomposition of $\mathbf{M} = \mathbf{L}\mathbf{L}^T$
   - Transform the problem to a standard eigenvalue problem: $\mathbf{L}^{-1}\mathbf{H}\mathbf{L}^{-T}\mathbf{y} = E\mathbf{y}$, where $\mathbf{y} = \mathbf{L}^T\boldsymbol{\psi}$
   - Solve the standard eigenvalue problem using Eigen's SelfAdjointEigenSolver
   - Transform the eigenvectors back: $\boldsymbol{\psi} = \mathbf{L}^{-T}\mathbf{y}$

2. **Iterative methods**: For large systems, QDSim uses iterative methods such as the Arnoldi method (ARPACK):
   - Use the shift-and-invert mode to compute the lowest eigenvalues
   - Iteratively build a Krylov subspace and compute the eigenvalues and eigenvectors of the projected problem
   - Transform the eigenvectors back to the original problem

3. **Parallel methods**: For very large systems, QDSim uses parallel methods such as SLEPc:
   - Distribute the matrices across multiple processors
   - Use parallel algorithms for matrix-vector multiplication and orthogonalization
   - Compute the eigenvalues and eigenvectors in parallel

### Normalization

The eigenvectors are normalized with respect to the mass matrix:

\begin{equation}
\boldsymbol{\psi}^T \mathbf{M} \boldsymbol{\psi} = 1
\end{equation}

This ensures that the wavefunctions are properly normalized:

\begin{equation}
\int_\Omega |\psi(\mathbf{r})|^2 \, d\Omega = 1
\end{equation}

## Material Properties

### Effective Mass

The effective mass of electrons and holes in semiconductors depends on the material and can be anisotropic. QDSim uses the following effective masses:

- Electron effective mass ($m_e^*$)
- Light hole effective mass ($m_{lh}^*$)
- Heavy hole effective mass ($m_{hh}^*$)
- Split-off hole effective mass ($m_{so}^*$)

For isotropic materials, the effective mass is a scalar. For anisotropic materials, the effective mass is a tensor.

### Bandgap

The bandgap of a semiconductor is the energy difference between the conduction band minimum and the valence band maximum. It depends on the material and temperature.

The temperature dependence of the bandgap is described by the Varshni equation:

\begin{equation}
E_g(T) = E_g(0) - \frac{\alpha T^2}{T + \beta}
\end{equation}

where $E_g(0)$ is the bandgap at 0 K, $\alpha$ and $\beta$ are material-dependent parameters, and $T$ is the temperature in Kelvin.

### Band Offsets

The band offsets between different semiconductor materials determine the confinement potential for electrons and holes. The conduction band offset ($\Delta E_c$) and valence band offset ($\Delta E_v$) are related to the bandgap difference:

\begin{equation}
\Delta E_g = \Delta E_c + \Delta E_v
\end{equation}

The band alignment can be type I (straddling gap), type II (staggered gap), or type III (broken gap).

### Alloys

Semiconductor alloys such as Al$_x$Ga$_{1-x}$As have properties that depend on the composition parameter $x$. The dependence is often non-linear and is described by bowing parameters:

\begin{equation}
P(A_x B_{1-x}) = x P(A) + (1-x) P(B) - x(1-x) C_P
\end{equation}

where $P$ is a property such as the bandgap, $P(A)$ and $P(B)$ are the properties of the constituent materials, and $C_P$ is the bowing parameter.

## Spin-Orbit Coupling

### Rashba Spin-Orbit Coupling

Rashba spin-orbit coupling arises from structural inversion asymmetry, such as in asymmetric quantum wells or at interfaces. The Rashba Hamiltonian is:

\begin{equation}
H_R = \alpha (\sigma_x k_y - \sigma_y k_x)
\end{equation}

where $\alpha$ is the Rashba parameter, $\sigma_x$ and $\sigma_y$ are the Pauli matrices, and $k_x$ and $k_y$ are the wave vectors.

The Rashba parameter depends on the electric field perpendicular to the 2D plane:

\begin{equation}
\alpha = \alpha_0 E_z
\end{equation}

where $\alpha_0$ is a material-dependent parameter and $E_z$ is the electric field.

### Dresselhaus Spin-Orbit Coupling

Dresselhaus spin-orbit coupling arises from bulk inversion asymmetry in zinc-blende semiconductors. The Dresselhaus Hamiltonian for a 2D system is:

\begin{equation}
H_D = \beta (\sigma_x k_x - \sigma_y k_y)
\end{equation}

where $\beta$ is the Dresselhaus parameter.

The Dresselhaus parameter depends on the confinement:

\begin{equation}
\beta = \gamma \langle k_z^2 \rangle
\end{equation}

where $\gamma$ is a material-dependent parameter and $\langle k_z^2 \rangle$ is the expectation value of $k_z^2$, which depends on the confinement in the z-direction.

### Combined Spin-Orbit Coupling

When both Rashba and Dresselhaus spin-orbit coupling are present, the total spin-orbit Hamiltonian is:

\begin{equation}
H_{SO} = H_R + H_D = \alpha (\sigma_x k_y - \sigma_y k_x) + \beta (\sigma_x k_x - \sigma_y k_y)
\end{equation}

This leads to anisotropic spin splitting and spin textures.

## Self-Consistent Calculations

### Poisson Equation

In self-consistent calculations, the potential $V(\mathbf{r})$ depends on the charge density, which in turn depends on the wavefunctions. The potential is computed by solving the Poisson equation:

\begin{equation}
\nabla \cdot (\epsilon(\mathbf{r}) \nabla \phi(\mathbf{r})) = -\rho(\mathbf{r})
\end{equation}

where $\epsilon(\mathbf{r})$ is the dielectric permittivity, $\phi(\mathbf{r})$ is the electrostatic potential, and $\rho(\mathbf{r})$ is the charge density.

### Charge Density

The charge density is computed from the wavefunctions:

\begin{equation}
\rho(\mathbf{r}) = -e \sum_i n_i |\psi_i(\mathbf{r})|^2
\end{equation}

where $e$ is the elementary charge, $n_i$ is the occupation number of state $i$, and $\psi_i(\mathbf{r})$ is the wavefunction of state $i$.

### Self-Consistent Loop

The self-consistent calculation proceeds as follows:

1. Start with an initial guess for the potential $V(\mathbf{r})$
2. Solve the Schrödinger equation to find the wavefunctions $\psi_i(\mathbf{r})$
3. Compute the charge density $\rho(\mathbf{r})$
4. Solve the Poisson equation to find the new potential $V_{new}(\mathbf{r})$
5. Update the potential: $V(\mathbf{r}) = (1-\alpha) V(\mathbf{r}) + \alpha V_{new}(\mathbf{r})$, where $\alpha$ is a damping factor
6. Repeat steps 2-5 until convergence

Convergence is achieved when the change in the potential or the energy eigenvalues is below a specified tolerance.

## Numerical Methods

### Mesh Generation

QDSim uses a structured or unstructured triangular mesh to discretize the simulation domain. The mesh is generated using the following steps:

1. Create a rectangular grid of points
2. Triangulate the grid using Delaunay triangulation
3. Refine the mesh in regions of interest

### Adaptive Mesh Refinement

Adaptive mesh refinement improves the accuracy of the simulation by refining the mesh in regions where the solution varies rapidly. The refinement process involves:

1. Compute an error estimator for each element
2. Mark elements for refinement or coarsening based on the error
3. Refine or coarsen the marked elements
4. Update the mesh data structures

### Error Estimation

QDSim uses several error estimators, including:

1. **Residual-based estimators**: Compute the residual of the Schrödinger equation in each element
2. **Recovery-based estimators**: Compare the computed solution with a recovered solution
3. **Hierarchical estimators**: Compare the solution with a higher-order approximation

### Convergence Acceleration

For self-consistent calculations, QDSim uses several techniques to accelerate convergence:

1. **Damping**: Use a damping factor to update the potential
2. **Anderson acceleration**: Use a linear combination of previous iterations
3. **Broyden's method**: Use a quasi-Newton method to update the potential

### Parallel Computing

QDSim supports parallel computing using:

1. **OpenMP**: For shared-memory parallelism
2. **MPI**: For distributed-memory parallelism
3. **CUDA**: For GPU acceleration

## References

1. S. Datta, "Quantum Transport: Atom to Transistor," Cambridge University Press, 2005.
2. D. Vasileska, S. M. Goodnick, and G. Klimeck, "Computational Electronics: Semiclassical and Quantum Device Modeling and Simulation," CRC Press, 2010.
3. J. Davies, "The Physics of Low-Dimensional Semiconductors: An Introduction," Cambridge University Press, 1998.
4. P. Harrison, "Quantum Wells, Wires and Dots: Theoretical and Computational Physics of Semiconductor Nanostructures," Wiley, 2005.
5. T. Chakraborty, "Quantum Dots: A Survey of the Properties of Artificial Atoms," Elsevier, 1999.
6. R. Winkler, "Spin-Orbit Coupling Effects in Two-Dimensional Electron and Hole Systems," Springer, 2003.
7. M. Ainsworth and J. T. Oden, "A Posteriori Error Estimation in Finite Element Analysis," Wiley, 2000.
8. S. C. Brenner and L. R. Scott, "The Mathematical Theory of Finite Element Methods," Springer, 2008.
