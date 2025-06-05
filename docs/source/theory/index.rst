Theory and Mathematical Formulation
====================================

This section provides comprehensive theoretical background for QDSim's quantum mechanical simulations, numerical methods, and implementation details.

.. toctree::
   :maxdepth: 2

   quantum_mechanics
   numerical_methods
   open_systems
   finite_elements
   semiconductor_physics
   performance_theory

Overview
--------

QDSim implements state-of-the-art quantum mechanical simulations based on rigorous theoretical foundations:

**Quantum Mechanics**
    - Time-independent Schrödinger equation with complex boundary conditions
    - Open quantum system theory with finite lifetimes
    - Self-consistent field theory for coupled Poisson-Schrödinger equations
    - Dirac delta normalization for scattering states

**Numerical Methods**
    - Finite Element Method (FEM) with weak formulation
    - Advanced eigenvalue algorithms for large sparse matrices
    - Adaptive mesh refinement and error estimation
    - Complex absorbing potentials for open boundaries

**Semiconductor Physics**
    - Effective mass approximation in heterostructures
    - Band structure and material properties
    - Interface physics and boundary conditions
    - Device-specific modeling approaches

Mathematical Notation
--------------------

Throughout this documentation, we use the following mathematical conventions:

**Quantum Mechanical Quantities**
    - :math:`\psi(\mathbf{r})`: Wavefunction
    - :math:`\hat{H}`: Hamiltonian operator
    - :math:`E`: Energy eigenvalue (complex for open systems)
    - :math:`\hbar`: Reduced Planck constant
    - :math:`m^*`: Effective mass

**Spatial Coordinates**
    - :math:`\mathbf{r} = (x, y, z)`: Position vector
    - :math:`\Omega`: Computational domain
    - :math:`\partial\Omega`: Domain boundary
    - :math:`\mathbf{n}`: Outward normal vector

**Finite Element Notation**
    - :math:`\phi_i(\mathbf{r})`: Basis functions
    - :math:`\mathbf{K}`: Stiffness matrix
    - :math:`\mathbf{M}`: Mass matrix
    - :math:`\mathbf{u}_h`: Finite element solution

**Complex Analysis**
    - :math:`\text{Re}(z)`: Real part of complex number
    - :math:`\text{Im}(z)`: Imaginary part of complex number
    - :math:`|z|`: Modulus of complex number
    - :math:`z^*`: Complex conjugate

Physical Constants
-----------------

QDSim uses the following physical constants:

.. list-table:: Physical Constants
   :widths: 30 20 30 20
   :header-rows: 1

   * - Quantity
     - Symbol
     - Value
     - Units
   * - Reduced Planck constant
     - :math:`\hbar`
     - 1.054571817×10⁻³⁴
     - J⋅s
   * - Electron mass
     - :math:`m_e`
     - 9.1093837015×10⁻³¹
     - kg
   * - Elementary charge
     - :math:`e`
     - 1.602176634×10⁻¹⁹
     - C
   * - Boltzmann constant
     - :math:`k_B`
     - 1.380649×10⁻²³
     - J/K
   * - Vacuum permittivity
     - :math:`\epsilon_0`
     - 8.8541878128×10⁻¹²
     - F/m

Theoretical Framework
--------------------

The theoretical foundation of QDSim rests on several key principles:

1. **Quantum Mechanical Formalism**
   
   The time-independent Schrödinger equation forms the core of our simulations:
   
   .. math::
      \hat{H}\psi(\mathbf{r}) = E\psi(\mathbf{r})
   
   where the Hamiltonian includes kinetic and potential energy terms:
   
   .. math::
      \hat{H} = -\frac{\hbar^2}{2m^*(\mathbf{r})}\nabla^2 + V(\mathbf{r})

2. **Open System Theory**
   
   For open quantum systems, we extend to complex eigenvalue problems:
   
   .. math::
      \hat{H}_{\text{eff}}\psi(\mathbf{r}) = E_{\text{complex}}\psi(\mathbf{r})
   
   where :math:`E_{\text{complex}} = E_{\text{real}} + i\Gamma/2` includes finite lifetimes.

3. **Finite Element Discretization**
   
   The weak formulation leads to the generalized eigenvalue problem:
   
   .. math::
      \mathbf{K}\mathbf{u} = \lambda\mathbf{M}\mathbf{u}
   
   where :math:`\mathbf{K}` and :math:`\mathbf{M}` are the stiffness and mass matrices.

4. **Self-Consistent Field Theory**
   
   For coupled systems, we solve iteratively:
   
   .. math::
      \begin{align}
      \nabla^2 V(\mathbf{r}) &= -\frac{\rho(\mathbf{r})}{\epsilon(\mathbf{r})} \\
      \hat{H}[V]\psi(\mathbf{r}) &= E\psi(\mathbf{r})
      \end{align}

Implementation Philosophy
------------------------

QDSim's implementation follows these guiding principles:

**Physical Accuracy**
    All implementations are validated against analytical solutions and experimental data where available.

**Numerical Stability**
    Robust algorithms with error estimation and adaptive refinement ensure reliable results.

**Computational Efficiency**
    High-performance computing techniques enable large-scale simulations.

**Extensibility**
    Modular design allows easy addition of new physics and numerical methods.

**Reproducibility**
    Comprehensive documentation and testing ensure reproducible scientific results.

References
----------

The theoretical foundations implemented in QDSim are based on:

1. Griffiths, D. J. (2018). *Introduction to Quantum Mechanics*. Cambridge University Press.
2. Sakurai, J. J., & Napolitano, J. (2017). *Modern Quantum Mechanics*. Cambridge University Press.
3. Tannor, D. J. (2007). *Introduction to Quantum Mechanics: A Time-Dependent Perspective*. University Science Books.
4. Brenner, S., & Scott, R. (2007). *The Mathematical Theory of Finite Element Methods*. Springer.
5. Moiseyev, N. (2011). *Non-Hermitian Quantum Mechanics*. Cambridge University Press.
