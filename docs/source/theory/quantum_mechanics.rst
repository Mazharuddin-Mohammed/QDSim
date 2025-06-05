Quantum Mechanics Formulation
=============================

This section details the quantum mechanical foundations underlying QDSim's simulations, including the time-independent Schrödinger equation, open system theory, and semiconductor-specific considerations.

Time-Independent Schrödinger Equation
-------------------------------------

The fundamental equation governing quantum mechanical systems in QDSim is the time-independent Schrödinger equation:

.. math::
   \hat{H}\psi(\mathbf{r}) = E\psi(\mathbf{r})

where:
- :math:`\hat{H}` is the Hamiltonian operator
- :math:`\psi(\mathbf{r})` is the wavefunction
- :math:`E` is the energy eigenvalue
- :math:`\mathbf{r} = (x, y, z)` is the position vector

Hamiltonian Operator
~~~~~~~~~~~~~~~~~~~

For semiconductor nanostructures, the Hamiltonian consists of kinetic and potential energy terms:

.. math::
   \hat{H} = \hat{T} + \hat{V} = -\frac{\hbar^2}{2m^*(\mathbf{r})}\nabla^2 + V(\mathbf{r})

**Kinetic Energy Operator**

The kinetic energy operator accounts for spatially varying effective mass:

.. math::
   \hat{T} = -\frac{\hbar^2}{2}\nabla \cdot \left[\frac{1}{m^*(\mathbf{r})}\nabla\right]

For constant effective mass regions, this simplifies to:

.. math::
   \hat{T} = -\frac{\hbar^2}{2m^*}\nabla^2

**Potential Energy**

The potential energy includes multiple contributions:

.. math::
   V(\mathbf{r}) = V_{\text{conf}}(\mathbf{r}) + V_{\text{ext}}(\mathbf{r}) + V_{\text{H}}(\mathbf{r}) + V_{\text{xc}}(\mathbf{r})

where:
- :math:`V_{\text{conf}}(\mathbf{r})`: Confinement potential from band offsets
- :math:`V_{\text{ext}}(\mathbf{r})`: External applied potentials
- :math:`V_{\text{H}}(\mathbf{r})`: Hartree potential from charge distribution
- :math:`V_{\text{xc}}(\mathbf{r})`: Exchange-correlation potential

Effective Mass Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In semiconductor heterostructures, we use the effective mass approximation:

.. math::
   m^*(\mathbf{r}) = \begin{cases}
   m^*_1 & \text{in material 1} \\
   m^*_2 & \text{in material 2} \\
   \vdots & \vdots
   \end{cases}

The effective mass tensor for anisotropic materials is:

.. math::
   \mathbf{m}^* = \begin{pmatrix}
   m^*_{xx} & m^*_{xy} & m^*_{xz} \\
   m^*_{yx} & m^*_{yy} & m^*_{yz} \\
   m^*_{zx} & m^*_{zy} & m^*_{zz}
   \end{pmatrix}

Boundary Conditions
------------------

QDSim supports multiple types of boundary conditions for different physical scenarios.

Closed System Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~

**Dirichlet Boundary Conditions (Hard Walls)**

For infinite potential barriers:

.. math::
   \psi(\mathbf{r}) = 0 \quad \text{on } \partial\Omega

**Neumann Boundary Conditions (Soft Walls)**

For finite potential barriers:

.. math::
   \frac{\partial\psi}{\partial n} = 0 \quad \text{on } \partial\Omega

Open System Boundaries
~~~~~~~~~~~~~~~~~~~~~~

For open quantum systems, we implement Complex Absorbing Potentials (CAP):

.. math::
   V_{\text{CAP}}(\mathbf{r}) = -i\eta(\mathbf{r})W(\mathbf{r})

where :math:`\eta(\mathbf{r})` is the absorption strength and :math:`W(\mathbf{r})` is the absorbing function.

**Polynomial CAP**

.. math::
   W(\mathbf{r}) = \begin{cases}
   \left(\frac{d(\mathbf{r})}{d_0}\right)^n & \text{if } d(\mathbf{r}) < d_0 \\
   0 & \text{otherwise}
   \end{cases}

where :math:`d(\mathbf{r})` is the distance from the boundary and :math:`n` is typically 2 or 3.

Interface Conditions
~~~~~~~~~~~~~~~~~~~

At material interfaces, we enforce continuity conditions:

**Wavefunction Continuity**

.. math::
   \psi_1(\mathbf{r}_{\text{interface}}) = \psi_2(\mathbf{r}_{\text{interface}})

**Current Continuity**

.. math::
   \frac{1}{m^*_1}\frac{\partial\psi_1}{\partial n} = \frac{1}{m^*_2}\frac{\partial\psi_2}{\partial n}

Open Quantum Systems
--------------------

For open systems with finite lifetimes, eigenvalues become complex:

.. math::
   E = E_{\text{real}} + i\Gamma/2

where :math:`\Gamma` is the decay width related to the lifetime by:

.. math::
   \tau = \frac{\hbar}{\Gamma}

Complex Eigenvalue Problem
~~~~~~~~~~~~~~~~~~~~~~~~~

The generalized eigenvalue problem becomes:

.. math::
   (\mathbf{K} + i\mathbf{K}_{\text{CAP}})\mathbf{u} = \lambda\mathbf{M}\mathbf{u}

where :math:`\mathbf{K}_{\text{CAP}}` represents the CAP contribution.

Normalization for Open Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For scattering states, we use Dirac delta normalization:

.. math::
   \langle\psi_{E'}|\psi_E\rangle = \delta(E - E')

rather than the standard :math:`L^2` normalization.

Self-Consistent Field Theory
----------------------------

For systems with significant charge redistribution, we solve the coupled Poisson-Schrödinger equations self-consistently.

Poisson Equation
~~~~~~~~~~~~~~~~

The electrostatic potential satisfies:

.. math::
   \nabla \cdot [\epsilon(\mathbf{r})\nabla V(\mathbf{r})] = -\rho(\mathbf{r})

where the charge density includes:

.. math::
   \rho(\mathbf{r}) = -e\sum_i f_i|\psi_i(\mathbf{r})|^2 + \rho_{\text{ion}}(\mathbf{r})

with :math:`f_i` being the occupation factors and :math:`\rho_{\text{ion}}` the ionized dopant density.

Self-Consistent Iteration
~~~~~~~~~~~~~~~~~~~~~~~~~

The self-consistent loop proceeds as:

1. **Initial Guess**: Start with an initial potential :math:`V^{(0)}(\mathbf{r})`

2. **Solve Schrödinger**: 
   .. math::
      \hat{H}[V^{(n)}]\psi_i^{(n)} = E_i^{(n)}\psi_i^{(n)}

3. **Update Charge Density**:
   .. math::
      \rho^{(n+1)}(\mathbf{r}) = -e\sum_i f_i|\psi_i^{(n)}(\mathbf{r})|^2 + \rho_{\text{ion}}(\mathbf{r})

4. **Solve Poisson**:
   .. math::
      \nabla \cdot [\epsilon(\mathbf{r})\nabla V^{(n+1)}(\mathbf{r})] = -\rho^{(n+1)}(\mathbf{r})

5. **Check Convergence**: Repeat until :math:`|V^{(n+1)} - V^{(n)}| < \text{tolerance}`

Mixing Schemes
~~~~~~~~~~~~~~

To improve convergence, we use mixing schemes:

**Linear Mixing**

.. math::
   V^{(n+1)}_{\text{mixed}} = \alpha V^{(n+1)} + (1-\alpha)V^{(n)}

**Anderson Mixing**

More sophisticated mixing using history of previous iterations to accelerate convergence.

Many-Body Effects
----------------

For systems where electron-electron interactions are important, we include exchange-correlation effects.

Local Density Approximation (LDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   V_{\text{xc}}(\mathbf{r}) = \frac{\delta E_{\text{xc}}[\rho]}{\delta \rho(\mathbf{r})}

where :math:`E_{\text{xc}}[\rho]` is the exchange-correlation energy functional.

Quantum Confinement Effects
---------------------------

In quantum dots and wells, confinement leads to discrete energy levels and modified density of states.

Size Quantization
~~~~~~~~~~~~~~~~~

For a particle in a box with dimensions :math:`L_x \times L_y \times L_z`:

.. math::
   E_{n_x,n_y,n_z} = \frac{\hbar^2\pi^2}{2m^*}\left(\frac{n_x^2}{L_x^2} + \frac{n_y^2}{L_y^2} + \frac{n_z^2}{L_z^2}\right)

Coulomb Blockade
~~~~~~~~~~~~~~~

In small quantum dots, charging energy becomes important:

.. math::
   E_{\text{charging}} = \frac{e^2}{2C}

where :math:`C` is the capacitance of the quantum dot.

Spin-Orbit Coupling
------------------

For materials with significant spin-orbit coupling, the Hamiltonian includes additional terms:

.. math::
   \hat{H}_{\text{SO}} = \frac{\hbar}{4m_0^2c^2}\boldsymbol{\sigma} \cdot (\nabla V \times \mathbf{p})

where :math:`\boldsymbol{\sigma}` are the Pauli matrices and :math:`\mathbf{p}` is the momentum operator.

Validation and Benchmarks
-------------------------

QDSim's quantum mechanical implementations are validated against:

1. **Analytical Solutions**: Particle in a box, harmonic oscillator, hydrogen atom
2. **Experimental Data**: Quantum dot spectroscopy, transport measurements
3. **Other Simulation Codes**: Comparison with established quantum simulation packages
4. **Literature Results**: Reproduction of published theoretical calculations

The validation ensures that QDSim produces physically accurate results for a wide range of quantum mechanical systems.

Implementation Examples
----------------------

**Basic Quantum Dot Simulation**

.. code-block:: python

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

    # Create solver
    solver = qdsim.FixedOpenSystemSolver(
        nx=8, ny=6, Lx=25e-9, Ly=20e-9,
        m_star_func=m_star_func,
        potential_func=potential_func,
        use_open_boundaries=True
    )

    # Solve quantum system
    eigenvals, eigenvecs = solver.solve(num_states=5)

**Complex Eigenvalue Analysis**

.. code-block:: python

    # Analyze complex eigenvalues for open systems
    for i, E in enumerate(eigenvals):
        E_real_eV = np.real(E) / 1.602176634e-19
        E_imag_eV = np.imag(E) / 1.602176634e-19

        if abs(np.imag(E)) > 1e-25:
            # Calculate lifetime from imaginary part
            lifetime_fs = 1.054571817e-34 / (2 * abs(np.imag(E))) * 1e15
            print(f"E_{i+1}: {E_real_eV:.6f} + {E_imag_eV:.6f}j eV")
            print(f"      Lifetime: {lifetime_fs:.1f} fs")
        else:
            print(f"E_{i+1}: {E_real_eV:.6f} eV (bound state)")

This comprehensive quantum mechanical framework enables QDSim to accurately simulate a wide range of semiconductor nanostructures and quantum devices.
