Quick Start Guide
=================

This guide will get you up and running with QDSim in just a few minutes.

Installation
-----------

First, install QDSim and its dependencies:

.. code-block:: bash

    # Install system dependencies
    sudo apt-get install python3-dev python3-matplotlib python3-numpy python3-scipy
    
    # Clone and install QDSim
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    pip install -e .

Your First Quantum Simulation
-----------------------------

Let's create a simple quantum dot simulation:

.. code-block:: python

    import qdsim
    import numpy as np

    # Define material properties
    def m_star_func(x, y):
        """Effective mass function for InGaAs"""
        return 0.067 * 9.1093837015e-31

    def potential_func(x, y):
        """Quantum well potential"""
        if 5e-9 < x < 15e-9:
            return -0.06 * 1.602176634e-19  # -60 meV well
        return 0.0

    # Create quantum solver
    solver = qdsim.FixedOpenSystemSolver(
        nx=8, ny=6,                    # Grid points
        Lx=25e-9, Ly=20e-9,           # Domain size
        m_star_func=m_star_func,
        potential_func=potential_func,
        use_open_boundaries=True       # Enable open system
    )

    # Apply open system physics
    solver.apply_open_system_boundary_conditions()
    solver.apply_dirac_delta_normalization()
    solver.configure_device_specific_solver('quantum_well')

    # Solve the quantum system
    eigenvals, eigenvecs = solver.solve(num_states=5)

    print(f"Found {len(eigenvals)} quantum states")

Analyzing Results
----------------

Examine the complex eigenvalues to understand the physics:

.. code-block:: python

    # Analyze eigenvalues
    for i, E in enumerate(eigenvals):
        E_real_eV = np.real(E) / 1.602176634e-19
        E_imag_eV = np.imag(E) / 1.602176634e-19
        
        if abs(np.imag(E)) > 1e-25:
            # Complex eigenvalue - finite lifetime
            lifetime_fs = 1.054571817e-34 / (2 * abs(np.imag(E))) * 1e15
            print(f"State {i+1}: {E_real_eV:.6f} + {E_imag_eV:.6f}j eV")
            print(f"          Lifetime: {lifetime_fs:.1f} fs")
        else:
            # Real eigenvalue - bound state
            print(f"State {i+1}: {E_real_eV:.6f} eV (bound)")

Visualization
------------

Create publication-quality plots:

.. code-block:: python

    from qdsim.visualization import WavefunctionPlotter

    # Create plotter
    plotter = WavefunctionPlotter()

    # Plot energy levels
    plotter.plot_energy_levels(eigenvals, "Quantum Energy Levels")

    # Plot ground state wavefunction
    plotter.plot_wavefunction_2d(
        solver.nodes_x, solver.nodes_y, 
        eigenvecs[0], "Ground State Wavefunction"
    )

    # Comprehensive analysis plot
    plotter.plot_comprehensive_analysis(
        solver.nodes_x, solver.nodes_y,
        eigenvals, eigenvecs,
        potential_func, m_star_func,
        "Complete Quantum Analysis"
    )

What's Next?
-----------

Now that you have QDSim working, explore these resources:

- **Tutorials**: :doc:`tutorials/index` - Step-by-step guides
- **Examples**: :doc:`examples/index` - Real-world applications  
- **Theory**: :doc:`theory/index` - Mathematical foundations
- **API Reference**: :doc:`api/index` - Complete documentation

Common Use Cases
---------------

**Quantum Dots in p-n Junctions**
    Simulate chromium quantum dots in InGaAs diodes under bias conditions.

**Open Quantum Systems**
    Study electron injection and extraction in semiconductor devices.

**Resonant Tunneling**
    Analyze complex eigenvalues in tunneling structures.

**Device Optimization**
    Perform parameter sweeps to optimize quantum device performance.

Getting Help
-----------

If you need assistance:

- **Documentation**: This comprehensive guide
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Email**: qdsim-support@example.com

Congratulations! You've successfully run your first quantum simulation with QDSim.
