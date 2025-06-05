QDSim Documentation
===================

.. image:: https://readthedocs.org/projects/qdsim/badge/?version=latest
    :target: https://qdsim.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
    :target: https://www.python.org/downloads/
    :alt: Python 3.8+

Welcome to QDSim, a state-of-the-art quantum dot simulator for semiconductor nanostructures featuring advanced finite element methods, open quantum systems, GPU acceleration, and comprehensive visualization capabilities.

🚀 Quick Start
--------------

.. code-block:: python

    import qdsim
    import numpy as np

    # Define material properties
    def m_star_func(x, y):
        return 0.067 * 9.1093837015e-31  # InGaAs effective mass

    def potential_func(x, y):
        if 5e-9 < x < 15e-9:
            return -0.06 * 1.602176634e-19  # -60 meV well
        return 0.0

    # Create open system solver
    solver = qdsim.FixedOpenSystemSolver(
        nx=8, ny=6, Lx=25e-9, Ly=20e-9,
        m_star_func=m_star_func,
        potential_func=potential_func,
        use_open_boundaries=True
    )

    # Solve quantum system
    eigenvals, eigenvecs = solver.solve(num_states=5)

    # Visualize results
    from qdsim.visualization import WavefunctionPlotter
    plotter = WavefunctionPlotter()
    plotter.plot_energy_levels(eigenvals, "Energy Levels")

🎯 Key Features
---------------

**Quantum Physics Engine**
    - Open quantum systems with complex eigenvalues and finite lifetimes
    - Self-consistent Poisson-Schrödinger solvers
    - Advanced eigenvalue algorithms (ARPACK, FEAST, Jacobi-Davidson)
    - Dirac delta normalization for scattering states

**High-Performance Computing**
    - Cython backend for C-level performance
    - GPU acceleration with CUDA support
    - Advanced memory management and optimization
    - Hybrid MPI+OpenMP+CUDA parallelization

**Advanced Numerical Methods**
    - Finite Element Method (FEM) with adaptive refinement
    - Complex boundary conditions for open systems
    - Material interface handling
    - Robust numerical stability

**Comprehensive Visualization**
    - Interactive 3D plotting capabilities
    - Energy level diagrams with lifetime analysis
    - Device structure visualization
    - Publication-quality figures

📚 Documentation Sections
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Theory and Methods

   theory/index
   theory/quantum_mechanics
   theory/numerical_methods
   theory/open_systems
   theory/finite_elements

.. toctree::
   :maxdepth: 2
   :caption: Enhancements (Chronological Order)

   enhancements/index
   enhancements/cython_migration
   enhancements/memory_management
   enhancements/gpu_acceleration
   enhancements/open_systems

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/basic_usage
   user_guide/advanced_features
   user_guide/visualization
   user_guide/performance

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/solvers
   api/visualization
   api/materials
   api/utilities

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/basic_quantum_dot
   examples/open_systems
   examples/complex_devices
   examples/performance_optimization

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/index
   developer/contributing
   developer/architecture
   developer/testing
   developer/documentation
   developer/extensions
   developer/future_development

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   bibliography
   glossary
   faq

🔬 Scientific Applications
--------------------------

QDSim is designed for cutting-edge quantum device simulation:

- **Quantum Dots in p-n Junctions**: Chromium QDs in InGaAs diodes under bias
- **Open Quantum Systems**: Electron injection and extraction in semiconductor devices  
- **Resonant Tunneling**: Complex eigenvalue analysis of tunneling structures
- **Heterojunctions**: Band alignment and carrier confinement studies
- **Device Optimization**: Parameter sweeps and performance analysis

🤝 Community and Support
------------------------

- **GitHub Repository**: `QDSim on GitHub <https://github.com/your-username/QDSim>`_
- **Issue Tracker**: `Report bugs and request features <https://github.com/your-username/QDSim/issues>`_
- **Discussions**: `Community discussions <https://github.com/your-username/QDSim/discussions>`_
- **Contributing**: See our :doc:`developer/contributing` guide

📄 License and Citation
-----------------------

QDSim is released under the MIT License. If you use QDSim in your research, please cite:

.. code-block:: bibtex

    @software{qdsim2024,
      title={QDSim: Advanced Quantum Dot Simulator},
      author={Dr. Mazharuddin Mohammed},
      year={2024},
      url={https://github.com/your-username/QDSim},
      version={2.0.0}
    }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
