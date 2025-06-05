Examples and Use Cases
======================

This section provides comprehensive examples demonstrating how to use QDSim for various quantum simulation scenarios, from basic quantum dots to advanced open systems.

.. toctree::
   :maxdepth: 2

   basic_quantum_dot
   open_systems
   complex_devices
   performance_optimization
   realistic_devices
   advanced_physics

Quick Example Gallery
--------------------

**Basic Quantum Dot Simulation**
    Simple quantum dot in a 2D potential well with bound states.

**Open Quantum System**
    Quantum dot with finite lifetimes using Complex Absorbing Potentials.

**Chromium QD in InGaAs**
    Realistic device simulation with material properties and bias conditions.

**GPU-Accelerated Simulation**
    Large-scale quantum system using GPU acceleration.

**Multi-Physics Coupling**
    Self-consistent Poisson-Schrödinger simulation with charge redistribution.

**Parameter Sweeps**
    Systematic studies of device performance vs. design parameters.

Example Categories
-----------------

**Beginner Examples**
    - Basic quantum mechanics concepts
    - Simple potential wells and barriers
    - Energy level calculations
    - Wavefunction visualization

**Intermediate Examples**
    - Quantum dots and wells
    - Material interfaces
    - Applied electric fields
    - Self-consistent calculations

**Advanced Examples**
    - Open quantum systems
    - Complex eigenvalue analysis
    - Finite lifetime calculations
    - Device-specific optimizations

**Research Examples**
    - Realistic device geometries
    - Experimental parameter matching
    - Transport calculations
    - Many-body effects

Running the Examples
-------------------

All examples can be run directly:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-username/QDSim.git
    cd QDSim
    
    # Install QDSim
    pip install -e .
    
    # Run basic example
    python examples/basic_quantum_dot.py
    
    # Run with GPU acceleration
    python examples/gpu_accelerated_simulation.py
    
    # Run realistic device example
    python examples/chromium_qd_ingaas.py

Example Data and Validation
---------------------------

Each example includes:

- **Complete source code** with detailed comments
- **Expected output** and result interpretation
- **Validation data** comparing with analytical or experimental results
- **Performance benchmarks** showing execution times
- **Visualization scripts** for publication-quality plots

Interactive Notebooks
---------------------

Jupyter notebooks are available for interactive exploration:

.. code-block:: bash

    # Install notebook dependencies
    pip install jupyter matplotlib
    
    # Start Jupyter
    jupyter notebook examples/notebooks/
    
    # Open desired notebook
    # - basic_quantum_mechanics.ipynb
    # - open_systems_tutorial.ipynb
    # - device_optimization.ipynb

Example Validation
------------------

All examples are validated against:

- **Analytical solutions** where available
- **Experimental data** from literature
- **Other simulation codes** for cross-validation
- **Physical constraints** and conservation laws

The validation ensures that examples demonstrate correct physics and provide reliable starting points for research applications.

Contributing Examples
---------------------

We welcome community contributions of examples:

1. **Fork the repository** on GitHub
2. **Create example** following the template structure
3. **Add validation** and documentation
4. **Submit pull request** with example description

See :doc:`../developer/contributing` for detailed guidelines on contributing examples.

Getting Help
-----------

If you need help with examples:

- **Documentation**: Each example has detailed inline documentation
- **GitHub Issues**: Report problems or request new examples
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: Contact qdsim-support@example.com

The examples provide practical demonstrations of QDSim's capabilities and serve as starting points for your own quantum simulations.
