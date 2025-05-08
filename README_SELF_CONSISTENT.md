# Self-Consistent Poisson-Drift-Diffusion Solver

This document describes the implementation of the self-consistent Poisson-drift-diffusion solver in the QDSim project.

## Overview

The self-consistent Poisson-drift-diffusion solver is used to simulate the electrostatic potential and carrier concentrations in semiconductor devices. It solves the Poisson equation and the drift-diffusion equations self-consistently, which means that the potential and carrier concentrations are updated iteratively until convergence.

## Implementation

We have implemented three different solvers:

1. `BasicSolver`: A simple solver that implements a drift-diffusion model for a p-n junction.
2. `SimpleSelfConsistentSolver`: A self-consistent solver that uses the `PoissonSolver` class.
3. `ImprovedSelfConsistentSolver`: A more robust self-consistent solver with proper error handling.

### BasicSolver

The `BasicSolver` class provides a simple implementation of a p-n junction solver with proper drift-diffusion physics. It calculates the electrostatic potential and carrier concentrations based on the depletion approximation and Boltzmann statistics.

### SimpleSelfConsistentSolver

The `SimpleSelfConsistentSolver` class is a self-consistent solver that uses the `PoissonSolver` class to solve the Poisson equation. It updates the carrier concentrations based on the potential and then recalculates the potential based on the new carrier concentrations. This process is repeated until convergence.

### ImprovedSelfConsistentSolver

The `ImprovedSelfConsistentSolver` class is a more robust self-consistent solver with proper error handling. It builds on the `BasicSolver` to provide a more reliable implementation that avoids segmentation faults and NaN values. It also includes a more sophisticated finite difference solver for the Poisson equation.

## Usage

Here's an example of how to use the `ImprovedSelfConsistentSolver`:

```python
import qdsim_cpp as cpp
import numpy as np

# Create a mesh
Lx = 100.0  # nm
Ly = 50.0   # nm
nx = 100
ny = 50
element_order = 1
mesh = cpp.Mesh(Lx, Ly, nx, ny, element_order)

# Define callback functions
def epsilon_r(x, y):
    """Relative permittivity function."""
    return 12.9  # GaAs

def rho(x, y, n, p):
    """Charge density function."""
    q = 1.602e-19  # Elementary charge in C
    if len(n) == 0 or len(p) == 0:
        return 0.0

    # Find the nearest node
    nodes = np.array(mesh.get_nodes())
    distances = np.sqrt((nodes[:, 0] - x)**2 + (nodes[:, 1] - y)**2)
    idx = np.argmin(distances)

    # Return the charge density at the nearest node
    return q * (p[idx] - n[idx])

# Create the ImprovedSelfConsistentSolver
solver = cpp.create_improved_self_consistent_solver(mesh, epsilon_r, rho)

# Solve the self-consistent Poisson-drift-diffusion equations
V_p = 0.0  # Voltage at the p-contact
V_n = 0.7  # Voltage at the n-contact (forward bias)
N_A = 1e18  # Acceptor doping concentration
N_D = 1e18  # Donor doping concentration
tolerance = 1e-6
max_iter = 100

solver.solve(V_p, V_n, N_A, N_D, tolerance, max_iter)

# Get the results
potential = np.array(solver.get_potential())
n = np.array(solver.get_n())
p = np.array(solver.get_p())

# Process the results
# ...

# Clear the callbacks when you're done
cpp.clear_callbacks()
```

## Known Issues

1. The `SimpleSelfConsistentSolver` class may produce segmentation faults or NaN values in some cases.
2. The finite difference solver in the `ImprovedSelfConsistentSolver` class is a simplified version and may not be accurate for complex geometries.

## Memory Management

To avoid memory leaks and segmentation faults, it's important to clear the Python callbacks when you're done with the solver. You can do this by calling the `clear_callbacks` function:

```python
import qdsim_cpp as cpp

# Create and use the solver
# ...

# Clear the callbacks when you're done
cpp.clear_callbacks()
```

This is especially important when using matplotlib for visualization, as the Python callbacks may be garbage collected before the C++ code is done with them.

## Future Improvements

1. Implement a more accurate finite element solver for the Poisson equation.
2. Improve the drift-diffusion model to include more physical effects such as recombination and generation.
3. Fix the segmentation faults when using matplotlib for visualization.
4. Implement a more efficient solver for the linear system in the Poisson equation.
5. Add support for more complex device structures such as MOSFETs and HEMTs.

## References

1. S. Selberherr, "Analysis and Simulation of Semiconductor Devices", Springer-Verlag, 1984.
2. D. Vasileska, S. M. Goodnick, and G. Klimeck, "Computational Electronics: Semiclassical and Quantum Device Modeling and Simulation", CRC Press, 2010.
3. M. S. Lundstrom, "Fundamentals of Carrier Transport", Cambridge University Press, 2000.
