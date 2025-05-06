# QDSim: Quantum Dot Simulator

QDSim is a high-performance 2D Quantum Dot (QD) Simulator implemented in C++ and Python, designed to solve the time-independent Schrödinger equation for quantum dots in semiconductor nanostructures.

## Features

- **Finite Element Method (FEM)**: Solve the Schrödinger equation using the finite element method with high-order elements (P1, P2, P3).
- **Adaptive Mesh Refinement**: Automatically refine the mesh in regions of interest for improved accuracy.
- **Multiple Potentials**: Support for various potential types, including harmonic, square well, and custom potentials.
- **Self-Consistent Solver**: Solve the coupled Poisson-Schrödinger equations self-consistently.
- **GPU Acceleration**: Utilize GPU acceleration for faster simulations.
- **Memory Optimization**: Advanced memory optimization techniques for large-scale simulations.
- **Physically Accurate Models**: Comprehensive physical models for realistic simulations.
- **Visualization Tools**: Built-in visualization tools for simulation results.
- **Python and C++ Interface**: Use QDSim from Python or C++ for maximum flexibility.

## Documentation Structure

This documentation is organized into several sections:

- **User Guide**: Instructions for installing and using QDSim.
- **API Reference**: Detailed documentation of the QDSim API.
- **Theory**: Mathematical and physical background for the simulations.
- **Examples**: Example simulations demonstrating QDSim's capabilities.
- **Developer Guide**: Information for contributors to QDSim.

## Getting Started

To get started with QDSim, see the [Installation](@ref installation) and [Quick Start](@ref quick_start) guides.

## License

QDSim is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use QDSim in your research, please cite:

```bibtex
@software{qdsim,
  author = {Mohammed, Mazharuddin},
  title = {QDSim: Quantum Dot Simulator},
  url = {https://github.com/username/qdsim},
  version = {1.0.0},
  year = {2023}
}
```

## Acknowledgments

QDSim was developed with support from:

- The Quantum Computing Research Group
- The Semiconductor Physics Laboratory
- The High-Performance Computing Center

## Contact

For questions, issues, or collaborations, please contact:

- Dr. Mazharuddin Mohammed: [email@example.com](mailto:email@example.com)
- GitHub Issues: [https://github.com/username/qdsim/issues](https://github.com/username/qdsim/issues)
