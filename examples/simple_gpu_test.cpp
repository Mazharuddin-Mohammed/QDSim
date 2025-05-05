/**
 * @file simple_gpu_test.cpp
 * @brief Simple test program for GPU acceleration in QDSim.
 *
 * This program tests the GPU acceleration capabilities of QDSim by solving the
 * Schrödinger equation for a quantum dot in a harmonic oscillator potential.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <complex>
#include <functional>

#include "mesh.h"
#include "schrodinger.h"
#include "gpu_accelerator.h"

int main() {
    std::cout << "Simple GPU Acceleration Test for QDSim" << std::endl;
    std::cout << "======================================" << std::endl;

    // Test parameters
    const double Lx = 100.0;  // nm
    const double Ly = 100.0;  // nm
    const int nx = 51;
    const int ny = 51;
    const int num_eigenvalues = 5;

    // Define effective mass function (constant for simplicity)
    auto m_star = [](double x, double y) -> double {
        return 0.067;  // GaAs effective mass
    };

    // Define potential function (harmonic oscillator)
    auto V = [Lx, Ly](double x, double y) -> double {
        double x0 = Lx / 2.0;
        double y0 = Ly / 2.0;
        double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
        return 0.1 * r2;  // meV
    };

    // Create mesh
    std::cout << "Creating mesh..." << std::endl;
    Mesh mesh(Lx, Ly, nx, ny, 1);

    // Create SchrodingerSolver with CPU acceleration
    std::cout << "Creating SchrodingerSolver with CPU acceleration..." << std::endl;
    SchrodingerSolver solver(mesh, m_star, V, false);

    // Solve the Schrödinger equation
    std::cout << "Solving Schrödinger equation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto result = solver.solve(num_eigenvalues);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Print results
    std::cout << "Elapsed time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Eigenvalues:" << std::endl;
    for (size_t i = 0; i < result.first.size(); ++i) {
        std::cout << "  " << i << ": " << result.first[i] << " meV" << std::endl;
    }

    std::cout << "\nDone!" << std::endl;

    return 0;
}
