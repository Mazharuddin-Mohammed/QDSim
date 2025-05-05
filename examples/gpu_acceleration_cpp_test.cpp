/**
 * @file gpu_acceleration_cpp_test.cpp
 * @brief Test program for GPU acceleration in QDSim.
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
#include <fstream>
#include <iomanip>

#include "mesh.h"
#include "schrodinger.h"
#include "gpu_accelerator.h"

// Function to measure execution time
template<typename Func>
double measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

// Function to save eigenvalues to a file
void save_eigenvalues(const std::string& filename, const std::vector<double>& eigenvalues) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << std::setprecision(15);
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        file << i << " " << eigenvalues[i] << std::endl;
    }
}

// Function to save eigenvectors to a file
void save_eigenvectors(const std::string& filename, const Mesh& mesh,
                      const std::vector<Eigen::VectorXcd>& eigenvectors) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << std::setprecision(15);

    // Get mesh nodes
    const auto& nodes = mesh.getNodes();

    // Write header
    file << "x y";
    for (size_t i = 0; i < eigenvectors.size(); ++i) {
        file << " psi" << i;
    }
    file << std::endl;

    // Write data
    for (size_t i = 0; i < nodes.size(); ++i) {
        file << nodes[i][0] << " " << nodes[i][1];
        for (size_t j = 0; j < eigenvectors.size(); ++j) {
            if (i < eigenvectors[j].size()) {
                file << " " << std::abs(eigenvectors[j](i));
            } else {
                file << " 0.0";
            }
        }
        file << std::endl;
    }
}

int main() {
    std::cout << "GPU Acceleration Test for QDSim" << std::endl;
    std::cout << "===============================" << std::endl;

    // Test parameters
    const double Lx = 100.0;  // nm
    const double Ly = 100.0;  // nm
    const int nx_small = 51;
    const int ny_small = 51;
    const int nx_large = 101;
    const int ny_large = 101;
    const int num_eigenvalues = 10;

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

    // Test with linear elements (P1) - small mesh
    std::cout << "\nTesting with linear elements (P1) - small mesh..." << std::endl;

    // Create mesh
    Mesh mesh_p1_small(Lx, Ly, nx_small, ny_small, 1);

    // CPU test
    std::cout << "  Running CPU test..." << std::endl;
    SchrodingerSolver solver_cpu_p1_small(mesh_p1_small, m_star, V, false);

    std::vector<double> eigenvalues_cpu_p1_small;
    std::vector<Eigen::VectorXcd> eigenvectors_cpu_p1_small;

    double cpu_time_p1_small = measure_time([&]() {
        auto result = solver_cpu_p1_small.solve(num_eigenvalues);
        eigenvalues_cpu_p1_small = result.first;

        // Convert VectorXd to VectorXcd
        eigenvectors_cpu_p1_small.resize(result.second.size());
        for (size_t i = 0; i < result.second.size(); ++i) {
            eigenvectors_cpu_p1_small[i] = result.second[i].cast<std::complex<double>>();
        }
    });

    std::cout << "  CPU time: " << cpu_time_p1_small << " seconds" << std::endl;

    // GPU test
    std::cout << "  Running GPU test..." << std::endl;
    SchrodingerSolver solver_gpu_p1_small(mesh_p1_small, m_star, V, true);

    std::vector<double> eigenvalues_gpu_p1_small;
    std::vector<Eigen::VectorXcd> eigenvectors_gpu_p1_small;

    double gpu_time_p1_small = measure_time([&]() {
        auto result = solver_gpu_p1_small.solve(num_eigenvalues);
        eigenvalues_gpu_p1_small = result.first;

        // Convert VectorXd to VectorXcd
        eigenvectors_gpu_p1_small.resize(result.second.size());
        for (size_t i = 0; i < result.second.size(); ++i) {
            eigenvectors_gpu_p1_small[i] = result.second[i].cast<std::complex<double>>();
        }
    });

    std::cout << "  GPU time: " << gpu_time_p1_small << " seconds" << std::endl;
    std::cout << "  Speedup: " << cpu_time_p1_small / gpu_time_p1_small << "x" << std::endl;

    // Test with linear elements (P1) - large mesh
    std::cout << "\nTesting with linear elements (P1) - large mesh..." << std::endl;

    // Create mesh
    Mesh mesh_p1_large(Lx, Ly, nx_large, ny_large, 1);

    // CPU test
    std::cout << "  Running CPU test..." << std::endl;
    SchrodingerSolver solver_cpu_p1_large(mesh_p1_large, m_star, V, false);

    std::vector<double> eigenvalues_cpu_p1_large;
    std::vector<Eigen::VectorXcd> eigenvectors_cpu_p1_large;

    double cpu_time_p1_large = measure_time([&]() {
        auto result = solver_cpu_p1_large.solve(num_eigenvalues);
        eigenvalues_cpu_p1_large = result.first;

        // Convert VectorXd to VectorXcd
        eigenvectors_cpu_p1_large.resize(result.second.size());
        for (size_t i = 0; i < result.second.size(); ++i) {
            eigenvectors_cpu_p1_large[i] = result.second[i].cast<std::complex<double>>();
        }
    });

    std::cout << "  CPU time: " << cpu_time_p1_large << " seconds" << std::endl;

    // GPU test
    std::cout << "  Running GPU test..." << std::endl;
    SchrodingerSolver solver_gpu_p1_large(mesh_p1_large, m_star, V, true);

    std::vector<double> eigenvalues_gpu_p1_large;
    std::vector<Eigen::VectorXcd> eigenvectors_gpu_p1_large;

    double gpu_time_p1_large = measure_time([&]() {
        auto result = solver_gpu_p1_large.solve(num_eigenvalues);
        eigenvalues_gpu_p1_large = result.first;

        // Convert VectorXd to VectorXcd
        eigenvectors_gpu_p1_large.resize(result.second.size());
        for (size_t i = 0; i < result.second.size(); ++i) {
            eigenvectors_gpu_p1_large[i] = result.second[i].cast<std::complex<double>>();
        }
    });

    std::cout << "  GPU time: " << gpu_time_p1_large << " seconds" << std::endl;
    std::cout << "  Speedup: " << cpu_time_p1_large / gpu_time_p1_large << "x" << std::endl;

    // Test with quadratic elements (P2) - small mesh
    std::cout << "\nTesting with quadratic elements (P2) - small mesh..." << std::endl;

    // Create mesh
    Mesh mesh_p2_small(Lx, Ly, nx_small, ny_small, 2);

    // CPU test
    std::cout << "  Running CPU test..." << std::endl;
    SchrodingerSolver solver_cpu_p2_small(mesh_p2_small, m_star, V, false);

    std::vector<double> eigenvalues_cpu_p2_small;
    std::vector<Eigen::VectorXcd> eigenvectors_cpu_p2_small;

    double cpu_time_p2_small = measure_time([&]() {
        auto result = solver_cpu_p2_small.solve(num_eigenvalues);
        eigenvalues_cpu_p2_small = result.first;

        // Convert VectorXd to VectorXcd
        eigenvectors_cpu_p2_small.resize(result.second.size());
        for (size_t i = 0; i < result.second.size(); ++i) {
            eigenvectors_cpu_p2_small[i] = result.second[i].cast<std::complex<double>>();
        }
    });

    std::cout << "  CPU time: " << cpu_time_p2_small << " seconds" << std::endl;

    // GPU test
    std::cout << "  Running GPU test..." << std::endl;
    SchrodingerSolver solver_gpu_p2_small(mesh_p2_small, m_star, V, true);

    std::vector<double> eigenvalues_gpu_p2_small;
    std::vector<Eigen::VectorXcd> eigenvectors_gpu_p2_small;

    double gpu_time_p2_small = measure_time([&]() {
        auto result = solver_gpu_p2_small.solve(num_eigenvalues);
        eigenvalues_gpu_p2_small = result.first;

        // Convert VectorXd to VectorXcd
        eigenvectors_gpu_p2_small.resize(result.second.size());
        for (size_t i = 0; i < result.second.size(); ++i) {
            eigenvectors_gpu_p2_small[i] = result.second[i].cast<std::complex<double>>();
        }
    });

    std::cout << "  GPU time: " << gpu_time_p2_small << " seconds" << std::endl;
    std::cout << "  Speedup: " << cpu_time_p2_small / gpu_time_p2_small << "x" << std::endl;

    // Print summary
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Linear elements (P1) - small mesh:" << std::endl;
    std::cout << "    CPU time: " << cpu_time_p1_small << " seconds" << std::endl;
    std::cout << "    GPU time: " << gpu_time_p1_small << " seconds" << std::endl;
    std::cout << "    Speedup: " << cpu_time_p1_small / gpu_time_p1_small << "x" << std::endl;

    std::cout << "  Linear elements (P1) - large mesh:" << std::endl;
    std::cout << "    CPU time: " << cpu_time_p1_large << " seconds" << std::endl;
    std::cout << "    GPU time: " << gpu_time_p1_large << " seconds" << std::endl;
    std::cout << "    Speedup: " << cpu_time_p1_large / gpu_time_p1_large << "x" << std::endl;

    std::cout << "  Quadratic elements (P2) - small mesh:" << std::endl;
    std::cout << "    CPU time: " << cpu_time_p2_small << " seconds" << std::endl;
    std::cout << "    GPU time: " << gpu_time_p2_small << " seconds" << std::endl;
    std::cout << "    Speedup: " << cpu_time_p2_small / gpu_time_p2_small << "x" << std::endl;

    // Save results
    std::cout << "\nSaving results..." << std::endl;

    // Save eigenvalues
    save_eigenvalues("eigenvalues_cpu_p1_small.txt", eigenvalues_cpu_p1_small);
    save_eigenvalues("eigenvalues_gpu_p1_small.txt", eigenvalues_gpu_p1_small);
    save_eigenvalues("eigenvalues_cpu_p1_large.txt", eigenvalues_cpu_p1_large);
    save_eigenvalues("eigenvalues_gpu_p1_large.txt", eigenvalues_gpu_p1_large);
    save_eigenvalues("eigenvalues_cpu_p2_small.txt", eigenvalues_cpu_p2_small);
    save_eigenvalues("eigenvalues_gpu_p2_small.txt", eigenvalues_gpu_p2_small);

    // Save eigenvectors
    save_eigenvectors("eigenvectors_gpu_p1_large.txt", mesh_p1_large, eigenvectors_gpu_p1_large);
    save_eigenvectors("eigenvectors_gpu_p2_small.txt", mesh_p2_small, eigenvectors_gpu_p2_small);

    std::cout << "\nDone!" << std::endl;

    return 0;
}
