/**
 * @file poisson_dd_test.cpp
 * @brief Test program for the FullPoissonDriftDiffusionSolver.
 *
 * This program demonstrates the use of the FullPoissonDriftDiffusionSolver
 * to simulate a P-N junction diode with a quantum dot.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <chrono>

#include "mesh.h"
#include "materials.h"
#include "full_poisson_dd_solver.h"

// Function to save the results to a file
void save_results(const std::string& filename,
                 const Mesh& mesh,
                 const Eigen::VectorXd& phi,
                 const Eigen::VectorXd& n,
                 const Eigen::VectorXd& p,
                 const std::vector<Eigen::Vector2d>& E_field,
                 const std::vector<Eigen::Vector2d>& J_n,
                 const std::vector<Eigen::Vector2d>& J_p) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write header
    file << "x,y,phi,n,p,Ex,Ey,Jnx,Jny,Jpx,Jpy" << std::endl;

    // Write data
    for (size_t i = 0; i < mesh.getNumNodes(); ++i) {
        const auto& node = mesh.getNodes()[i];
        file << node[0] << "," << node[1] << ","
             << phi[i] << "," << n[i] << "," << p[i] << ","
             << E_field[i][0] << "," << E_field[i][1] << ","
             << J_n[i][0] << "," << J_n[i][1] << ","
             << J_p[i][0] << "," << J_p[i][1] << std::endl;
    }

    file.close();
}

int main() {
    // Create a mesh for the P-N junction diode
    double Lx = 200.0; // nm
    double Ly = 100.0; // nm
    int nx = 51;
    int ny = 26;
    Mesh mesh(Lx, Ly, nx, ny, 1); // Linear elements

    // Define the doping profile
    auto doping_profile = [](double x, double y) {
        // P-N junction at x = 0
        if (x < 0.0) {
            return -1e17; // P-type (acceptors)
        } else {
            return 1e17; // N-type (donors)
        }
    };

    // Define the relative permittivity
    auto epsilon_r = [](double x, double y) {
        return 12.9; // GaAs
    };

    // Define the quantum dot potential
    auto qd_potential = [Lx, Ly](double x, double y) {
        // Quantum dot at the center of the device
        double x0 = 0.0;
        double y0 = Ly / 2.0;
        double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
        double sigma = 10.0; // nm
        double depth = 0.5; // eV

        return -depth * std::exp(-r2 / (2.0 * sigma * sigma));
    };

    // Create the solver
    FullPoissonDriftDiffusionSolver solver(mesh, epsilon_r, doping_profile);

    // Set the carrier statistics model
    solver.set_carrier_statistics_model(false); // Use Boltzmann statistics

    // Define the materials for the heterojunction
    Materials::Material gaas;
    gaas.N_c = 4.7e17; // Effective density of states in conduction band (cm^-3)
    gaas.N_v = 7.0e18; // Effective density of states in valence band (cm^-3)
    gaas.E_g = 1.424;  // Band gap (eV)
    gaas.mu_n = 8500;  // Electron mobility (cm^2/V·s)
    gaas.mu_p = 400;   // Hole mobility (cm^2/V·s)
    gaas.epsilon_r = 12.9; // Relative permittivity
    gaas.m_e = 0.067;  // Effective electron mass (m0)
    gaas.m_h = 0.45;   // Effective hole mass (m0)
    // Note: Material struct doesn't have chi, N_a, or N_d fields
    // We'll use these values in our doping profile function instead

    // Set the heterojunction
    std::vector<Materials::Material> materials = {gaas};
    std::vector<std::function<bool(double, double)>> regions = {
        [](double x, double y) { return true; } // GaAs everywhere
    };
    solver.set_heterojunction(materials, regions);

    // Set the mobility models
    auto mu_n_model = [](double x, double y, double E, const Materials::Material& mat) {
        // Field-dependent mobility model
        double mu0 = mat.mu_n;
        double vsat = 1e7; // Saturation velocity (cm/s)
        double beta = 1.0; // Exponent

        return mu0 / std::pow(1.0 + std::pow(mu0 * E / vsat, beta), 1.0 / beta);
    };

    auto mu_p_model = [](double x, double y, double E, const Materials::Material& mat) {
        // Field-dependent mobility model
        double mu0 = mat.mu_p;
        double vsat = 1e7; // Saturation velocity (cm/s)
        double beta = 1.0; // Exponent

        return mu0 / std::pow(1.0 + std::pow(mu0 * E / vsat, beta), 1.0 / beta);
    };

    solver.set_mobility_models(mu_n_model, mu_p_model);

    // Set the generation-recombination model
    auto g_r_model = [](double x, double y, double n, double p, const Materials::Material& mat) {
        // Constants
        const double kT = 0.0259; // eV at 300K
        const double ni = std::sqrt(mat.N_c * mat.N_v) * std::exp(-mat.E_g / (2.0 * kT));
        const double tau_n = 1e-9; // Electron lifetime (s)
        const double tau_p = 1e-9; // Hole lifetime (s)

        // SRH recombination rate
        double R_SRH = (n * p - ni * ni) / (tau_p * (n + ni) + tau_n * (p + ni));

        return R_SRH;
    };

    solver.set_generation_recombination_model(g_r_model);

    // Solve the coupled Poisson-drift-diffusion equations for different bias voltages
    std::vector<double> bias_voltages = {-1.0, -0.5, 0.0, 0.5, 1.0};

    for (double bias : bias_voltages) {
        std::cout << "Solving for bias voltage = " << bias << " V" << std::endl;

        // Start the timer
        auto start_time = std::chrono::high_resolution_clock::now();

        // Solve the coupled equations
        solver.solve(0.0, bias, 1e-6, 100);

        // Stop the timer
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Solution time: " << duration.count() << " ms" << std::endl;

        // Save the results
        std::string filename = "pn_junction_bias_" + std::to_string(bias) + ".csv";
        save_results(filename, mesh, solver.get_potential(), solver.get_electron_concentration(),
                    solver.get_hole_concentration(), solver.get_electric_field(),
                    solver.get_electron_current_density(), solver.get_hole_current_density());

        std::cout << "Results saved to " << filename << std::endl;
    }

    return 0;
}
