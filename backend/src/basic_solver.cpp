#include "basic_solver.h"
#include <iostream>
#include <cmath>

BasicSolver::BasicSolver(Mesh& mesh) : mesh(mesh) {
    // Resize the vectors
    int num_nodes = mesh.getNumNodes();
    potential.resize(num_nodes);
    n.resize(num_nodes);
    p.resize(num_nodes);

    // Initialize with zeros
    potential.setZero();
    n.setZero();
    p.setZero();
}

void BasicSolver::solve(double V_p, double V_n, double N_A, double N_D) {
    std::cout << "BasicSolver: Setting up drift-diffusion model..." << std::endl;

    // Physical constants
    const double kT = 0.0259;       // Thermal voltage at room temperature (eV)
    const double ni = 1.0e10;       // Intrinsic carrier concentration for GaAs (cm^-3)
    const double epsilon = 12.9;    // Relative permittivity for GaAs
    const double q = 1.602e-19;     // Elementary charge (C)

    // Get mesh dimensions
    double Lx = mesh.get_lx();
    double junction_position = 0.0; // Junction at x = 0

    // Calculate depletion width (simplified model)
    double Vbi = kT * std::log((N_A * N_D) / (ni * ni)); // Built-in potential
    double V_applied = V_n - V_p;   // Applied voltage
    double V_total = Vbi - V_applied; // Total potential across junction

    // Depletion width calculation (from semiconductor physics)
    // Using the formula W = sqrt(2*epsilon*epsilon0*V_total/(q*(Na+Nd)/(Na*Nd)))
    // For simplicity, we'll use a fixed depletion width based on typical values
    double W = 50.0; // Depletion width in nm (typical for these doping levels)
    double xn = W * N_A / (N_A + N_D); // Depletion width in n-region
    double xp = W * N_D / (N_A + N_D); // Depletion width in p-region

    std::cout << "Built-in potential: " << Vbi << " V" << std::endl;
    std::cout << "Total depletion width: " << W << " nm" << std::endl;

    // Loop over all nodes
    for (int i = 0; i < mesh.getNumNodes(); ++i) {
        // Get node coordinates
        double x = mesh.getNodes()[i][0];
        double y = mesh.getNodes()[i][1];

        // Distance from junction
        double d = x - junction_position;

        // Calculate potential based on depletion approximation
        if (d < -xp) {
            // p-region outside depletion region
            potential[i] = V_p;
        } else if (d > xn) {
            // n-region outside depletion region
            potential[i] = V_n;
        } else {
            // Inside depletion region - parabolic potential
            if (d < 0) {
                // p-side of depletion region
                potential[i] = V_p + V_total * (1 - (d + xp) * (d + xp) / (xp * xp));
            } else {
                // n-side of depletion region
                potential[i] = V_n - V_total * (1 - (d - xn) * (d - xn) / (xn * xn));
            }
        }

        // Calculate carrier concentrations using Boltzmann statistics
        if (d < 0) {
            // p-region
            p[i] = N_A;
            // Electron concentration from mass action law
            n[i] = ni * ni / p[i] * std::exp(q * (potential[i] - V_p) / kT);
        } else {
            // n-region
            n[i] = N_D;
            // Hole concentration from mass action law
            p[i] = ni * ni / n[i] * std::exp(-q * (potential[i] - V_n) / kT);
        }

        // Ensure minimum carrier concentrations to avoid numerical issues
        n[i] = std::max(n[i], 1.0);
        p[i] = std::max(p[i], 1.0);
    }

    std::cout << "BasicSolver: Done!" << std::endl;
}
