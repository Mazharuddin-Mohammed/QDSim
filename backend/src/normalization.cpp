/**
 * @file normalization.cpp
 * @brief Implementation of the Normalizer class for wavefunction normalization.
 *
 * This file contains the implementation of the Normalizer class, which provides
 * methods for normalizing wavefunctions according to various normalization schemes,
 * including delta-function normalization for scattering states.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "normalization.h"
#include <cmath>

Normalizer::Normalizer(const Mesh& mesh, double m_star, double hbar)
    : mesh(mesh), m_star(m_star), hbar(hbar) {}

double Normalizer::computeAsymptoticAmplitude(const Eigen::VectorXd& psi) const {
    double sum = 0.0;
    int count = 0;
    const auto& nodes = mesh.getNodes();
    double Lx = nodes.back()(0), Ly = nodes.back()(1);
    for (int i = 0; i < psi.size(); ++i) {
        double x = nodes[i](0), y = nodes[i](1);
        if (std::abs(x) > 0.8 * Lx / 2 || std::abs(y) > 0.8 * Ly / 2) {
            sum += psi(i) * psi(i);
            ++count;
        }
    }
    return std::sqrt(sum / count);
}

Eigen::VectorXd Normalizer::deltaNormalize(const Eigen::VectorXd& psi, double E) const {
    // Constants
    const double e_charge = 1.602e-19; // Electron charge in C

    // Convert energy to Joules if it's in eV
    double E_J = (E < 1.0) ? E * e_charge : E; // Assume E is in eV if < 1.0

    // Calculate density of states (DOS)
    double rho = m_star / (M_PI * hbar * hbar); // DOS in J^-1·m^-2

    // Get asymptotic amplitude
    double A = computeAsymptoticAmplitude(psi);

    // Get domain size (use max dimension for better scaling)
    const auto& nodes = mesh.getNodes();
    double Lx = 0.0, Ly = 0.0;
    for (const auto& node : nodes) {
        Lx = std::max(Lx, std::abs(node(0)));
        Ly = std::max(Ly, std::abs(node(1)));
    }
    double L = 2.0 * std::max(Lx, Ly); // Full domain size

    // Normalize the wavefunction
    // For bound states (A very small), use standard normalization
    if (A < 1e-10) {
        double norm = 0.0;
        for (int i = 0; i < psi.size(); ++i) {
            norm += psi(i) * psi(i);
        }
        norm = std::sqrt(norm);
        return psi / (norm > 0 ? norm : 1.0);
    }

    // For continuum states, use delta normalization
    return std::sqrt(rho) * psi / (A * L);
}