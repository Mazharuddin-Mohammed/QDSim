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
    double rho = m_star / (M_PI * hbar * hbar); // DOS in J^-1·m^-2
    double A = computeAsymptoticAmplitude(psi);
    double L = mesh.getNodes().back()(0); // Domain size
    return std::sqrt(rho) * psi / (A * L);
}