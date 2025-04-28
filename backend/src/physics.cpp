#include "physics.h"
#include <cmath>

namespace Physics {

double effective_mass(double x, double y, const Materials::Material& qd_mat, 
                      const Materials::Material& matrix_mat, double R) {
    return (x * x + y * y <= R * R) ? qd_mat.m_e : matrix_mat.m_e;
}

double potential(double x, double y, const Materials::Material& qd_mat, 
                 const Materials::Material& matrix_mat, double R, const std::string& type, 
                 const Eigen::VectorXd& phi) {
    double V_qd = 0.0;
    if (type == "square") {
        if (x * x + y * y <= R * R) {
            V_qd = 0.0;
        } else {
            V_qd = qd_mat.Delta_E_c * 1.602e-19; // Convert eV to J
        }
    } else if (type == "gaussian") {
        V_qd = qd_mat.Delta_E_c * 1.602e-19 * std::exp(-(x * x + y * y) / (2 * R * R));
    }
    // Add electrostatic potential (V_elec = q * phi, q = electron charge)
    double V_elec = 1.602e-19 * phi[0]; // Simplified; interpolate phi at (x,y)
    return V_qd + V_elec;
}

double epsilon_r(double x, double y, const Materials::Material& p_mat, 
                 const Materials::Material& n_mat) {
    // Assume p-side at x < 0, n-side at x > 0
    return (x < 0) ? p_mat.epsilon_r : n_mat.epsilon_r;
}

double charge_density(double x, double y, double N_A, double N_D, double W_d) {
    const double q = 1.602e-19; // C
    double x_p = W_d * N_D / (N_A + N_D); // p-side depletion
    double x_n = W_d * N_A / (N_A + N_D); // n-side depletion
    if (x >= -x_p && x <= x_n) {
        return (x < 0) ? -q * N_A : q * N_D;
    }
    return 0.0;
}

double cap(double x, double y, double eta, double Lx, double Ly, double d) {
    double eta_x = (std::abs(x) > Lx / 2 - d) ? eta * std::pow((std::abs(x) - Lx / 2) / d, 2) : 0.0;
    double eta_y = (std::abs(y) > Ly / 2 - d) ? eta * std::pow((std::abs(y) - Ly / 2) / d, 2) : 0.0;
    return std::max(eta_x, eta_y);
}

} // namespace Physics