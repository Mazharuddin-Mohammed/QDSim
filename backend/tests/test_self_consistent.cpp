#include <catch2/catch.hpp>
#include "self_consistent.h"

TEST_CASE("Self-consistent solver", "[self_consistent]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 1);
    auto epsilon_r = [](double x, double y) { return 12.9; };
    auto rho = [](double x, double y, const Eigen::VectorXd& n, const Eigen::VectorXd& p) {
        return 1.602e-19 * (p[0] - n[0]);
    };
    auto n_conc = [](double x, double y, double phi, const Materials::Material& mat) {
        return mat.N_c * std::exp(-phi / 0.0259);
    };
    auto p_conc = [](double x, double y, double phi, const Materials::Material& mat) {
        return mat.N_v * std::exp(phi / 0.0259);
    };
    auto mu_n = [](double x, double y, const Materials::Material& mat) { return mat.mu_n; };
    auto mu_p = [](double x, double y, const Materials::Material& mat) { return mat.mu_p; };
    SelfConsistentSolver sc_solver(mesh, epsilon_r, rho, n_conc, p_conc, mu_n, mu_p);
    sc_solver.solve(0.0, 1.5, 1e24, 1e24);
    auto phi = sc_solver.get_potential();
    REQUIRE(phi.size() == mesh.get_num_nodes());
}