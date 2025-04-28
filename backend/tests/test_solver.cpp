#include <catch2/catch.hpp>
#include "solver.h"
#include "fem.h"

TEST_CASE("Eigenvalue solver", "[solver]") {
    Mesh mesh(1e-7, 1e-7, 5, 5);
    auto m_star = [](double x, double y) { return 0.09 * 9.11e-31; };
    auto V = [](double x, double y) { return 0.0; };
    auto cap = [](double x, double y) { return 0.0; };
    FEMSolver fem(mesh, m_star, V, cap);
    fem.assembleMatrices();
    EigenSolver solver(fem);
    solver.solve(2);
    REQUIRE(solver.getEigenvalues().size() == 2);
}