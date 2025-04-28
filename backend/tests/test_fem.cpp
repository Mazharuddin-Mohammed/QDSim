#include <catch2/catch.hpp>
#include "fem.h"
#include "physics.h"

TEST_CASE("FEM matrix assembly", "[fem]") {
    Mesh mesh(1e-7, 1e-7, 10, 10);
    auto m_star = [](double x, double y) { return 0.09 * 9.11e-31; };
    auto V = [](double x, double y) { return 0.0; };
    auto cap = [](double x, double y) { return 0.0; };
    FEMSolver fem(mesh, m_star, V, cap);
    fem.assembleMatrices();
    REQUIRE(fem.getStiffnessMatrix().nonZeros() > 0);
}