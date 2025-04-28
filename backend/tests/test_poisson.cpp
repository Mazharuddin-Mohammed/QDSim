#include <catch2/catch.hpp>
#include "poisson.h"

TEST_CASE("Poisson solver", "[poisson]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 1);
    auto epsilon_r = [](double x, double y) { return 12.9; };
    auto rho = [](double x, double y) { return (x < 0) ? -1e24 * 1.602e-19 : 1e24 * 1.602e-19; };
    PoissonSolver poisson(mesh, epsilon_r, rho);
    poisson.solve(0.0, 1.5);
    auto phi = poisson.get_potential();
    REQUIRE(phi.size() == mesh.get_num_nodes());
}