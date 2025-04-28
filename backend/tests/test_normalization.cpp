#include <catch2/catch.hpp>
#include "normalization.h"

TEST_CASE("Delta normalization", "[normalization]") {
    Mesh mesh(1e-7, 1e-7, 5, 5);
    Normalizer norm(mesh, 0.09 * 9.11e-31, 1.054e-34);
    Eigen::VectorXd psi = Eigen::VectorXd::Ones(mesh.getNumNodes());
    auto norm_psi = norm.deltaNormalize(psi, 0.1 * 1.602e-19);
    REQUIRE(norm_psi.norm() > 0);
}