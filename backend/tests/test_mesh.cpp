#include <catch2/catch.hpp>
#include "mesh.h"

TEST_CASE("Mesh generation", "[mesh]") {
    Mesh mesh(1.0, 1.0, 10, 10);
    REQUIRE(mesh.getNodes().size() == 121);
    REQUIRE(mesh.getElements().size() == 200);
}