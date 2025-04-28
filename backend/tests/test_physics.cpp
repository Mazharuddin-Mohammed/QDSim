#include <catch2/catch.hpp>
#include "physics.h"

TEST_CASE("Physics functions", "[physics]") {
    REQUIRE(Physics::effectiveMass(0, 0, 0.09 * 9.11e-31, 0.1 * 9.11e-31, 1e-8) == 0.1 * 9.11e-31);
    REQUIRE(Physics::potential(0, 0, 0.5 * 1.602e-19, 1e-8, 1.5 * 1.602e-19, 1e-7) == Approx(-0.5 * 1.602e-19));
}