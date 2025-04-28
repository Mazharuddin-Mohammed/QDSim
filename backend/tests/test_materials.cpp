#include <catch2/catch.hpp>
#include "materials.h"

TEST_CASE("Material database", "[materials]") {
    Materials::MaterialDatabase db;
    SECTION("Valid material") {
        auto mat = db.get_material("GaAs");
        REQUIRE(mat.m_e == Approx(0.067 * 9.11e-31));
        REQUIRE(mat.E_g == Approx(1.43));
        REQUIRE(mat.epsilon_r == Approx(12.9));
    }
    SECTION("Invalid material") {
        REQUIRE_THROWS_AS(db.get_material("Unknown"), std::runtime_error);
    }
}