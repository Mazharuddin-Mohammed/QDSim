/**
 * @file test_mobility_models.cpp
 * @brief Tests for the MobilityModels module.
 *
 * This file contains unit tests for the MobilityModels module,
 * verifying its functionality, accuracy, and physical correctness.
 *
 * @author Dr. Mazharuddin Mohammed
 * @date 2023-07-15
 */

#include <catch2/catch.hpp>
#include "mobility_models.h"
#include "physical_constants.h"
#include <cmath>

/**
 * @brief Test temperature-dependent mobility models.
 *
 * This test verifies that the temperature-dependent mobility models
 * are computed correctly for various temperatures.
 */
TEST_CASE("Temperature-dependent mobility models", "[mobility_models]") {
    // Test parameters
    double mu_0 = 1400.0;  // Low-field mobility at 300K (cm^2/Vs)
    double alpha = 2.5;  // Temperature exponent
    
    // Test for different temperatures
    SECTION("Different temperatures") {
        std::vector<double> temperatures = {200.0, 250.0, 300.0, 350.0, 400.0};
        
        for (double T : temperatures) {
            // Calculate mobility using temperature-dependent model
            double mu = MobilityModels::temperature_dependent_mobility(mu_0, T, alpha);
            
            // Calculate expected mobility
            double expected = mu_0 * std::pow(300.0 / T, alpha);
            
            REQUIRE(mu == Approx(expected).epsilon(0.01));
        }
    }
    
    // Test reference temperature
    SECTION("Reference temperature") {
        double T_ref = 300.0;  // Reference temperature (K)
        
        // Mobility at reference temperature should be equal to mu_0
        double mu = MobilityModels::temperature_dependent_mobility(mu_0, T_ref, alpha);
        
        REQUIRE(mu == Approx(mu_0).epsilon(0.01));
    }
}

/**
 * @brief Test doping-dependent mobility models.
 *
 * This test verifies that the doping-dependent mobility models
 * are computed correctly for various doping concentrations.
 */
TEST_CASE("Doping-dependent mobility models", "[mobility_models]") {
    // Test Caughey-Thomas model
    SECTION("Caughey-Thomas model") {
        // Test parameters for silicon electrons
        double mu_min = 65.0;  // Minimum mobility (cm^2/Vs)
        double mu_max = 1400.0;  // Maximum mobility (cm^2/Vs)
        double N_ref = 1.3e17;  // Reference doping concentration (cm^-3)
        double alpha = 0.91;  // Exponent
        
        // Test for different doping concentrations
        std::vector<double> doping_values = {1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19};
        
        for (double N : doping_values) {
            // Calculate mobility using Caughey-Thomas model
            double mu = MobilityModels::caughey_thomas_mobility(mu_min, mu_max, N, N_ref, alpha);
            
            // Calculate expected mobility
            double expected = mu_min + (mu_max - mu_min) / (1.0 + std::pow(N / N_ref, alpha));
            
            REQUIRE(mu == Approx(expected).epsilon(0.01));
        }
        
        // Test limiting cases
        // For very low doping, mobility should approach mu_max
        double mu_low = MobilityModels::caughey_thomas_mobility(mu_min, mu_max, 1.0e12, N_ref, alpha);
        REQUIRE(mu_low == Approx(mu_max).epsilon(0.1));
        
        // For very high doping, mobility should approach mu_min
        double mu_high = MobilityModels::caughey_thomas_mobility(mu_min, mu_max, 1.0e21, N_ref, alpha);
        REQUIRE(mu_high == Approx(mu_min).epsilon(0.1));
    }
    
    // Test Arora model
    SECTION("Arora model") {
        // Test parameters for silicon electrons
        double mu_min = 88.0;  // Minimum mobility (cm^2/Vs)
        double mu_max = 1252.0;  // Maximum mobility (cm^2/Vs)
        double N_ref = 1.26e17;  // Reference doping concentration (cm^-3)
        double alpha = 0.88;  // Exponent
        double T = 300.0;  // Temperature (K)
        
        // Test for different doping concentrations
        std::vector<double> doping_values = {1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19};
        
        for (double N : doping_values) {
            // Calculate mobility using Arora model
            double mu = MobilityModels::arora_mobility(mu_min, mu_max, N, N_ref, alpha, T);
            
            // Calculate expected mobility
            double T_n = T / 300.0;
            double mu_min_T = mu_min * std::pow(T_n, -0.57);
            double mu_max_T = mu_max * std::pow(T_n, -2.33);
            double N_ref_T = N_ref * std::pow(T_n, 2.4);
            double expected = mu_min_T + (mu_max_T - mu_min_T) / (1.0 + std::pow(N / N_ref_T, alpha));
            
            REQUIRE(mu == Approx(expected).epsilon(0.01));
        }
    }
}

/**
 * @brief Test field-dependent mobility models.
 *
 * This test verifies that the field-dependent mobility models
 * are computed correctly for various electric fields.
 */
TEST_CASE("Field-dependent mobility models", "[mobility_models]") {
    // Test parameters
    double mu_0 = 1400.0;  // Low-field mobility (cm^2/Vs)
    double v_sat = 1.0e7;  // Saturation velocity (cm/s)
    double beta = 2.0;  // Field exponent
    
    // Test for different electric fields
    SECTION("Different electric fields") {
        std::vector<double> field_values = {1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6};
        
        for (double E : field_values) {
            // Calculate mobility using field-dependent model
            double mu = MobilityModels::field_dependent_mobility(mu_0, E, v_sat, beta);
            
            // Calculate expected mobility
            double expected = mu_0 / std::pow(1.0 + std::pow(mu_0 * E / v_sat, beta), 1.0 / beta);
            
            REQUIRE(mu == Approx(expected).epsilon(0.01));
        }
    }
    
    // Test limiting cases
    SECTION("Limiting cases") {
        // For very low field, mobility should approach mu_0
        double mu_low = MobilityModels::field_dependent_mobility(mu_0, 1.0, v_sat, beta);
        REQUIRE(mu_low == Approx(mu_0).epsilon(0.1));
        
        // For very high field, mobility should approach v_sat / E
        double E_high = 1.0e7;
        double mu_high = MobilityModels::field_dependent_mobility(mu_0, E_high, v_sat, beta);
        REQUIRE(mu_high == Approx(v_sat / E_high).epsilon(0.1));
    }
}

/**
 * @brief Test combined mobility models.
 *
 * This test verifies that the combined mobility models
 * are computed correctly for various conditions.
 */
TEST_CASE("Combined mobility models", "[mobility_models]") {
    // Test parameters
    double mu_min = 65.0;  // Minimum mobility (cm^2/Vs)
    double mu_max = 1400.0;  // Maximum mobility (cm^2/Vs)
    double N_ref = 1.3e17;  // Reference doping concentration (cm^-3)
    double alpha_doping = 0.91;  // Doping exponent
    double v_sat = 1.0e7;  // Saturation velocity (cm/s)
    double beta = 2.0;  // Field exponent
    double T = 300.0;  // Temperature (K)
    double alpha_temp = 2.5;  // Temperature exponent
    
    // Test combined temperature and doping dependence
    SECTION("Temperature and doping dependence") {
        double N = 1.0e17;  // Doping concentration (cm^-3)
        
        // Calculate mobility using combined model
        double mu = MobilityModels::combined_temp_doping_mobility(mu_min, mu_max, N, N_ref, alpha_doping, T, alpha_temp);
        
        // Calculate expected mobility
        double mu_max_T = mu_max * std::pow(300.0 / T, alpha_temp);
        double mu_min_T = mu_min * std::pow(300.0 / T, alpha_temp);
        double expected = mu_min_T + (mu_max_T - mu_min_T) / (1.0 + std::pow(N / N_ref, alpha_doping));
        
        REQUIRE(mu == Approx(expected).epsilon(0.01));
    }
    
    // Test combined temperature, doping, and field dependence
    SECTION("Temperature, doping, and field dependence") {
        double N = 1.0e17;  // Doping concentration (cm^-3)
        double E = 1.0e4;  // Electric field (V/cm)
        
        // Calculate mobility using combined model
        double mu = MobilityModels::combined_mobility(mu_min, mu_max, N, N_ref, alpha_doping, T, alpha_temp, E, v_sat, beta);
        
        // Calculate expected mobility
        double mu_max_T = mu_max * std::pow(300.0 / T, alpha_temp);
        double mu_min_T = mu_min * std::pow(300.0 / T, alpha_temp);
        double mu_doping = mu_min_T + (mu_max_T - mu_min_T) / (1.0 + std::pow(N / N_ref, alpha_doping));
        double expected = mu_doping / std::pow(1.0 + std::pow(mu_doping * E / v_sat, beta), 1.0 / beta);
        
        REQUIRE(mu == Approx(expected).epsilon(0.01));
    }
}

/**
 * @brief Test material-specific mobility models.
 *
 * This test verifies that the material-specific mobility models
 * are computed correctly for various materials.
 */
TEST_CASE("Material-specific mobility models", "[mobility_models]") {
    // Test silicon electron mobility
    SECTION("Silicon electron mobility") {
        double N = 1.0e17;  // Doping concentration (cm^-3)
        double T = 300.0;  // Temperature (K)
        double E = 1.0e4;  // Electric field (V/cm)
        
        double mu = MobilityModels::silicon_electron_mobility(N, T, E);
        
        // Silicon electron mobility should be positive and within reasonable range
        REQUIRE(mu > 0.0);
        REQUIRE(mu < 1500.0);
    }
    
    // Test silicon hole mobility
    SECTION("Silicon hole mobility") {
        double N = 1.0e17;  // Doping concentration (cm^-3)
        double T = 300.0;  // Temperature (K)
        double E = 1.0e4;  // Electric field (V/cm)
        
        double mu = MobilityModels::silicon_hole_mobility(N, T, E);
        
        // Silicon hole mobility should be positive and within reasonable range
        REQUIRE(mu > 0.0);
        REQUIRE(mu < 500.0);
    }
    
    // Test GaAs electron mobility
    SECTION("GaAs electron mobility") {
        double N = 1.0e17;  // Doping concentration (cm^-3)
        double T = 300.0;  // Temperature (K)
        double E = 1.0e4;  // Electric field (V/cm)
        
        double mu = MobilityModels::gaas_electron_mobility(N, T, E);
        
        // GaAs electron mobility should be positive and within reasonable range
        REQUIRE(mu > 0.0);
        REQUIRE(mu < 9000.0);
    }
    
    // Test GaAs hole mobility
    SECTION("GaAs hole mobility") {
        double N = 1.0e17;  // Doping concentration (cm^-3)
        double T = 300.0;  // Temperature (K)
        double E = 1.0e4;  // Electric field (V/cm)
        
        double mu = MobilityModels::gaas_hole_mobility(N, T, E);
        
        // GaAs hole mobility should be positive and within reasonable range
        REQUIRE(mu > 0.0);
        REQUIRE(mu < 500.0);
    }
}
