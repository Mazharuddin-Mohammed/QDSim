/**
 * @file test_carrier_statistics.cpp
 * @brief Tests for the CarrierStatistics module.
 *
 * This file contains unit tests for the CarrierStatistics module,
 * verifying its functionality, accuracy, and physical correctness.
 *
 * @author Dr. Mazharuddin Mohammed
 * @date 2023-07-15
 */

#include <catch2/catch.hpp>
#include "carrier_statistics.h"
#include "physical_constants.h"
#include <cmath>

/**
 * @brief Test Fermi-Dirac integral of order 1/2.
 *
 * This test verifies that the Fermi-Dirac integral of order 1/2
 * is computed correctly for various values of the reduced Fermi level.
 */
TEST_CASE("Fermi-Dirac integral of order 1/2", "[carrier_statistics]") {
    // Test for large positive eta (asymptotic expansion)
    SECTION("Large positive eta") {
        double eta = 15.0;
        double result = CarrierStatistics::fermi_dirac_half(eta);
        double expected = (2.0/3.0) * std::pow(eta, 1.5) * (1.0 + (1.0/8.0) * std::pow(M_PI/eta, 2));
        REQUIRE(result == Approx(expected).epsilon(0.01));
    }
    
    // Test for large negative eta (exponential approximation)
    SECTION("Large negative eta") {
        double eta = -15.0;
        double result = CarrierStatistics::fermi_dirac_half(eta);
        double expected = std::exp(eta);
        REQUIRE(result == Approx(expected).epsilon(0.01));
    }
    
    // Test for intermediate values (Bednarczyk and Bednarczyk approximation)
    SECTION("Intermediate values") {
        // Test against known values
        std::vector<double> eta_values = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0};
        std::vector<double> expected_values = {0.00673, 0.13534, 0.38278, 0.76569, 1.33780, 2.15654, 7.03192};
        
        for (size_t i = 0; i < eta_values.size(); ++i) {
            double result = CarrierStatistics::fermi_dirac_half(eta_values[i]);
            REQUIRE(result == Approx(expected_values[i]).epsilon(0.01));
        }
    }
}

/**
 * @brief Test inverse of Fermi-Dirac integral of order 1/2.
 *
 * This test verifies that the inverse of the Fermi-Dirac integral of order 1/2
 * is computed correctly for various values.
 */
TEST_CASE("Inverse of Fermi-Dirac integral of order 1/2", "[carrier_statistics]") {
    // Test for small x (logarithmic approximation)
    SECTION("Small x") {
        double x = 0.001;
        double result = CarrierStatistics::inverse_fermi_dirac_half(x);
        double expected = std::log(x);
        REQUIRE(result == Approx(expected).epsilon(0.01));
    }
    
    // Test for large x (asymptotic expansion)
    SECTION("Large x") {
        double x = 15.0;
        double result = CarrierStatistics::inverse_fermi_dirac_half(x);
        double expected = std::pow(1.5 * x, 2.0/3.0) * (1.0 - (1.0/12.0) * std::pow(M_PI/std::pow(1.5*x, 2.0/3.0), 2));
        REQUIRE(result == Approx(expected).epsilon(0.01));
    }
    
    // Test for intermediate values (Joyce-Dixon approximation)
    SECTION("Intermediate values") {
        // Test against known values
        std::vector<double> x_values = {0.1, 0.5, 1.0, 2.0, 5.0};
        std::vector<double> expected_values = {-2.25, -0.56, 0.0, 0.83, 2.28};
        
        for (size_t i = 0; i < x_values.size(); ++i) {
            double result = CarrierStatistics::inverse_fermi_dirac_half(x_values[i]);
            REQUIRE(result == Approx(expected_values[i]).epsilon(0.1));
        }
    }
    
    // Test inverse relationship
    SECTION("Inverse relationship") {
        std::vector<double> eta_values = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0};
        
        for (double eta : eta_values) {
            double x = CarrierStatistics::fermi_dirac_half(eta);
            double eta_inverse = CarrierStatistics::inverse_fermi_dirac_half(x);
            REQUIRE(eta_inverse == Approx(eta).epsilon(0.1));
        }
    }
}

/**
 * @brief Test electron concentration calculations.
 *
 * This test verifies that the electron concentration is computed correctly
 * using both Fermi-Dirac and Boltzmann statistics.
 */
TEST_CASE("Electron concentration calculations", "[carrier_statistics]") {
    // Test parameters
    double E_c = 0.0;  // Conduction band edge (eV)
    double N_c = 2.8e19;  // Effective density of states (cm^-3)
    double temperature = 300.0;  // Temperature (K)
    
    // Test Fermi-Dirac statistics
    SECTION("Fermi-Dirac statistics") {
        // Test for different Fermi levels
        std::vector<double> E_f_values = {-0.2, -0.1, 0.0, 0.1, 0.2};
        
        for (double E_f : E_f_values) {
            // Calculate electron concentration using Fermi-Dirac statistics
            double n_fd = CarrierStatistics::electron_concentration_fd(E_c, E_f, N_c, temperature);
            
            // Calculate reduced Fermi level
            double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature * PhysicalConstants::J_TO_EV;
            double eta = (E_f - E_c) / kT;
            
            // Calculate expected electron concentration
            double expected = N_c * CarrierStatistics::fermi_dirac_half(eta);
            
            REQUIRE(n_fd == Approx(expected).epsilon(0.01));
        }
    }
    
    // Test Boltzmann statistics
    SECTION("Boltzmann statistics") {
        // Test for different Fermi levels
        std::vector<double> E_f_values = {-0.2, -0.1, 0.0, 0.1, 0.2};
        
        for (double E_f : E_f_values) {
            // Calculate electron concentration using Boltzmann statistics
            double n_boltzmann = CarrierStatistics::electron_concentration_boltzmann(E_c, E_f, N_c, temperature);
            
            // Calculate expected electron concentration
            double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature * PhysicalConstants::J_TO_EV;
            double expected = N_c * std::exp((E_f - E_c) / kT);
            
            REQUIRE(n_boltzmann == Approx(expected).epsilon(0.01));
        }
    }
    
    // Compare Fermi-Dirac and Boltzmann statistics
    SECTION("Compare Fermi-Dirac and Boltzmann statistics") {
        // For non-degenerate semiconductors (E_f << E_c), Fermi-Dirac and Boltzmann statistics should give similar results
        double E_f = E_c - 0.2;  // Fermi level well below conduction band
        
        double n_fd = CarrierStatistics::electron_concentration_fd(E_c, E_f, N_c, temperature);
        double n_boltzmann = CarrierStatistics::electron_concentration_boltzmann(E_c, E_f, N_c, temperature);
        
        REQUIRE(n_fd == Approx(n_boltzmann).epsilon(0.1));
        
        // For degenerate semiconductors (E_f > E_c), Fermi-Dirac statistics should give lower concentration than Boltzmann
        E_f = E_c + 0.2;  // Fermi level above conduction band
        
        n_fd = CarrierStatistics::electron_concentration_fd(E_c, E_f, N_c, temperature);
        n_boltzmann = CarrierStatistics::electron_concentration_boltzmann(E_c, E_f, N_c, temperature);
        
        REQUIRE(n_fd < n_boltzmann);
    }
}

/**
 * @brief Test hole concentration calculations.
 *
 * This test verifies that the hole concentration is computed correctly
 * using both Fermi-Dirac and Boltzmann statistics.
 */
TEST_CASE("Hole concentration calculations", "[carrier_statistics]") {
    // Test parameters
    double E_v = 0.0;  // Valence band edge (eV)
    double N_v = 1.04e19;  // Effective density of states (cm^-3)
    double temperature = 300.0;  // Temperature (K)
    
    // Test Fermi-Dirac statistics
    SECTION("Fermi-Dirac statistics") {
        // Test for different Fermi levels
        std::vector<double> E_f_values = {-0.2, -0.1, 0.0, 0.1, 0.2};
        
        for (double E_f : E_f_values) {
            // Calculate hole concentration using Fermi-Dirac statistics
            double p_fd = CarrierStatistics::hole_concentration_fd(E_v, E_f, N_v, temperature);
            
            // Calculate reduced Fermi level
            double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature * PhysicalConstants::J_TO_EV;
            double eta = (E_v - E_f) / kT;
            
            // Calculate expected hole concentration
            double expected = N_v * CarrierStatistics::fermi_dirac_half(eta);
            
            REQUIRE(p_fd == Approx(expected).epsilon(0.01));
        }
    }
    
    // Test Boltzmann statistics
    SECTION("Boltzmann statistics") {
        // Test for different Fermi levels
        std::vector<double> E_f_values = {-0.2, -0.1, 0.0, 0.1, 0.2};
        
        for (double E_f : E_f_values) {
            // Calculate hole concentration using Boltzmann statistics
            double p_boltzmann = CarrierStatistics::hole_concentration_boltzmann(E_v, E_f, N_v, temperature);
            
            // Calculate expected hole concentration
            double kT = PhysicalConstants::BOLTZMANN_CONSTANT * temperature * PhysicalConstants::J_TO_EV;
            double expected = N_v * std::exp((E_v - E_f) / kT);
            
            REQUIRE(p_boltzmann == Approx(expected).epsilon(0.01));
        }
    }
}

/**
 * @brief Test recombination rate calculations.
 *
 * This test verifies that the recombination rates are computed correctly
 * for SRH, Auger, and radiative recombination mechanisms.
 */
TEST_CASE("Recombination rate calculations", "[carrier_statistics]") {
    // Test parameters
    double n = 1.0e16;  // Electron concentration (cm^-3)
    double p = 1.0e16;  // Hole concentration (cm^-3)
    double n_i = 1.0e10;  // Intrinsic carrier concentration (cm^-3)
    
    // Test SRH recombination
    SECTION("SRH recombination") {
        double tau_n = 1.0e-6;  // Electron lifetime (s)
        double tau_p = 1.0e-6;  // Hole lifetime (s)
        double n1 = 1.0e10;  // Electron SRH concentration (cm^-3)
        double p1 = 1.0e10;  // Hole SRH concentration (cm^-3)
        
        double R_SRH = CarrierStatistics::srh_recombination(n, p, n_i, tau_n, tau_p, n1, p1);
        
        // Calculate expected SRH recombination rate
        double expected = (n * p - n_i * n_i) / (tau_p * (n + n1) + tau_n * (p + p1));
        
        REQUIRE(R_SRH == Approx(expected).epsilon(0.01));
    }
    
    // Test Auger recombination
    SECTION("Auger recombination") {
        double C_n = 2.8e-31;  // Electron Auger coefficient (cm^6/s)
        double C_p = 9.9e-32;  // Hole Auger coefficient (cm^6/s)
        
        double R_Auger = CarrierStatistics::auger_recombination(n, p, n_i, C_n, C_p);
        
        // Calculate expected Auger recombination rate
        double expected = (C_n * n + C_p * p) * (n * p - n_i * n_i);
        
        REQUIRE(R_Auger == Approx(expected).epsilon(0.01));
    }
    
    // Test radiative recombination
    SECTION("Radiative recombination") {
        double B = 1.1e-14;  // Radiative recombination coefficient (cm^3/s)
        
        double R_rad = CarrierStatistics::radiative_recombination(n, p, n_i, B);
        
        // Calculate expected radiative recombination rate
        double expected = B * (n * p - n_i * n_i);
        
        REQUIRE(R_rad == Approx(expected).epsilon(0.01));
    }
    
    // Test total recombination
    SECTION("Total recombination") {
        double tau_n = 1.0e-6;  // Electron lifetime (s)
        double tau_p = 1.0e-6;  // Hole lifetime (s)
        double n1 = 1.0e10;  // Electron SRH concentration (cm^-3)
        double p1 = 1.0e10;  // Hole SRH concentration (cm^-3)
        double C_n = 2.8e-31;  // Electron Auger coefficient (cm^6/s)
        double C_p = 9.9e-32;  // Hole Auger coefficient (cm^6/s)
        double B = 1.1e-14;  // Radiative recombination coefficient (cm^3/s)
        
        double R_total = CarrierStatistics::total_recombination(n, p, n_i, tau_n, tau_p, n1, p1, C_n, C_p, B);
        
        // Calculate expected recombination rates
        double R_SRH = CarrierStatistics::srh_recombination(n, p, n_i, tau_n, tau_p, n1, p1);
        double R_Auger = CarrierStatistics::auger_recombination(n, p, n_i, C_n, C_p);
        double R_rad = CarrierStatistics::radiative_recombination(n, p, n_i, B);
        
        // Calculate expected total recombination rate
        double expected = R_SRH + R_Auger + R_rad;
        
        REQUIRE(R_total == Approx(expected).epsilon(0.01));
    }
}
