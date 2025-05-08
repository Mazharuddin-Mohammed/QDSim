#pragma once
/**
 * @file normalization.h
 * @brief Defines the Normalizer class for wavefunction normalization.
 *
 * This file contains the declaration of the Normalizer class, which implements
 * methods for normalizing wavefunctions according to various normalization schemes,
 * including delta-function normalization for scattering states.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "mesh.h"
#include <Eigen/Dense>

/**
 * @brief Class for normalizing wavefunctions.
 *
 * This class provides methods for normalizing wavefunctions according to
 * various normalization schemes, including delta-function normalization
 * for scattering states.
 */
class Normalizer {
public:
    Normalizer(const Mesh& mesh, double m_star, double hbar);
    Eigen::VectorXd deltaNormalize(const Eigen::VectorXd& psi, double E) const;
private:
    const Mesh& mesh;
    double m_star, hbar;
    double computeAsymptoticAmplitude(const Eigen::VectorXd& psi) const;
};