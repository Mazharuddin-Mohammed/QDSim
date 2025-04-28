#pragma once
#include "mesh.h"
#include <Eigen/Dense>

class Normalizer {
public:
    Normalizer(const Mesh& mesh, double m_star, double hbar);
    Eigen::VectorXd deltaNormalize(const Eigen::VectorXd& psi, double E) const;
private:
    const Mesh& mesh;
    double m_star, hbar;
    double computeAsymptoticAmplitude(const Eigen::VectorXd& psi) const;
};