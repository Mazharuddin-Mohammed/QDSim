#pragma once
#include "materials.h"
#include <Eigen/Dense>
#include <string>

namespace Physics {

double effective_mass(double x, double y, const Materials::Material& qd_mat, 
                      const Materials::Material& matrix_mat, double R);
double potential(double x, double y, const Materials::Material& qd_mat, 
                 const Materials::Material& matrix_mat, double R, const std::string& type, 
                 const Eigen::VectorXd& phi);
double epsilon_r(double x, double y, const Materials::Material& p_mat, 
                 const Materials::Material& n_mat);
double charge_density(double x, double y, double N_A, double N_D, double depletion_width);
double cap(double x, double y, double eta, double Lx, double Ly, double d);

} // namespace Physics