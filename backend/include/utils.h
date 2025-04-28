#pragma once
#include <Eigen/Sparse>

void saveMatrix(const Eigen::SparseMatrix<double>& mat, const std::string& filename);