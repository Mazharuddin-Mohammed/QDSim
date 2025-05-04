#pragma once
/**
 * @file utils.h
 * @brief Utility functions for QDSim.
 *
 * This file contains utility functions for QDSim, including functions for
 * saving matrices to files, converting between different units, and other
 * helper functions.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <Eigen/Sparse>

/**
 * @brief Save a sparse matrix to a file.
 *
 * This function saves a sparse matrix to a file in a format that can be
 * read by external tools like MATLAB or Python.
 *
 * @param mat The sparse matrix to save
 * @param filename The name of the file to save to
 */
void saveMatrix(const Eigen::SparseMatrix<double>& mat, const std::string& filename);