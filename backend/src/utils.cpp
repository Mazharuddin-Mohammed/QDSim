/**
 * @file utils.cpp
 * @brief Implementation of utility functions for QDSim.
 *
 * This file contains the implementation of utility functions for QDSim,
 * including functions for saving matrices to files, converting between
 * different units, and other helper functions.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "utils.h"
#include <fstream>

void saveMatrix(const Eigen::SparseMatrix<double>& mat, const std::string& filename) {
    std::ofstream out(filename);
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            out << it.row() << " " << it.col() << " " << it.value() << "\n";
        }
    }
}