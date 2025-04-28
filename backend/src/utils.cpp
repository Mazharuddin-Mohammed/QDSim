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