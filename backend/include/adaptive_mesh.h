#pragma once
#include "mesh.h"
#include <Eigen/Dense>
#include <vector>
#include <mpi.h>

class AdaptiveMesh {
public:
    static void refineMesh(Mesh& mesh, const std::vector<bool>& refine_flags, MPI_Comm comm = MPI_COMM_WORLD);
    static std::vector<bool> computeRefinementFlags(const Mesh& mesh, const Eigen::VectorXd& psi, double threshold);
    static void smoothMesh(Mesh& mesh);
    static double computeTriangleQuality(const Mesh& mesh, int elem_idx);
    static bool isMeshConforming(const Mesh& mesh);
};