#include <catch2/catch.hpp>
#include "adaptive_mesh.h"
#include "mesh.h"

TEST_CASE("Adaptive mesh refinement linear", "[adaptive_mesh]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 1);
    int initial_elements = mesh.getNumElements();
    Eigen::VectorXd psi = Eigen::VectorXd::Ones(mesh.getNumNodes());
    std::vector<bool> refine_flags = AdaptiveMesh::computeRefinementFlags(mesh, psi, 0.0);
    mesh.refine(refine_flags);
    REQUIRE(mesh.getNumElements() > initial_elements);
}

TEST_CASE("Adaptive mesh refinement quadratic", "[adaptive_mesh]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 2);
    int initial_elements = mesh.getNumElements();
    Eigen::VectorXd psi = Eigen::VectorXd::Ones(mesh.getNumNodes());
    std::vector<bool> refine_flags = AdaptiveMesh::computeRefinementFlags(mesh, psi, 0.0);
    mesh.refine(refine_flags);
    REQUIRE(mesh.getNumElements() > initial_elements);
    REQUIRE(mesh.getQuadraticElements().size() == mesh.getNumElements());
}

TEST_CASE("Adaptive mesh refinement cubic", "[adaptive_mesh]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 3);
    int initial_elements = mesh.getNumElements();
    Eigen::VectorXd psi = Eigen::VectorXd::Ones(mesh.getNumNodes());
    std::vector<bool> refine_flags = AdaptiveMesh::computeRefinementFlags(mesh, psi, 0.0);
    mesh.refine(refine_flags);
    REQUIRE(mesh.getNumElements() > initial_elements);
    REQUIRE(mesh.getCubicElements().size() == mesh.getNumElements());
}

TEST_CASE("Mesh conformity", "[adaptive_mesh]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 3);
    Eigen::VectorXd psi = Eigen::VectorXd::Ones(mesh.getNumNodes());
    std::vector<bool> refine_flags = AdaptiveMesh::computeRefinementFlags(mesh, psi, 0.0);
    mesh.refine(refine_flags);
    REQUIRE(AdaptiveMesh::isMeshConforming(mesh));
}

TEST_CASE("Triangle quality", "[adaptive_mesh]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 3);
    for (int i = 0; i < mesh.getNumElements(); ++i) {
        double quality = AdaptiveMesh::computeTriangleQuality(mesh, i);
        REQUIRE(quality > 0.1);
    }
}

TEST_CASE("Mesh smoothing", "[adaptive_mesh]") {
    Mesh mesh(1e-7, 1e-7, 5, 5, 3);
    Eigen::VectorXd psi = Eigen::VectorXd::Ones(mesh.getNumNodes());
    std::vector<bool> refine_flags = AdaptiveMesh::computeRefinementFlags(mesh, psi, 0.0);
    mesh.refine(refine_flags);
    double initial_quality = AdaptiveMesh::computeTriangleQuality(mesh, 0);
    AdaptiveMesh::smoothMesh(mesh);
    double final_quality = AdaptiveMesh::computeTriangleQuality(mesh, 0);
    REQUIRE(final_quality >= initial_quality);
}