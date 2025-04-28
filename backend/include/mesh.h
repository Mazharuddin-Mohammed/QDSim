#pragma once
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <string>

class Mesh {
public:
    Mesh(double Lx, double Ly, int nx, int ny, int element_order = 1);
    const std::vector<Eigen::Vector2d>& getNodes() const { return nodes; }
    const std::vector<std::array<int, 3>>& getElements() const { return elements; }
    const std::vector<std::array<int, 6>>& getQuadraticElements() const { return quadratic_elements; }
    const std::vector<std::array<int, 10>>& getCubicElements() const { return cubic_elements; } // New: For P3
    int getNumNodes() const { return nodes.size(); }
    int getNumElements() const { return elements.size(); }
    int getElementOrder() const { return element_order; }
    void refine(const std::vector<bool>& refine_flags);
    void save(const std::string& filename) const;
    static Mesh load(const std::string& filename);
private:
    std::vector<Eigen::Vector2d> nodes;
    std::vector<std::array<int, 3>> elements; // P1 elements
    std::vector<std::array<int, 6>> quadratic_elements; // P2 elements
    std::vector<std::array<int, 10>> cubic_elements; // P3 elements
    int element_order; // 1 for P1, 2 for P2, 3 for P3
    void generateTriangularMesh(double Lx, double Ly, int nx, int ny);
};