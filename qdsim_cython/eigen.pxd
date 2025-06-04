# distutils: language = c++
# cython: language_level = 3

"""
Cython declarations for Eigen library types

This file provides Cython declarations for commonly used Eigen types
in the QDSim project.
"""

from libcpp cimport bool as bint
from libcpp.complex cimport complex

cdef extern from "Eigen/Dense" namespace "Eigen":
    # Vector types
    cdef cppclass Vector2d:
        Vector2d() except +
        Vector2d(double x, double y) except +
        double& operator[](int index)
        const double& operator[](int index) const
        double x() const
        double y() const
        double norm() const
        Vector2d normalized() const
        
    cdef cppclass Vector3d:
        Vector3d() except +
        Vector3d(double x, double y, double z) except +
        double& operator[](int index)
        const double& operator[](int index) const
        double x() const
        double y() const
        double z() const
        double norm() const
        Vector3d normalized() const
        
    cdef cppclass VectorXd:
        VectorXd() except +
        VectorXd(int size) except +
        double& operator[](int index)
        const double& operator[](int index) const
        int size() const
        void resize(int size)
        double norm() const
        VectorXd normalized() const

    cdef cppclass VectorXcd:
        VectorXcd() except +
        VectorXcd(int size) except +
        complex[double]& operator[](int index)
        const complex[double]& operator[](int index) const
        int size() const
        void resize(int size)
        double norm() const
        VectorXcd normalized() const

    # Matrix types
    cdef cppclass Matrix2d:
        Matrix2d() except +
        double& operator()(int row, int col)
        const double& operator()(int row, int col) const
        
    cdef cppclass Matrix3d:
        Matrix3d() except +
        double& operator()(int row, int col)
        const double& operator()(int row, int col) const
        
    cdef cppclass MatrixXd:
        MatrixXd() except +
        MatrixXd(int rows, int cols) except +
        double& operator()(int row, int col)
        const double& operator()(int row, int col) const
        int rows() const
        int cols() const
        void resize(int rows, int cols)

    cdef cppclass MatrixXcd:
        MatrixXcd() except +
        MatrixXcd(int rows, int cols) except +
        complex[double]& operator()(int row, int col)
        const complex[double]& operator()(int row, int col) const
        int rows() const
        int cols() const
        void resize(int rows, int cols)

cdef extern from "Eigen/Sparse" namespace "Eigen":
    # Sparse matrix types
    cdef cppclass SparseMatrix[T]:
        SparseMatrix() except +
        SparseMatrix(int rows, int cols) except +
        T& coeffRef(int row, int col)
        T coeff(int row, int col) const
        int rows() const
        int cols() const
        int nonZeros() const
        void resize(int rows, int cols)
        void reserve(int nnz)
        void makeCompressed()
        
    # Commonly used sparse matrix types
    ctypedef SparseMatrix[double] SparseMatrixXd
    ctypedef SparseMatrix[float] SparseMatrixXf
    ctypedef SparseMatrix[complex[double]] SparseMatrixXcd
