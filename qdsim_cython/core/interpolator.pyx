# distutils: language = c++
# cython: language_level = 3

"""
Cython wrapper for FEInterpolator class

High-performance finite element interpolation for quantum wavefunctions
and physical fields in semiconductor devices.

This module provides optimized interpolation capabilities for:
- Quantum wavefunctions and probability densities
- Electric and magnetic fields
- Material properties (effective mass, dielectric constant)
- Current densities and charge distributions
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as bint
from ..eigen cimport VectorXd, VectorXcd
from .mesh cimport Mesh

# Import C++ declarations
from .interpolator cimport (
    FEInterpolator as CppFEInterpolator,
    WavefunctionInterpolator as CppWavefunctionInterpolator,
    PotentialInterpolator as CppPotentialInterpolator,
    MaterialInterpolator as CppMaterialInterpolator,
    InterpolationMethod, InterpolationConfig
)

# Initialize NumPy
cnp.import_array()

cdef class FEInterpolator:
    """
    High-performance finite element interpolator.
    
    Provides fast interpolation of scalar and vector fields defined on
    finite element meshes, with support for complex-valued functions
    and gradient computation.
    """
    
    cdef CppFEInterpolator* _interpolator
    cdef bint _owns_interpolator
    cdef object _mesh_ref  # Keep reference to mesh
    
    def __cinit__(self, Mesh mesh):
        """
        Initialize the FE interpolator.
        
        Parameters:
        -----------
        mesh : Mesh
            Finite element mesh for interpolation
        """
        self._interpolator = NULL
        self._owns_interpolator = False
        self._mesh_ref = mesh
        
        try:
            self._interpolator = new CppFEInterpolator(mesh._mesh[0])
            self._owns_interpolator = True
        except Exception as e:
            raise RuntimeError(f"Failed to create FE interpolator: {e}")
    
    def __dealloc__(self):
        """Clean up C++ resources"""
        if self._owns_interpolator and self._interpolator != NULL:
            del self._interpolator
            self._interpolator = NULL
    
    def interpolate(self, double x, double y, values):
        """
        Interpolate scalar field at point (x, y).
        
        Parameters:
        -----------
        x, y : float
            Coordinates of interpolation point
        values : array_like
            Field values at mesh nodes
        
        Returns:
        --------
        float
            Interpolated value
        """
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[double, ndim=1] np_values = np.asarray(values, dtype=np.float64)
        cdef int n = np_values.shape[0]
        
        # Convert to Eigen vector
        cdef VectorXd eigen_values = VectorXd(n)
        for i in range(n):
            eigen_values[i] = np_values[i]
        
        try:
            return self._interpolator.interpolate(x, y, eigen_values)
        except Exception as e:
            raise RuntimeError(f"Interpolation failed: {e}")
    
    def interpolate_complex(self, double x, double y, complex_values, str component='real'):
        """
        Interpolate complex field at point (x, y).
        
        Parameters:
        -----------
        x, y : float
            Coordinates of interpolation point
        complex_values : array_like
            Complex field values at mesh nodes
        component : str
            Component to interpolate: 'real', 'imag', 'abs'
        
        Returns:
        --------
        float
            Interpolated component value
        """
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[complex, ndim=1] np_values = np.asarray(complex_values, dtype=np.complex128)
        cdef int n = np_values.shape[0]
        
        # Convert to Eigen complex vector
        cdef VectorXcd eigen_values = VectorXcd(n)
        for i in range(n):
            eigen_values[i] = np_values[i].real + 1j * np_values[i].imag
        
        try:
            if component == 'real':
                return self._interpolator.interpolate_complex_real(x, y, eigen_values)
            elif component == 'imag':
                return self._interpolator.interpolate_complex_imag(x, y, eigen_values)
            elif component == 'abs':
                return self._interpolator.interpolate_complex_abs(x, y, eigen_values)
            else:
                raise ValueError(f"Unknown component: {component}")
        except Exception as e:
            raise RuntimeError(f"Complex interpolation failed: {e}")
    
    def interpolate_vector(self, x_points, y_points, values):
        """
        Interpolate at multiple points simultaneously.
        
        Parameters:
        -----------
        x_points, y_points : array_like
            Arrays of interpolation coordinates
        values : array_like
            Field values at mesh nodes
        
        Returns:
        --------
        numpy.ndarray
            Array of interpolated values
        """
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[double, ndim=1] np_x = np.asarray(x_points, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] np_y = np.asarray(y_points, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1] np_values = np.asarray(values, dtype=np.float64)
        
        if np_x.shape[0] != np_y.shape[0]:
            raise ValueError("x_points and y_points must have same length")
        
        cdef int num_points = np_x.shape[0]
        cdef int num_values = np_values.shape[0]
        
        # Convert to C++ vectors
        cdef vector[double] cpp_x_points
        cdef vector[double] cpp_y_points
        cdef VectorXd eigen_values = VectorXd(num_values)
        
        for i in range(num_points):
            cpp_x_points.push_back(np_x[i])
            cpp_y_points.push_back(np_y[i])
        
        for i in range(num_values):
            eigen_values[i] = np_values[i]
        
        try:
            cdef vector[double] result = self._interpolator.interpolate_vector(
                cpp_x_points, cpp_y_points, eigen_values)
            
            # Convert back to NumPy
            cdef cnp.ndarray[double, ndim=1] np_result = np.empty(num_points, dtype=np.float64)
            for i in range(num_points):
                np_result[i] = result[i]
            
            return np_result
        except Exception as e:
            raise RuntimeError(f"Vector interpolation failed: {e}")
    
    def interpolate_gradient(self, double x, double y, values):
        """
        Interpolate gradient at point (x, y).
        
        Parameters:
        -----------
        x, y : float
            Coordinates of interpolation point
        values : array_like
            Field values at mesh nodes
        
        Returns:
        --------
        tuple
            (grad_x, grad_y) gradient components
        """
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[double, ndim=1] np_values = np.asarray(values, dtype=np.float64)
        cdef int n = np_values.shape[0]
        
        # Convert to Eigen vector
        cdef VectorXd eigen_values = VectorXd(n)
        for i in range(n):
            eigen_values[i] = np_values[i]
        
        try:
            cdef vector[double] gradient = self._interpolator.interpolate_gradient(x, y, eigen_values)
            return (gradient[0], gradient[1])
        except Exception as e:
            raise RuntimeError(f"Gradient interpolation failed: {e}")
    
    def is_inside_domain(self, double x, double y):
        """Check if point is inside the computational domain"""
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        return self._interpolator.is_inside_domain(x, y)
    
    def find_containing_element(self, double x, double y):
        """Find the element containing the given point"""
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        return self._interpolator.find_containing_element(x, y)
    
    def estimate_interpolation_error(self, double x, double y, values):
        """Estimate interpolation error at given point"""
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[double, ndim=1] np_values = np.asarray(values, dtype=np.float64)
        cdef int n = np_values.shape[0]
        
        cdef VectorXd eigen_values = VectorXd(n)
        for i in range(n):
            eigen_values[i] = np_values[i]
        
        return self._interpolator.estimate_interpolation_error(x, y, eigen_values)
    
    def enable_caching(self, bint enable):
        """Enable or disable interpolation caching for performance"""
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        self._interpolator.enable_caching(enable)
    
    def clear_cache(self):
        """Clear interpolation cache"""
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        self._interpolator.clear_cache()
    
    def get_interpolation_statistics(self):
        """Get interpolation performance statistics"""
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        cdef string stats = self._interpolator.get_interpolation_statistics()
        return stats.decode('utf-8')

cdef class WavefunctionInterpolator:
    """
    Specialized interpolator for quantum wavefunctions.
    
    Provides optimized interpolation for quantum mechanical quantities:
    - Probability density |ψ|²
    - Current density j = (ℏ/2mi)[ψ*∇ψ - ψ∇ψ*]
    - Phase and amplitude
    """
    
    cdef CppWavefunctionInterpolator* _interpolator
    cdef bint _owns_interpolator
    cdef object _mesh_ref
    
    def __cinit__(self, Mesh mesh):
        """Initialize wavefunction interpolator"""
        self._interpolator = NULL
        self._owns_interpolator = False
        self._mesh_ref = mesh
        
        try:
            self._interpolator = new CppWavefunctionInterpolator(mesh._mesh[0])
            self._owns_interpolator = True
        except Exception as e:
            raise RuntimeError(f"Failed to create wavefunction interpolator: {e}")
    
    def __dealloc__(self):
        """Clean up C++ resources"""
        if self._owns_interpolator and self._interpolator != NULL:
            del self._interpolator
            self._interpolator = NULL
    
    def interpolate_probability_density(self, double x, double y, psi):
        """
        Interpolate probability density |ψ|² at point (x, y).
        
        Parameters:
        -----------
        x, y : float
            Coordinates of interpolation point
        psi : array_like
            Complex wavefunction values at mesh nodes
        
        Returns:
        --------
        float
            Probability density |ψ(x,y)|²
        """
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[complex, ndim=1] np_psi = np.asarray(psi, dtype=np.complex128)
        cdef int n = np_psi.shape[0]
        
        cdef VectorXcd eigen_psi = VectorXcd(n)
        for i in range(n):
            eigen_psi[i] = np_psi[i].real + 1j * np_psi[i].imag
        
        try:
            return self._interpolator.interpolate_probability_density(x, y, eigen_psi)
        except Exception as e:
            raise RuntimeError(f"Probability density interpolation failed: {e}")
    
    def interpolate_current_density(self, double x, double y, psi):
        """
        Interpolate current density at point (x, y).
        
        Parameters:
        -----------
        x, y : float
            Coordinates of interpolation point
        psi : array_like
            Complex wavefunction values at mesh nodes
        
        Returns:
        --------
        tuple
            (j_x, j_y) current density components
        """
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[complex, ndim=1] np_psi = np.asarray(psi, dtype=np.complex128)
        cdef int n = np_psi.shape[0]
        
        cdef VectorXcd eigen_psi = VectorXcd(n)
        for i in range(n):
            eigen_psi[i] = np_psi[i].real + 1j * np_psi[i].imag
        
        try:
            cdef vector[double] current = self._interpolator.interpolate_current_density(x, y, eigen_psi)
            return (current[0], current[1])
        except Exception as e:
            raise RuntimeError(f"Current density interpolation failed: {e}")

cdef class PotentialInterpolator:
    """
    Specialized interpolator for electrostatic potentials and fields.
    
    Provides optimized interpolation for:
    - Electric field E = -∇V
    - Charge density ρ = -ε∇²V
    - Energy landscapes
    """
    
    cdef CppPotentialInterpolator* _interpolator
    cdef bint _owns_interpolator
    cdef object _mesh_ref
    
    def __cinit__(self, Mesh mesh):
        """Initialize potential interpolator"""
        self._interpolator = NULL
        self._owns_interpolator = False
        self._mesh_ref = mesh
        
        try:
            self._interpolator = new CppPotentialInterpolator(mesh._mesh[0])
            self._owns_interpolator = True
        except Exception as e:
            raise RuntimeError(f"Failed to create potential interpolator: {e}")
    
    def __dealloc__(self):
        """Clean up C++ resources"""
        if self._owns_interpolator and self._interpolator != NULL:
            del self._interpolator
            self._interpolator = NULL
    
    def interpolate_electric_field(self, double x, double y, potential):
        """
        Interpolate electric field E = -∇V at point (x, y).
        
        Parameters:
        -----------
        x, y : float
            Coordinates of interpolation point
        potential : array_like
            Potential values at mesh nodes
        
        Returns:
        --------
        tuple
            (E_x, E_y) electric field components
        """
        if self._interpolator == NULL:
            raise RuntimeError("Interpolator not initialized")
        
        cdef cnp.ndarray[double, ndim=1] np_potential = np.asarray(potential, dtype=np.float64)
        cdef int n = np_potential.shape[0]
        
        cdef VectorXd eigen_potential = VectorXd(n)
        for i in range(n):
            eigen_potential[i] = np_potential[i]
        
        try:
            cdef vector[double] field = self._interpolator.interpolate_electric_field(x, y, eigen_potential)
            return (field[0], field[1])
        except Exception as e:
            raise RuntimeError(f"Electric field interpolation failed: {e}")

# Utility functions
def clear_all_interpolator_caches():
    """Clear all interpolator caches to free memory"""
    # This would call a global cache clearing function
    pass
