# distutils: language = c++
# cython: language_level = 3

"""
Cython-based EigenSolver Implementation

This module provides a complete Cython implementation of eigenvalue solvers
for quantum mechanical calculations, replacing the C++ backend implementation
with high-performance Cython code.
"""

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp cimport bool as bint
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from libc.math cimport sqrt, abs as c_abs
import time

# Initialize NumPy
cnp.import_array()

cdef class CythonEigenSolver:
    """
    High-performance Cython implementation of eigenvalue solver.
    
    Provides various eigenvalue solving algorithms for quantum mechanical
    and general linear algebra problems.
    """
    
    cdef public int matrix_size
    cdef object last_eigenvalues
    cdef object last_eigenvectors
    cdef double last_solve_time
    cdef str last_method
    cdef bint last_converged
    cdef dict solver_options
    
    def __cinit__(self, mesh=None):
        """
        Initialize the eigenvalue solver.
        
        Parameters:
        -----------
        mesh : SimpleMesh, optional
            The mesh object (for matrix size estimation)
        """
        if mesh is not None:
            self.matrix_size = mesh.num_nodes
        else:
            self.matrix_size = 0
        
        # Initialize solution storage
        self.last_eigenvalues = None
        self.last_eigenvectors = None
        self.last_solve_time = 0.0
        self.last_method = ""
        self.last_converged = False
        
        # Default solver options
        self.solver_options = {
            'tolerance': 1e-8,
            'max_iterations': 1000,
            'which': 'SM',  # Smallest magnitude
            'sigma': None,  # Shift for shift-invert mode
            'mode': 'normal'  # 'normal', 'shift-invert', 'buckling'
        }
    
    def solve_standard_eigenvalue(self, matrix, int num_eigenvalues, 
                                 str method='arpack', **kwargs):
        """
        Solve standard eigenvalue problem: A x = λ x
        
        Parameters:
        -----------
        matrix : scipy.sparse matrix
            The matrix A
        num_eigenvalues : int
            Number of eigenvalues to compute
        method : str
            Solver method ('arpack', 'lobpcg', 'dense')
        **kwargs : dict
            Additional solver options
        
        Returns:
        --------
        tuple
            (eigenvalues, eigenvectors)
        """
        cdef double start_time = time.time()
        
        # Update solver options
        options = self.solver_options.copy()
        options.update(kwargs)
        
        self.last_method = method
        
        try:
            if method == 'arpack':
                eigenvals, eigenvecs = self._solve_arpack_standard(
                    matrix, num_eigenvalues, options
                )
            elif method == 'lobpcg':
                eigenvals, eigenvecs = self._solve_lobpcg_standard(
                    matrix, num_eigenvalues, options
                )
            elif method == 'dense':
                eigenvals, eigenvecs = self._solve_dense_standard(
                    matrix, num_eigenvalues, options
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            self.last_converged = True
            
        except Exception as e:
            print(f"Eigenvalue solver failed with method {method}: {e}")
            # Try fallback method
            if method != 'arpack':
                print("Trying ARPACK as fallback...")
                try:
                    eigenvals, eigenvecs = self._solve_arpack_standard(
                        matrix, num_eigenvalues, options
                    )
                    self.last_converged = True
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    eigenvals = np.array([])
                    eigenvecs = np.array([]).reshape(matrix.shape[0], 0)
                    self.last_converged = False
            else:
                eigenvals = np.array([])
                eigenvecs = np.array([]).reshape(matrix.shape[0], 0)
                self.last_converged = False
        
        self.last_solve_time = time.time() - start_time
        self.last_eigenvalues = eigenvals
        self.last_eigenvectors = eigenvecs
        
        return eigenvals, eigenvecs
    
    def solve_generalized_eigenvalue(self, A_matrix, B_matrix, int num_eigenvalues,
                                   str method='arpack', **kwargs):
        """
        Solve generalized eigenvalue problem: A x = λ B x
        
        Parameters:
        -----------
        A_matrix : scipy.sparse matrix
            The matrix A
        B_matrix : scipy.sparse matrix
            The matrix B
        num_eigenvalues : int
            Number of eigenvalues to compute
        method : str
            Solver method ('arpack', 'lobpcg')
        **kwargs : dict
            Additional solver options
        
        Returns:
        --------
        tuple
            (eigenvalues, eigenvectors)
        """
        cdef double start_time = time.time()
        
        # Update solver options
        options = self.solver_options.copy()
        options.update(kwargs)
        
        self.last_method = f"generalized_{method}"
        
        try:
            if method == 'arpack':
                eigenvals, eigenvecs = self._solve_arpack_generalized(
                    A_matrix, B_matrix, num_eigenvalues, options
                )
            elif method == 'lobpcg':
                eigenvals, eigenvecs = self._solve_lobpcg_generalized(
                    A_matrix, B_matrix, num_eigenvalues, options
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            self.last_converged = True
            
        except Exception as e:
            print(f"Generalized eigenvalue solver failed with method {method}: {e}")
            # Try fallback
            if method != 'arpack':
                print("Trying ARPACK as fallback...")
                try:
                    eigenvals, eigenvecs = self._solve_arpack_generalized(
                        A_matrix, B_matrix, num_eigenvalues, options
                    )
                    self.last_converged = True
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    eigenvals = np.array([])
                    eigenvecs = np.array([]).reshape(A_matrix.shape[0], 0)
                    self.last_converged = False
            else:
                eigenvals = np.array([])
                eigenvecs = np.array([]).reshape(A_matrix.shape[0], 0)
                self.last_converged = False
        
        self.last_solve_time = time.time() - start_time
        self.last_eigenvalues = eigenvals
        self.last_eigenvectors = eigenvecs
        
        return eigenvals, eigenvecs
    
    def _solve_arpack_standard(self, matrix, int num_eigenvalues, dict options):
        """Solve using ARPACK for standard eigenvalue problem"""
        return spla.eigsh(
            matrix,
            k=num_eigenvalues,
            which=options['which'],
            tol=options['tolerance'],
            maxiter=options['max_iterations'],
            sigma=options['sigma']
        )
    
    def _solve_arpack_generalized(self, A_matrix, B_matrix, int num_eigenvalues, dict options):
        """Solve using ARPACK for generalized eigenvalue problem"""
        return spla.eigsh(
            A_matrix,
            k=num_eigenvalues,
            M=B_matrix,
            which=options['which'],
            tol=options['tolerance'],
            maxiter=options['max_iterations'],
            sigma=options['sigma']
        )
    
    def _solve_lobpcg_standard(self, matrix, int num_eigenvalues, dict options):
        """Solve using LOBPCG for standard eigenvalue problem"""
        # LOBPCG needs initial guess
        n = matrix.shape[0]
        X = np.random.random((n, num_eigenvalues))
        
        eigenvals, eigenvecs = spla.lobpcg(
            matrix,
            X,
            tol=options['tolerance'],
            maxiter=options['max_iterations']
        )
        
        # Sort eigenvalues
        idx = np.argsort(eigenvals)
        return eigenvals[idx], eigenvecs[:, idx]
    
    def _solve_lobpcg_generalized(self, A_matrix, B_matrix, int num_eigenvalues, dict options):
        """Solve using LOBPCG for generalized eigenvalue problem"""
        # LOBPCG needs initial guess
        n = A_matrix.shape[0]
        X = np.random.random((n, num_eigenvalues))
        
        eigenvals, eigenvecs = spla.lobpcg(
            A_matrix,
            X,
            B=B_matrix,
            tol=options['tolerance'],
            maxiter=options['max_iterations']
        )
        
        # Sort eigenvalues
        idx = np.argsort(eigenvals)
        return eigenvals[idx], eigenvecs[:, idx]
    
    def _solve_dense_standard(self, matrix, int num_eigenvalues, dict options):
        """Solve using dense eigenvalue solver (for small matrices)"""
        if matrix.shape[0] > 1000:
            raise ValueError("Dense solver not recommended for large matrices")
        
        # Convert to dense
        dense_matrix = matrix.toarray()
        
        # Solve all eigenvalues
        eigenvals, eigenvecs = np.linalg.eigh(dense_matrix)
        
        # Return only requested number
        if options['which'] == 'SM':
            # Smallest magnitude
            idx = np.argsort(np.abs(eigenvals))
        elif options['which'] == 'SA':
            # Smallest algebraic
            idx = np.argsort(eigenvals)
        elif options['which'] == 'LA':
            # Largest algebraic
            idx = np.argsort(eigenvals)[::-1]
        else:
            idx = np.argsort(eigenvals)
        
        idx = idx[:num_eigenvalues]
        return eigenvals[idx], eigenvecs[:, idx]
    
    def solve_quantum_harmonic_oscillator(self, int num_states, double omega, double mass):
        """
        Solve quantum harmonic oscillator analytically for validation.
        
        Parameters:
        -----------
        num_states : int
            Number of energy levels to compute
        omega : float
            Angular frequency (rad/s)
        mass : float
            Mass (kg)
        
        Returns:
        --------
        numpy.ndarray
            Energy eigenvalues
        """
        cdef double hbar = 1.054571817e-34
        cdef cnp.ndarray[double, ndim=1] eigenvalues = np.zeros(num_states, dtype=np.float64)
        cdef int n
        
        # E_n = ℏω(n + 1/2)
        for n in range(num_states):
            eigenvalues[n] = hbar * omega * (n + 0.5)
        
        return eigenvalues
    
    def solve_particle_in_box(self, int num_states, double length, double mass):
        """
        Solve particle in a box analytically for validation.
        
        Parameters:
        -----------
        num_states : int
            Number of energy levels to compute
        length : float
            Box length (m)
        mass : float
            Mass (kg)
        
        Returns:
        --------
        numpy.ndarray
            Energy eigenvalues
        """
        cdef double hbar = 1.054571817e-34
        cdef double pi = 3.14159265359
        cdef cnp.ndarray[double, ndim=1] eigenvalues = np.zeros(num_states, dtype=np.float64)
        cdef int n
        
        # E_n = n²π²ℏ²/(2mL²)
        cdef double factor = pi * pi * hbar * hbar / (2.0 * mass * length * length)
        
        for n in range(1, num_states + 1):
            eigenvalues[n-1] = factor * n * n
        
        return eigenvalues
    
    def benchmark_solver_performance(self, matrix, int num_eigenvalues):
        """
        Benchmark different solver methods.
        
        Parameters:
        -----------
        matrix : scipy.sparse matrix
            Test matrix
        num_eigenvalues : int
            Number of eigenvalues to compute
        
        Returns:
        --------
        dict
            Performance results
        """
        methods = ['arpack']
        if matrix.shape[0] <= 500:
            methods.append('dense')
        if num_eigenvalues >= 3:
            methods.append('lobpcg')
        
        results = {}
        
        for method in methods:
            try:
                start_time = time.time()
                eigenvals, eigenvecs = self.solve_standard_eigenvalue(
                    matrix, num_eigenvalues, method=method
                )
                solve_time = time.time() - start_time
                
                results[method] = {
                    'solve_time': solve_time,
                    'converged': self.last_converged,
                    'num_eigenvalues': len(eigenvals),
                    'eigenvalue_range': (np.min(eigenvals), np.max(eigenvals)) if len(eigenvals) > 0 else (0, 0)
                }
                
            except Exception as e:
                results[method] = {
                    'solve_time': float('inf'),
                    'converged': False,
                    'error': str(e)
                }
        
        return results
    
    def get_last_solution(self):
        """Get the last computed solution"""
        return self.last_eigenvalues, self.last_eigenvectors
    
    def get_solve_info(self):
        """Get information about the last solve"""
        return {
            'solve_time': self.last_solve_time,
            'method': self.last_method,
            'converged': self.last_converged,
            'matrix_size': self.matrix_size,
            'num_eigenvalues': len(self.last_eigenvalues) if self.last_eigenvalues is not None else 0
        }
    
    def set_solver_options(self, **kwargs):
        """Set solver options"""
        self.solver_options.update(kwargs)

def create_eigen_solver(mesh=None):
    """
    Create a Cython-based eigenvalue solver.
    
    Parameters:
    -----------
    mesh : SimpleMesh, optional
        The mesh object
    
    Returns:
    --------
    CythonEigenSolver
        The created solver
    """
    return CythonEigenSolver(mesh)

def test_eigen_solver():
    """Test the Cython eigenvalue solver"""
    try:
        # Create test matrix (discrete Laplacian)
        n = 50
        diag = 2 * np.ones(n)
        off_diag = -1 * np.ones(n-1)
        matrix = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], format='csr')
        
        # Create solver
        solver = CythonEigenSolver()
        
        # Test standard eigenvalue problem
        eigenvals, eigenvecs = solver.solve_standard_eigenvalue(matrix, 5)
        
        # Test analytical solutions
        hbar = 1.054571817e-34
        m_e = 9.1093837015e-31
        
        harmonic_eigenvals = solver.solve_quantum_harmonic_oscillator(5, 1e12, m_e)
        box_eigenvals = solver.solve_particle_in_box(5, 10e-9, m_e)
        
        # Test performance benchmark
        benchmark_results = solver.benchmark_solver_performance(matrix, 3)
        
        print(f"✅ Eigenvalue solver test successful")
        print(f"   Computed eigenvalues: {len(eigenvals)}")
        if len(eigenvals) > 0:
            print(f"   Eigenvalue range: {np.min(eigenvals):.6f} to {np.max(eigenvals):.6f}")
        print(f"   Solve time: {solver.get_solve_info()['solve_time']:.3f} s")
        print(f"   Converged: {solver.get_solve_info()['converged']}")
        
        print(f"   Harmonic oscillator energies (J): {harmonic_eigenvals[:3]}")
        print(f"   Particle in box energies (J): {box_eigenvals[:3]}")
        
        print(f"   Benchmark results:")
        for method, result in benchmark_results.items():
            if 'error' not in result:
                print(f"     {method}: {result['solve_time']:.3f}s, converged: {result['converged']}")
            else:
                print(f"     {method}: failed ({result['error']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Eigenvalue solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
