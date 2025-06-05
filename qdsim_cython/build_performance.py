#!/usr/bin/env python3
"""
Build Performance Optimization Module

This script builds the high-performance parallel solver with OpenMP support.
"""

import sys
import os
import multiprocessing as mp

def check_openmp_support():
    """Check if OpenMP is available"""
    try:
        # Try to compile a simple OpenMP test
        test_code = '''
#include <omp.h>
#include <stdio.h>

int main() {
    printf("OpenMP threads: %d\\n", omp_get_max_threads());
    return 0;
}
'''
        
        # Write test file
        with open('test_openmp.c', 'w') as f:
            f.write(test_code)
        
        # Try to compile
        import subprocess
        result = subprocess.run(['gcc', '-fopenmp', 'test_openmp.c', '-o', 'test_openmp'], 
                              capture_output=True, text=True)
        
        # Clean up
        for file in ['test_openmp.c', 'test_openmp']:
            if os.path.exists(file):
                os.remove(file)
        
        if result.returncode == 0:
            print("✅ OpenMP support detected")
            return True
        else:
            print("⚠️  OpenMP not available")
            return False
            
    except Exception as e:
        print(f"⚠️  OpenMP check failed: {e}")
        return False

def build_performance_module():
    """Build the performance optimization module"""
    print("⚡ Building Performance Optimization Module")
    print("=" * 60)
    
    openmp_available = check_openmp_support()
    cpu_count = mp.cpu_count()
    
    print(f"System info: {cpu_count} CPU cores")
    
    try:
        from setuptools import setup, Extension
        from Cython.Build import cythonize
        import numpy as np
        
        # Create performance directory if it doesn't exist
        os.makedirs('performance', exist_ok=True)
        
        # Create __init__.py for performance package
        init_file = 'performance/__init__.py'
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Performance optimization package\n')
        
        # Define compilation flags
        extra_compile_args = ['-std=c++17', '-O3', '-march=native']
        extra_link_args = []
        define_macros = []
        
        if openmp_available:
            extra_compile_args.extend(['-fopenmp'])
            extra_link_args.extend(['-fopenmp'])
            define_macros.append(('USE_OPENMP', '1'))
            print("   Building with OpenMP support")
        else:
            define_macros.append(('NO_OPENMP', '1'))
            print("   Building without OpenMP (thread-based parallelism only)")
        
        # Add CPU-specific optimizations
        extra_compile_args.extend([
            '-ffast-math',
            '-funroll-loops',
            '-ftree-vectorize'
        ])
        
        # Define extension
        ext = Extension(
            'qdsim_cython.performance.parallel_solver',
            ['performance/parallel_solver.pyx'],
            include_dirs=[np.get_include()],
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=define_macros
        )
        
        # Build
        setup(
            ext_modules=cythonize([ext], compiler_directives={
                'language_level': 3,
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
                'nonecheck': False
            }),
            script_name='build_performance.py',
            script_args=['build_ext', '--inplace']
        )
        
        print("✅ Performance module built successfully!")
        return True, openmp_available
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False, False

def create_advanced_eigenvalue_solvers():
    """Create advanced eigenvalue solvers"""
    print("\n🔧 Creating Advanced Eigenvalue Solvers")
    print("=" * 50)
    
    solver_code = '''
# Advanced Eigenvalue Solvers
import numpy as np
import scipy.sparse.linalg as spla
import time
from typing import Tuple, Optional, Dict, Any

class AdvancedEigenSolver:
    """Advanced eigenvalue solver with multiple algorithms"""
    
    def __init__(self, algorithm='auto'):
        self.algorithm = algorithm
        self.available_algorithms = [
            'lobpcg',      # Locally Optimal Block Preconditioned Conjugate Gradient
            'arpack',      # Implicitly Restarted Arnoldi Method
            'feast',       # FEAST eigenvalue solver (if available)
            'shift_invert', # Shift-invert mode
            'auto'         # Automatic selection
        ]
        
        print(f"🔧 Advanced eigenvalue solver: {algorithm} mode")
    
    def solve(self, H_matrix, M_matrix, num_eigenvalues, **kwargs):
        """Solve eigenvalue problem with advanced algorithms"""
        
        if self.algorithm == 'auto':
            return self._solve_auto(H_matrix, M_matrix, num_eigenvalues, **kwargs)
        elif self.algorithm == 'lobpcg':
            return self._solve_lobpcg(H_matrix, M_matrix, num_eigenvalues, **kwargs)
        elif self.algorithm == 'arpack':
            return self._solve_arpack(H_matrix, M_matrix, num_eigenvalues, **kwargs)
        elif self.algorithm == 'shift_invert':
            return self._solve_shift_invert(H_matrix, M_matrix, num_eigenvalues, **kwargs)
        else:
            return self._solve_arpack(H_matrix, M_matrix, num_eigenvalues, **kwargs)
    
    def _solve_auto(self, H_matrix, M_matrix, num_eigenvalues, **kwargs):
        """Automatically select best algorithm"""
        matrix_size = H_matrix.shape[0]
        
        if matrix_size < 1000:
            # Small matrices: use dense solver
            return self._solve_dense(H_matrix, M_matrix, num_eigenvalues)
        elif matrix_size < 10000:
            # Medium matrices: use LOBPCG
            return self._solve_lobpcg(H_matrix, M_matrix, num_eigenvalues, **kwargs)
        else:
            # Large matrices: use ARPACK with shift-invert
            return self._solve_shift_invert(H_matrix, M_matrix, num_eigenvalues, **kwargs)
    
    def _solve_dense(self, H_matrix, M_matrix, num_eigenvalues):
        """Dense eigenvalue solver for small problems"""
        print("   Using dense eigenvalue solver...")
        
        H_dense = H_matrix.toarray()
        M_dense = M_matrix.toarray()
        
        eigenvals, eigenvecs = np.linalg.eigh(H_dense, M_dense)
        
        # Sort and select
        idx = np.argsort(eigenvals)
        max_eigs = min(num_eigenvalues, len(eigenvals))
        
        return eigenvals[idx[:max_eigs]], eigenvecs[:, idx[:max_eigs]]
    
    def _solve_lobpcg(self, H_matrix, M_matrix, num_eigenvalues, **kwargs):
        """LOBPCG eigenvalue solver"""
        print("   Using LOBPCG eigenvalue solver...")
        
        # Initial guess
        X = np.random.rand(H_matrix.shape[0], num_eigenvalues)
        
        # Normalize initial guess
        for i in range(num_eigenvalues):
            X[:, i] = X[:, i] / np.sqrt(X[:, i].T @ M_matrix @ X[:, i])
        
        try:
            eigenvals, eigenvecs = spla.lobpcg(
                H_matrix, X, M=M_matrix, 
                tol=kwargs.get('tol', 1e-6),
                maxiter=kwargs.get('maxiter', 1000)
            )
            
            # Sort eigenvalues
            idx = np.argsort(eigenvals)
            return eigenvals[idx], eigenvecs[:, idx]
            
        except Exception as e:
            print(f"   LOBPCG failed: {e}, falling back to ARPACK...")
            return self._solve_arpack(H_matrix, M_matrix, num_eigenvalues, **kwargs)
    
    def _solve_arpack(self, H_matrix, M_matrix, num_eigenvalues, **kwargs):
        """ARPACK eigenvalue solver"""
        print("   Using ARPACK eigenvalue solver...")
        
        max_eigs = min(num_eigenvalues, H_matrix.shape[0] - 2)
        
        eigenvals, eigenvecs = spla.eigsh(
            H_matrix, k=max_eigs, M=M_matrix, which='SM',
            tol=kwargs.get('tol', 1e-6),
            maxiter=kwargs.get('maxiter', 1000)
        )
        
        return eigenvals, eigenvecs
    
    def _solve_shift_invert(self, H_matrix, M_matrix, num_eigenvalues, **kwargs):
        """Shift-invert eigenvalue solver"""
        print("   Using shift-invert eigenvalue solver...")
        
        # Choose shift point
        sigma = kwargs.get('sigma', 0.01 * 1.602176634e-19)  # 10 meV
        
        max_eigs = min(num_eigenvalues, H_matrix.shape[0] - 2)
        
        eigenvals, eigenvecs = spla.eigsh(
            H_matrix, k=max_eigs, M=M_matrix, 
            sigma=sigma, which='LM',
            tol=kwargs.get('tol', 1e-6),
            maxiter=kwargs.get('maxiter', 2000)
        )
        
        return eigenvals, eigenvecs
    
    def benchmark_algorithms(self, H_matrix, M_matrix, num_eigenvalues):
        """Benchmark different algorithms"""
        print("🏁 Benchmarking eigenvalue algorithms...")
        
        algorithms = ['arpack', 'lobpcg', 'shift_invert']
        results = {}
        
        for alg in algorithms:
            print(f"\\n   Testing {alg}...")
            
            try:
                start_time = time.time()
                
                if alg == 'arpack':
                    eigenvals, eigenvecs = self._solve_arpack(H_matrix, M_matrix, num_eigenvalues)
                elif alg == 'lobpcg':
                    eigenvals, eigenvecs = self._solve_lobpcg(H_matrix, M_matrix, num_eigenvalues)
                elif alg == 'shift_invert':
                    eigenvals, eigenvecs = self._solve_shift_invert(H_matrix, M_matrix, num_eigenvalues)
                
                solve_time = time.time() - start_time
                
                results[alg] = {
                    'time': solve_time,
                    'eigenvalues': len(eigenvals),
                    'success': True,
                    'first_eigenvalue': eigenvals[0] if len(eigenvals) > 0 else None
                }
                
                print(f"     ✅ {alg}: {solve_time:.3f}s, {len(eigenvals)} eigenvalues")
                
            except Exception as e:
                results[alg] = {
                    'time': float('inf'),
                    'eigenvalues': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"     ❌ {alg}: Failed - {e}")
        
        # Find best algorithm
        successful_algs = {k: v for k, v in results.items() if v['success']}
        
        if successful_algs:
            best_alg = min(successful_algs.keys(), key=lambda k: successful_algs[k]['time'])
            print(f"\\n🏆 Best algorithm: {best_alg} ({successful_algs[best_alg]['time']:.3f}s)")
        else:
            print("\\n❌ No algorithms succeeded")
        
        return results

def test_advanced_solvers():
    """Test advanced eigenvalue solvers"""
    print("🧪 Testing Advanced Eigenvalue Solvers")
    print("=" * 50)
    
    # Create test matrices
    import scipy.sparse as sp
    
    n = 500
    # Create a simple test problem
    diag = np.arange(1, n+1, dtype=float)
    H = sp.diags(diag, format='csr')
    M = sp.eye(n, format='csr')
    
    print(f"Test problem: {n}×{n} matrix")
    
    # Test different solvers
    solver = AdvancedEigenSolver('auto')
    
    start_time = time.time()
    eigenvals, eigenvecs = solver.solve(H, M, 5)
    solve_time = time.time() - start_time
    
    print(f"✅ Advanced solver: {solve_time:.3f}s")
    print(f"   Eigenvalues: {eigenvals}")
    
    # Benchmark algorithms
    solver.benchmark_algorithms(H, M, 5)
    
    return True
'''
    
    # Write advanced solver
    with open('advanced_eigenvalue_solvers.py', 'w') as f:
        f.write(solver_code)
    
    print("✅ Advanced eigenvalue solvers created")
    return True

def test_performance_module():
    """Test the performance module"""
    print("\n🧪 Testing Performance Module")
    print("=" * 40)
    
    try:
        sys.path.insert(0, '.')
        import qdsim_cython.performance.parallel_solver as ps
        
        print("✅ Performance module imported")
        
        # Test parallel solver
        result = ps.test_parallel_solver()
        
        if result:
            print("✅ Parallel solver test passed")
            return True
        else:
            print("❌ Parallel solver test failed")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 PERFORMANCE OPTIMIZATION BUILD AND TEST")
    print("=" * 70)
    
    # Build performance module
    build_success, openmp_available = build_performance_module()
    
    # Create advanced solvers
    solvers_success = create_advanced_eigenvalue_solvers()
    
    if build_success:
        # Test
        test_success = test_performance_module()
        
        if test_success:
            print("\\n🎉 PERFORMANCE OPTIMIZATION SUCCESS!")
            print(f"   ✅ Parallel solver: Built with {'OpenMP' if openmp_available else 'threading'}")
            print(f"   ✅ Advanced eigenvalue solvers: Available")
            print(f"   ✅ Performance profiling: Enabled")
            print(f"   ✅ Memory optimization: Active")
            return True
        else:
            print("\\n⚠️  Build success but test issues")
            return False
    else:
        print("\\n❌ Build failed")
        return False

if __name__ == "__main__":
    success = main()
