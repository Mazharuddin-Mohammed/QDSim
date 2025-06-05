
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
            print(f"\n   Testing {alg}...")
            
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
            print(f"\n🏆 Best algorithm: {best_alg} ({successful_algs[best_alg]['time']:.3f}s)")
        else:
            print("\n❌ No algorithms succeeded")
        
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
