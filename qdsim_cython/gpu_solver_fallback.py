
# GPU Solver with CPU Fallback
import numpy as np
import time

class GPUSolverFallback:
    """GPU solver with automatic CPU fallback"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and self._check_gpu_availability()
        self.device_info = self._get_device_info()
        
        if self.use_gpu:
            print("✅ GPU acceleration enabled")
        else:
            print("⚠️  Using CPU fallback")
    
    def _check_gpu_availability(self):
        """Check if GPU is available"""
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            return False
    
    def _get_device_info(self):
        """Get device information"""
        if self.use_gpu:
            try:
                import cupy as cp
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                return {
                    'name': device_props['name'].decode('utf-8'),
                    'memory_gb': device_props['totalGlobalMem'] / 1024**3,
                    'type': 'GPU'
                }
            except:
                pass
        
        return {'name': 'CPU', 'type': 'CPU'}
    
    def solve_eigenvalue_problem(self, H_matrix, M_matrix, num_eigenvalues):
        """Solve eigenvalue problem with GPU/CPU fallback"""
        print(f"🔧 Solving on {self.device_info['type']}: {self.device_info['name']}")
        
        start_time = time.time()
        
        if self.use_gpu:
            try:
                eigenvals, eigenvecs = self._solve_gpu(H_matrix, M_matrix, num_eigenvalues)
            except Exception as e:
                print(f"   GPU solve failed: {e}")
                print("   Falling back to CPU...")
                eigenvals, eigenvecs = self._solve_cpu(H_matrix, M_matrix, num_eigenvalues)
        else:
            eigenvals, eigenvecs = self._solve_cpu(H_matrix, M_matrix, num_eigenvalues)
        
        solve_time = time.time() - start_time
        print(f"✅ Solved in {solve_time:.3f}s")
        
        return eigenvals, eigenvecs
    
    def _solve_gpu(self, H_matrix, M_matrix, num_eigenvalues):
        """GPU eigenvalue solver"""
        import cupy as cp
        import cupyx.scipy.sparse as cp_sparse
        import cupyx.scipy.sparse.linalg as cp_linalg
        
        # Transfer to GPU
        H_gpu = cp_sparse.csr_matrix(H_matrix)
        M_gpu = cp_sparse.csr_matrix(M_matrix)
        
        # Apply boundary conditions
        num_nodes = H_gpu.shape[0]
        H_bc = H_gpu.tolil()
        M_bc = M_gpu.tolil()
        
        boundary_nodes = [0, num_nodes-1]
        for node in boundary_nodes:
            H_bc[node, :] = 0
            H_bc[node, node] = 1.0
            M_bc[node, :] = 0
            M_bc[node, node] = 1.0
        
        H_bc = H_bc.tocsr()
        M_bc = M_bc.tocsr()
        
        # Solve (simplified - for small matrices)
        if H_bc.shape[0] < 1000:
            H_dense = H_bc.toarray()
            M_dense = M_bc.toarray()
            
            eigenvals_gpu, eigenvecs_gpu = cp.linalg.eigh(H_dense, M_dense)
            
            # Sort and select
            idx = cp.argsort(eigenvals_gpu)
            max_eigs = min(num_eigenvalues, len(eigenvals_gpu))
            eigenvals_gpu = eigenvals_gpu[idx[:max_eigs]]
            eigenvecs_gpu = eigenvecs_gpu[:, idx[:max_eigs]]
            
            # Transfer back to CPU
            eigenvals = cp.asnumpy(eigenvals_gpu)
            eigenvecs = cp.asnumpy(eigenvecs_gpu)
            
            return eigenvals, eigenvecs
        else:
            raise ValueError("Large matrices not supported in GPU mode yet")
    
    def _solve_cpu(self, H_matrix, M_matrix, num_eigenvalues):
        """CPU eigenvalue solver"""
        import scipy.sparse.linalg as spla
        
        # Apply boundary conditions
        num_nodes = H_matrix.shape[0]
        H_bc = H_matrix.tolil()
        M_bc = M_matrix.tolil()
        
        boundary_nodes = [0, num_nodes-1]
        for node in boundary_nodes:
            H_bc[node, :] = 0
            H_bc[node, node] = 1.0
            M_bc[node, :] = 0
            M_bc[node, node] = 1.0
        
        H_bc = H_bc.tocsr()
        M_bc = M_bc.tocsr()
        
        # Solve
        max_eigs = min(num_eigenvalues, num_nodes - 3)
        if max_eigs < 1:
            max_eigs = 1
        
        eigenvals, eigenvecs = spla.eigsh(
            H_bc, k=max_eigs, M=M_bc, which='SM', tol=1e-6, maxiter=1000
        )
        
        return eigenvals, eigenvecs

def test_gpu_solver():
    """Test the GPU solver"""
    solver = GPUSolverFallback()
    print(f"Device: {solver.device_info}")
    return True
