#pragma once

#include "unified_parallel_manager.h"
#include "unified_memory_manager.h"
#include "thread_safe_resource_manager.h"
#include "async_gpu_execution_manager.h"
#include "fused_gpu_kernels.cuh"
#include "mesh.h"
#include "materials.h"
#include <vector>
#include <memory>
#include <functional>

namespace QDSim {

// Forward declarations
class Mesh;
class MaterialDatabase;

struct BoundaryConditions {
    std::vector<int> dirichlet_nodes;
    std::vector<double> dirichlet_values;
    std::vector<int> neumann_nodes;
    std::vector<double> neumann_values;
};

class UnifiedQDSimSolver {
public:
    struct SolverConfig {
        UnifiedParallelManager::ParallelConfig parallel_config;
        int max_eigenvalues = 10;
        double convergence_tolerance = 1e-8;
        int max_iterations = 1000;
        bool use_gpu_acceleration = true;
        bool enable_memory_optimization = true;
        bool enable_performance_monitoring = true;
        double temperature = 300.0; // Kelvin
        double fermi_level = 0.0;    // eV
    };
    
    struct QuantumDotResult {
        std::vector<double> eigenvalues;
        std::vector<std::vector<std::complex<double>>> eigenvectors;
        std::vector<double> electron_density;
        std::vector<double> potential;
        double total_energy;
        int iterations_converged;
        double final_error;
        double computation_time_ms;
        
        // Performance metrics
        struct PerformanceMetrics {
            double matrix_assembly_time_ms = 0.0;
            double eigenvalue_solve_time_ms = 0.0;
            double density_computation_time_ms = 0.0;
            double poisson_solve_time_ms = 0.0;
            double total_gpu_time_ms = 0.0;
            double total_cpu_time_ms = 0.0;
            double memory_usage_mb = 0.0;
            double parallel_efficiency = 0.0;
        } performance;
    };
    
    UnifiedQDSimSolver(const SolverConfig& config);
    ~UnifiedQDSimSolver();
    
    // Main solving interface
    QuantumDotResult solveQuantumDot(const Mesh& mesh, 
                                   const MaterialDatabase& materials,
                                   const BoundaryConditions& bc);
    
    // Individual solver components
    void assembleHamiltonian(const Mesh& mesh, 
                           const MaterialDatabase& materials,
                           const std::vector<double>& potential,
                           std::complex<double>* H_matrix,
                           std::complex<double>* M_matrix);
    
    std::vector<double> solveEigenvalueProblem(const std::complex<double>* H_matrix,
                                             const std::complex<double>* M_matrix,
                                             int matrix_size,
                                             std::vector<std::vector<std::complex<double>>>& eigenvectors);
    
    std::vector<double> computeElectronDensity(const std::vector<std::vector<std::complex<double>>>& eigenvectors,
                                             const std::vector<double>& eigenvalues,
                                             const Mesh& mesh);
    
    std::vector<double> solvePoissonEquation(const Mesh& mesh,
                                           const std::vector<double>& charge_density,
                                           const BoundaryConditions& bc);
    
    // Configuration and monitoring
    void updateConfig(const SolverConfig& config);
    SolverConfig getConfig() const { return config_; }
    
    // Performance monitoring
    void enablePerformanceMonitoring(bool enable) { config_.enable_performance_monitoring = enable; }
    void printPerformanceReport() const;
    void resetPerformanceCounters();
    
    // Memory management
    size_t getMemoryUsage() const;
    void optimizeMemoryUsage();
    
    // Error handling
    std::string getLastError() const { return last_error_; }
    bool hasError() const { return !last_error_.empty(); }
    void clearError() { last_error_.clear(); }
    
private:
    SolverConfig config_;
    std::string last_error_;
    
    // Parallel execution components
    std::unique_ptr<AsyncGPUExecutionManager> gpu_executor_;
    
    // Resource managers for thread-safe operations
    std::unique_ptr<ThreadSafeResourceManager<CUDAContext>> cuda_resource_mgr_;
    std::unique_ptr<ThreadSafeResourceManager<EigenSolver>> eigen_solver_mgr_;
    
    // Performance monitoring
    mutable std::mutex performance_mutex_;
    QuantumDotResult::PerformanceMetrics accumulated_performance_;
    
    // Memory management
    std::vector<std::shared_ptr<UnifiedMemoryManager::MemoryBlock>> active_memory_blocks_;
    
    // Helper structures
    struct CUDAContext {
        int device_id = -1;
        cudaStream_t stream = nullptr;
        
        CUDAContext() {
            cudaGetDevice(&device_id);
            cudaStreamCreate(&stream);
        }
        
        ~CUDAContext() {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
        
        void cleanup() {
            if (stream) {
                cudaStreamSynchronize(stream);
            }
        }
    };
    
    struct EigenSolver {
        // Placeholder for eigenvalue solver state
        void* solver_handle = nullptr;
        
        EigenSolver() {
            // Initialize solver (e.g., cuSOLVER handle)
        }
        
        ~EigenSolver() {
            cleanup();
        }
        
        void cleanup() {
            // Cleanup solver resources
        }
    };
    
    // Initialization and cleanup
    void initializeResourceManagers();
    void cleanup();
    
    // GPU-accelerated implementations
    void assembleHamiltonianGPU(const Mesh& mesh,
                               const MaterialDatabase& materials,
                               const std::vector<double>& potential,
                               std::complex<double>* H_matrix,
                               std::complex<double>* M_matrix);
    
    void assembleHamiltonianCPU(const Mesh& mesh,
                               const MaterialDatabase& materials,
                               const std::vector<double>& potential,
                               std::complex<double>* H_matrix,
                               std::complex<double>* M_matrix);
    
    // Self-consistent field iteration
    double computeConvergenceError(const std::vector<double>& old_potential,
                                  const std::vector<double>& new_potential) const;
    
    double computeTotalEnergy(const std::vector<double>& eigenvalues) const;
    
    std::vector<double> initializePotential(const Mesh& mesh,
                                          const MaterialDatabase& materials,
                                          const BoundaryConditions& bc) const;
    
    // Performance monitoring helpers
    void startTimer(const std::string& operation) const;
    void stopTimer(const std::string& operation, double& accumulator) const;
    
    // Error handling helpers
    void setError(const std::string& error);
    void checkCUDAError(const std::string& operation) const;
    
    // Memory allocation helpers
    std::shared_ptr<UnifiedMemoryManager::MemoryBlock> allocateMatrix(size_t rows, size_t cols, 
                                                                     const std::string& name);
    std::shared_ptr<UnifiedMemoryManager::MemoryBlock> allocateVector(size_t size, 
                                                                     const std::string& name);
    
    // Work distribution for parallel execution
    struct ElementAssemblyTask {
        int element_id;
        const Mesh* mesh;
        const MaterialDatabase* materials;
        const std::vector<double>* potential;
        
        ElementAssemblyTask(int id, const Mesh& m, const MaterialDatabase& mat, 
                           const std::vector<double>& pot)
            : element_id(id), mesh(&m), materials(&mat), potential(&pot) {}
    };
    
    void processElementAssembly(const ElementAssemblyTask& task,
                               std::complex<double>* H_matrix,
                               std::complex<double>* M_matrix,
                               const UnifiedParallelManager::ParallelContext& ctx);
};

} // namespace QDSim
