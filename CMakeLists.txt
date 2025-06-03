cmake_minimum_required(VERSION 3.10)
project(QDSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options for unified parallel architecture
option(USE_MPI "Enable MPI support" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_CUDA "Enable CUDA support" ON)
option(USE_PYTHON_INTEGRATION "Enable Python integration" ON)
option(BUILD_TESTS "Build test suite" ON)
option(USE_NUMA "Enable NUMA support" ON)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/backend/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenMP support
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DUSE_OPENMP)
endif()

# MPI support
if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    add_definitions(-DUSE_MPI)
endif()

# NUMA support (Linux only)
if(USE_NUMA AND UNIX AND NOT APPLE)
    find_library(NUMA_LIBRARY numa)
    if(NUMA_LIBRARY)
        add_definitions(-DUSE_NUMA)
    endif()
endif()

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)

    # Find cuBLAS and cuSOLVER
    find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(CUSOLVER_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(CUSPARSE_LIBRARY cusparse HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

    if(NOT CUBLAS_LIBRARY OR NOT CUSOLVER_LIBRARY OR NOT CUSPARSE_LIBRARY)
        message(FATAL_ERROR "cuBLAS, cuSOLVER, or cuSPARSE not found")
    endif()

    # Set CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
endif()

set(SOURCES
    # Original sources
    backend/src/mesh.cpp
    backend/src/fem.cpp
    backend/src/physics.cpp
    backend/src/solver.cpp
    backend/src/adaptive_mesh.cpp
    backend/src/normalization.cpp
    backend/src/poisson.cpp
    backend/src/self_consistent.cpp
    backend/src/materials.cpp
    backend/src/simple_mesh.cpp
    backend/src/simple_interpolator.cpp
    backend/src/pn_junction.cpp
    backend/src/basic_solver.cpp
    backend/src/improved_self_consistent.cpp
    backend/src/fe_interpolator.cpp
    backend/src/simple_self_consistent.cpp
    backend/src/schrodinger.cpp
    backend/src/gpu_accelerator.cpp
    backend/src/gpu_kernels.cu
    backend/src/gpu_eigensolver.cu
    backend/src/gpu_memory_pool.cpp
    backend/src/full_poisson_dd_solver.cpp
    backend/src/callback_system.cpp
    backend/src/bindings.cpp

    # Unified parallel architecture sources
    backend/src/unified_parallel_manager.cpp
    backend/src/unified_memory_manager.cpp
    backend/src/async_gpu_execution_manager.cpp
    backend/src/fused_gpu_kernels.cu
)

pybind11_add_module(qdsim_cpp ${SOURCES})
target_link_libraries(qdsim_cpp PRIVATE pybind11::module)

# Link libraries for unified parallel architecture
if(USE_OPENMP)
    target_link_libraries(qdsim_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_MPI)
    target_link_libraries(qdsim_cpp PRIVATE ${MPI_CXX_LIBRARIES})
endif()
if(USE_CUDA)
    target_link_libraries(qdsim_cpp PRIVATE ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY} ${CUSPARSE_LIBRARY})
endif()
if(USE_NUMA AND NUMA_LIBRARY)
    target_link_libraries(qdsim_cpp PRIVATE ${NUMA_LIBRARY})
endif()

# Create a static library for QDSim (excluding Python bindings)
set(STATIC_SOURCES
    # Original sources
    backend/src/mesh.cpp
    backend/src/fem.cpp
    backend/src/physics.cpp
    backend/src/solver.cpp
    backend/src/adaptive_mesh.cpp
    backend/src/normalization.cpp
    backend/src/poisson.cpp
    backend/src/self_consistent.cpp
    backend/src/materials.cpp
    backend/src/simple_mesh.cpp
    backend/src/simple_interpolator.cpp
    backend/src/pn_junction.cpp
    backend/src/basic_solver.cpp
    backend/src/improved_self_consistent.cpp
    backend/src/fe_interpolator.cpp
    backend/src/simple_self_consistent.cpp
    backend/src/schrodinger.cpp
    backend/src/gpu_accelerator.cpp
    backend/src/gpu_kernels.cu
    backend/src/gpu_eigensolver.cu
    backend/src/gpu_cusolver.cu
    backend/src/gpu_memory_pool.cpp
    backend/src/callback_system.cpp
    backend/src/full_poisson_dd_solver.cpp

    # Unified parallel architecture sources
    backend/src/unified_parallel_manager.cpp
    backend/src/unified_memory_manager.cpp
    backend/src/async_gpu_execution_manager.cpp
    backend/src/fused_gpu_kernels.cu
)
add_library(qdsim_static STATIC ${STATIC_SOURCES})
target_link_libraries(qdsim_static PRIVATE ${EIGEN3_LIBRARIES})

# Link libraries for unified parallel architecture
if(USE_OPENMP)
    target_link_libraries(qdsim_static PRIVATE OpenMP::OpenMP_CXX)
endif()
if(USE_MPI)
    target_link_libraries(qdsim_static PRIVATE ${MPI_CXX_LIBRARIES})
endif()
if(USE_CUDA)
    target_link_libraries(qdsim_static PRIVATE ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY} ${CUSPARSE_LIBRARY})
endif()
if(USE_NUMA AND NUMA_LIBRARY)
    target_link_libraries(qdsim_static PRIVATE ${NUMA_LIBRARY})
endif()

# Add the GPU acceleration test executable
add_executable(gpu_acceleration_test examples/gpu_acceleration_cpp_test.cpp)
target_link_libraries(gpu_acceleration_test PRIVATE qdsim_static ${EIGEN3_LIBRARIES})
if(USE_MPI)
    target_link_libraries(gpu_acceleration_test PRIVATE ${MPI_CXX_LIBRARIES})
endif()
if(USE_CUDA)
    target_link_libraries(gpu_acceleration_test PRIVATE ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY} ${CUSPARSE_LIBRARY})
endif()

# Add the simple GPU test executable
add_executable(simple_gpu_test examples/simple_gpu_test.cpp)
target_link_libraries(simple_gpu_test PRIVATE qdsim_static ${EIGEN3_LIBRARIES})
if(USE_MPI)
    target_link_libraries(simple_gpu_test PRIVATE ${MPI_CXX_LIBRARIES})
endif()
if(USE_CUDA)
    target_link_libraries(simple_gpu_test PRIVATE ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY} ${CUSPARSE_LIBRARY})
endif()

# Add the Poisson-Drift-Diffusion test executable
add_executable(poisson_dd_test examples/poisson_dd_test.cpp)
target_link_libraries(poisson_dd_test PRIVATE qdsim_static ${EIGEN3_LIBRARIES})
if(USE_MPI)
    target_link_libraries(poisson_dd_test PRIVATE ${MPI_CXX_LIBRARIES})
endif()
if(USE_CUDA)
    target_link_libraries(poisson_dd_test PRIVATE ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY} ${CUSPARSE_LIBRARY})
endif()

# Make testing optional
option(BUILD_TESTING "Build the testing tree" OFF)

if(BUILD_TESTING)
    enable_testing()
    find_package(Catch2 QUIET)
    if(Catch2_FOUND)
        set(TEST_SOURCES
            backend/tests/test_mesh.cpp
            backend/tests/test_physics.cpp
            backend/tests/test_solver.cpp
            backend/tests/test_adaptive_mesh.cpp
            backend/tests/test_normalization.cpp
            backend/tests/test_poisson.cpp
            backend/tests/test_self_consistent.cpp
            backend/tests/test_materials.cpp
        )

        foreach(test_src ${TEST_SOURCES})
            get_filename_component(test_name ${test_src} NAME_WE)
            add_executable(${test_name} ${test_src})
            target_link_libraries(${test_name} PRIVATE Catch2::Catch2)
            if(USE_MPI)
                target_link_libraries(${test_name} PRIVATE ${MPI_CXX_LIBRARIES})
            endif()
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()
    else()
        message(STATUS "Catch2 not found, tests will not be built")
    endif()
endif()