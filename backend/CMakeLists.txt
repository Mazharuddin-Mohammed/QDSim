cmake_minimum_required(VERSION 3.10)
project(qdsim_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find JsonCpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
if(JSONCPP_FOUND)
    include_directories(${JSONCPP_INCLUDE_DIRS})
    link_directories(${JSONCPP_LIBRARY_DIRS})
    add_definitions(-DUSE_JSONCPP)
else()
    message(STATUS "JsonCpp not found. Using bundled version.")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/jsoncpp/include)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/jsoncpp)
    set(JSONCPP_LIBRARIES jsoncpp_lib)
endif()

# Add MPI support (optional)
option(USE_MPI "Enable MPI support" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DUSE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
endif()

# Add CUDA support (optional)
option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        add_definitions(-DUSE_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
        message(STATUS "CUDA found: ${CUDA_VERSION}")
    else()
        message(WARNING "CUDA requested but not found. Disabling CUDA support.")
        set(USE_CUDA OFF)
    endif()
endif()

# Add OpenMP support (optional)
option(USE_OPENMP "Enable OpenMP support" ON)
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_definitions(-DUSE_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP requested but not found. Disabling OpenMP support.")
        set(USE_OPENMP OFF)
    endif()
endif()

# Add SLEPc support (optional)
option(USE_SLEPC "Enable SLEPc support" OFF)
if(USE_SLEPC)
    find_package(PETSc QUIET)
    find_package(SLEPc QUIET)
    if(PETSC_FOUND AND SLEPC_FOUND)
        add_definitions(-DUSE_SLEPC)
        include_directories(${PETSC_INCLUDES} ${SLEPC_INCLUDES})
        message(STATUS "PETSc found: ${PETSC_VERSION}")
        message(STATUS "SLEPc found: ${SLEPC_VERSION}")
    else()
        message(WARNING "SLEPc requested but not found. Disabling SLEPc support.")
        set(USE_SLEPC OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
set(SOURCES
    src/mesh.cpp
    src/physics.cpp
    src/materials.cpp
    src/poisson.cpp
    src/fem.cpp
    src/solver.cpp
    src/adaptive_mesh_simple.cpp
    src/normalization.cpp
    src/fe_interpolator.cpp
    src/self_consistent.cpp
    src/simple_self_consistent.cpp
    src/basic_solver.cpp
    src/improved_self_consistent.cpp
    src/error_estimator.cpp
    src/mesh_quality.cpp
    src/adaptive_refinement.cpp
    src/memory_efficient.cpp
    src/parallel_eigensolver.cpp
    src/gpu_accelerator.cpp
    src/spin_orbit.cpp
    src/error_handling.cpp
    src/simple_interpolator.cpp
    src/diagnostic_manager.cpp
    src/error_visualizer.cpp
    src/cpu_memory_pool.cpp
    src/gpu_memory_pool.cpp
    src/memory_efficient_sparse.cpp
    src/paged_matrix.cpp
)

# CUDA source files
if(USE_CUDA)
    set(CUDA_SOURCES
        src/gpu_kernels.cu
    )

    # Compile CUDA sources
    cuda_add_library(gpu_kernels STATIC ${CUDA_SOURCES})
endif()

# Create the main Python module
pybind11_add_module(qdsim_cpp src/bindings.cpp ${SOURCES})

# Create the FEInterpolator module
pybind11_add_module(fe_interpolator_module src/fe_interpolator_module.cpp src/mesh.cpp src/fe_interpolator.cpp src/adaptive_mesh_simple.cpp)

# Link libraries
target_link_libraries(qdsim_cpp PRIVATE ${JSONCPP_LIBRARIES})
target_link_libraries(fe_interpolator_module PRIVATE ${JSONCPP_LIBRARIES})

if(USE_MPI)
    target_link_libraries(qdsim_cpp PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(fe_interpolator_module PRIVATE ${MPI_CXX_LIBRARIES})
endif()

if(USE_CUDA)
    target_link_libraries(qdsim_cpp PRIVATE gpu_kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
endif()

if(USE_SLEPC)
    target_link_libraries(qdsim_cpp PRIVATE ${PETSC_LIBRARIES} ${SLEPC_LIBRARIES})
endif()

if(USE_OPENMP)
    target_link_libraries(qdsim_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Find and link Spectra library for eigenvalue computations
find_package(Spectra QUIET)
if(Spectra_FOUND)
    target_link_libraries(qdsim_cpp PRIVATE Spectra::Spectra)
    add_definitions(-DUSE_SPECTRA)
else()
    message(STATUS "Spectra library not found. Using Eigen's built-in eigensolvers.")
endif()

# Install the modules
install(TARGETS qdsim_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../frontend/qdsim)
install(TARGETS fe_interpolator_module DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../frontend/qdsim)

# Print configuration summary
message(STATUS "")
message(STATUS "QDSim C++ Configuration Summary:")
message(STATUS "  MPI support:       ${USE_MPI}")
message(STATUS "  CUDA support:      ${USE_CUDA}")
message(STATUS "  OpenMP support:    ${USE_OPENMP}")
message(STATUS "  SLEPc support:     ${USE_SLEPC}")
message(STATUS "  Spectra support:   ${Spectra_FOUND}")
message(STATUS "")
