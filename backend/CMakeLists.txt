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

# Add MPI support (optional)
option(USE_MPI "Enable MPI support" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DUSE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
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
)

# Create the main Python module
pybind11_add_module(qdsim_cpp src/bindings.cpp ${SOURCES})

# Create the FEInterpolator module
pybind11_add_module(fe_interpolator_module src/fe_interpolator_module.cpp src/mesh.cpp src/fe_interpolator.cpp src/adaptive_mesh_simple.cpp)

# Link MPI if enabled
if(USE_MPI)
    target_link_libraries(qdsim_cpp PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(fe_interpolator_module PRIVATE ${MPI_CXX_LIBRARIES})
endif()

# Install the modules
install(TARGETS qdsim_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../frontend/qdsim)
install(TARGETS fe_interpolator_module DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../frontend/qdsim)
