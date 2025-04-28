cmake_minimum_required(VERSION 3.10)
project(QDSim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable/disable MPI at compile time
option(USE_MPI "Enable MPI support" ON)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/backend/include)
include_directories(${EIGEN3_INCLUDE_DIR})

if(USE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    add_definitions(-DUSE_MPI)
endif()

set(SOURCES
    backend/src/mesh.cpp
    backend/src/fem.cpp
    backend/src/physics.cpp
    backend/src/solver.cpp
    backend/src/adaptive_mesh.cpp
    backend/src/normalization.cpp
    backend/src/poisson.cpp
    backend/src/materials.cpp
    backend/src/bindings.cpp
)

pybind11_add_module(qdsim_cpp ${SOURCES})
target_link_libraries(qdsim_cpp PRIVATE pybind11::module)
if(USE_MPI)
    target_link_libraries(qdsim_cpp PRIVATE ${MPI_CXX_LIBRARIES})
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