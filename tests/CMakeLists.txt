# QDSim Comprehensive Testing Framework
cmake_minimum_required(VERSION 3.18)

# Enable testing
enable_testing()

# Find required packages for testing
find_package(GTest QUIET)
find_package(Threads REQUIRED)

# If GTest is not found, try to build it from source
if(NOT GTest_FOUND)
    message(STATUS "GTest not found, attempting to build from source...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9fd7188ec8e5902e4ec0908.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Create aliases to match the find_package targets
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main ALIAS gtest_main)
    set(GTest_FOUND TRUE)
endif()

# Only proceed if GTest is available
if(NOT GTest_FOUND)
    message(WARNING "GTest not available, skipping test compilation")
    return()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/backend/include)

# Compiler flags for testing
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")

# Add ThreadSanitizer flags for parallel correctness tests
set(TSAN_FLAGS "-fsanitize=thread -g -O1 -fno-omit-frame-pointer")
set(ASAN_FLAGS "-fsanitize=address -g -O1 -fno-omit-frame-pointer")
set(MSAN_FLAGS "-fsanitize=memory -g -O1 -fno-omit-frame-pointer")

# Function to create test executable
function(create_test_executable TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    
    # Link with QDSim static library
    target_link_libraries(${TEST_NAME} PRIVATE qdsim_static)
    target_link_libraries(${TEST_NAME} PRIVATE GTest::GTest GTest::Main)
    target_link_libraries(${TEST_NAME} PRIVATE Threads::Threads)
    
    # Link with parallel libraries
    if(USE_OPENMP)
        target_link_libraries(${TEST_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    if(USE_MPI)
        target_link_libraries(${TEST_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
        target_include_directories(${TEST_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    endif()
    
    if(USE_CUDA)
        target_link_libraries(${TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
        set_property(TARGET ${TEST_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
    
    if(USE_NUMA AND NUMA_LIBRARY)
        target_link_libraries(${TEST_NAME} PRIVATE ${NUMA_LIBRARY})
    endif()
    
    # Add test to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Function to create sanitizer test variant
function(create_sanitizer_test TEST_NAME SOURCE_FILE SANITIZER_TYPE)
    set(SANITIZER_TEST_NAME "${TEST_NAME}_${SANITIZER_TYPE}")
    add_executable(${SANITIZER_TEST_NAME} ${SOURCE_FILE})
    
    # Apply sanitizer flags
    if(SANITIZER_TYPE STREQUAL "tsan")
        target_compile_options(${SANITIZER_TEST_NAME} PRIVATE ${TSAN_FLAGS})
        target_link_options(${SANITIZER_TEST_NAME} PRIVATE ${TSAN_FLAGS})
    elseif(SANITIZER_TYPE STREQUAL "asan")
        target_compile_options(${SANITIZER_TEST_NAME} PRIVATE ${ASAN_FLAGS})
        target_link_options(${SANITIZER_TEST_NAME} PRIVATE ${ASAN_FLAGS})
    elseif(SANITIZER_TYPE STREQUAL "msan")
        target_compile_options(${SANITIZER_TEST_NAME} PRIVATE ${MSAN_FLAGS})
        target_link_options(${SANITIZER_TEST_NAME} PRIVATE ${MSAN_FLAGS})
    endif()
    
    # Link libraries
    target_link_libraries(${SANITIZER_TEST_NAME} PRIVATE qdsim_static)
    target_link_libraries(${SANITIZER_TEST_NAME} PRIVATE GTest::GTest GTest::Main)
    target_link_libraries(${SANITIZER_TEST_NAME} PRIVATE Threads::Threads)
    
    if(USE_OPENMP)
        target_link_libraries(${SANITIZER_TEST_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    if(USE_MPI)
        target_link_libraries(${SANITIZER_TEST_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
        target_include_directories(${SANITIZER_TEST_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    endif()
    
    if(USE_CUDA)
        target_link_libraries(${SANITIZER_TEST_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
    endif()
    
    # Add test with special environment
    add_test(NAME ${SANITIZER_TEST_NAME} COMMAND ${SANITIZER_TEST_NAME})
    
    if(SANITIZER_TYPE STREQUAL "tsan")
        set_tests_properties(${SANITIZER_TEST_NAME} PROPERTIES 
            ENVIRONMENT "TSAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_thread_leaks=1")
    elseif(SANITIZER_TYPE STREQUAL "asan")
        set_tests_properties(${SANITIZER_TEST_NAME} PROPERTIES 
            ENVIRONMENT "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=1")
    endif()
endfunction()

# 1. PARALLEL CORRECTNESS TESTS
if(USE_OPENMP OR USE_MPI)
    # Thread sanitizer tests
    create_test_executable(thread_sanitizer_tests parallel_correctness/thread_sanitizer_tests.cpp)
    create_sanitizer_test(thread_sanitizer_tests parallel_correctness/thread_sanitizer_tests.cpp tsan)
    
    # Set test properties for thread sanitizer
    set_tests_properties(thread_sanitizer_tests_tsan PROPERTIES 
        TIMEOUT 300
        LABELS "parallel_correctness;thread_safety")
endif()

# GPU memory checker tests
if(USE_CUDA)
    create_test_executable(gpu_memory_checker parallel_correctness/gpu_memory_checker.cpp)
    create_sanitizer_test(gpu_memory_checker parallel_correctness/gpu_memory_checker.cpp asan)
    
    # Add CUDA-memcheck test
    add_test(NAME gpu_memory_checker_cuda_memcheck 
             COMMAND cuda-memcheck --tool=memcheck --error-exitcode=1 $<TARGET_FILE:gpu_memory_checker>)
    
    set_tests_properties(gpu_memory_checker_cuda_memcheck PROPERTIES 
        TIMEOUT 600
        LABELS "parallel_correctness;gpu_memory")
endif()

# 2. PERFORMANCE BENCHMARKING TESTS
create_test_executable(scalability_tests performance_benchmarking/scalability_tests.cpp)
create_test_executable(bandwidth_utilization_tests performance_benchmarking/bandwidth_utilization_tests.cpp)

# Set performance test properties
set_tests_properties(scalability_tests PROPERTIES 
    TIMEOUT 600
    LABELS "performance;scalability")

set_tests_properties(bandwidth_utilization_tests PROPERTIES 
    TIMEOUT 300
    LABELS "performance;bandwidth")

# 3. CROSS-PLATFORM VALIDATION TESTS
if(USE_CUDA)
    create_test_executable(cuda_architecture_tests cross_platform_validation/cuda_architecture_tests.cpp)
    
    set_tests_properties(cuda_architecture_tests PROPERTIES 
        TIMEOUT 300
        LABELS "cross_platform;cuda")
endif()

if(USE_MPI)
    create_test_executable(mpi_implementation_tests cross_platform_validation/mpi_implementation_tests.cpp)
    
    # MPI tests need special handling
    add_test(NAME mpi_implementation_tests_single 
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:mpi_implementation_tests>)
    
    add_test(NAME mpi_implementation_tests_multi 
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:mpi_implementation_tests>)
    
    set_tests_properties(mpi_implementation_tests_single mpi_implementation_tests_multi PROPERTIES 
        TIMEOUT 300
        LABELS "cross_platform;mpi")
endif()

# 4. INTEGRATION TESTS
create_test_executable(unified_architecture_integration_test unified_architecture_integration_test.cpp)

set_tests_properties(unified_architecture_integration_test PROPERTIES 
    TIMEOUT 600
    LABELS "integration;unified_architecture")

# 5. STRESS TESTS
create_test_executable(stress_tests stress_tests.cpp)

set_tests_properties(stress_tests PROPERTIES 
    TIMEOUT 1800  # 30 minutes
    LABELS "stress;long_running")

# Custom test targets
add_custom_target(test_parallel_correctness
    COMMAND ${CMAKE_CTEST_COMMAND} -L "parallel_correctness" --output-on-failure
    COMMENT "Running parallel correctness tests")

add_custom_target(test_performance
    COMMAND ${CMAKE_CTEST_COMMAND} -L "performance" --output-on-failure
    COMMENT "Running performance benchmarking tests")

add_custom_target(test_cross_platform
    COMMAND ${CMAKE_CTEST_COMMAND} -L "cross_platform" --output-on-failure
    COMMENT "Running cross-platform validation tests")

add_custom_target(test_sanitizers
    COMMAND ${CMAKE_CTEST_COMMAND} -R ".*_(tsan|asan|msan)" --output-on-failure
    COMMENT "Running sanitizer tests")

add_custom_target(test_quick
    COMMAND ${CMAKE_CTEST_COMMAND} -E "(stress|long_running)" --output-on-failure
    COMMENT "Running quick tests (excluding stress tests)")

add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running all tests")

# Generate test report
add_custom_target(test_report
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --output-junit test_results.xml
    COMMENT "Running tests and generating XML report")

# Valgrind integration
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(test_valgrind
        COMMAND ${CMAKE_CTEST_COMMAND} -T memcheck --output-on-failure
        COMMENT "Running tests with Valgrind")
endif()

# Code coverage (if GCC/Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    option(ENABLE_COVERAGE "Enable code coverage" OFF)
    
    if(ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        
        find_program(GCOV_EXECUTABLE gcov)
        find_program(LCOV_EXECUTABLE lcov)
        find_program(GENHTML_EXECUTABLE genhtml)
        
        if(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
            add_custom_target(coverage
                COMMAND ${LCOV_EXECUTABLE} --directory . --capture --output-file coverage.info
                COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' --output-file coverage.info
                COMMAND ${LCOV_EXECUTABLE} --list coverage.info
                COMMAND ${GENHTML_EXECUTABLE} coverage.info --output-directory coverage_html
                COMMENT "Generating code coverage report")
        endif()
    endif()
endif()

# Print test configuration summary
message(STATUS "QDSim Testing Configuration:")
message(STATUS "  Parallel Correctness Tests: ${USE_OPENMP} (OpenMP) ${USE_MPI} (MPI)")
message(STATUS "  GPU Memory Tests: ${USE_CUDA}")
message(STATUS "  Cross-Platform Tests: ${USE_CUDA} (CUDA) ${USE_MPI} (MPI)")
message(STATUS "  Sanitizers: Available")
message(STATUS "  Valgrind: ${VALGRIND_EXECUTABLE}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")

# Test data directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_reports)

# Install test scripts
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_comprehensive_tests.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/run_comprehensive_tests.sh @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/analyze_test_results.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/analyze_test_results.py @ONLY)
