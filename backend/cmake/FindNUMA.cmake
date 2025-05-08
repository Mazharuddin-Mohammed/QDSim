# FindNUMA.cmake
# Find the NUMA library
#
# This module defines:
#  NUMA_FOUND        - True if NUMA was found
#  NUMA_INCLUDE_DIRS - The NUMA include directories
#  NUMA_LIBRARIES    - The NUMA libraries
#
# Author: Dr. Mazharuddin Mohammed

# Find the NUMA header
find_path(NUMA_INCLUDE_DIR
    NAMES numa.h
    PATHS
        /usr/include
        /usr/local/include
)

# Find the NUMA library
find_library(NUMA_LIBRARY
    NAMES numa
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
)

# Set the NUMA_FOUND variable
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA
    REQUIRED_VARS NUMA_LIBRARY NUMA_INCLUDE_DIR
)

# Set the output variables
if(NUMA_FOUND)
    set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})
    set(NUMA_LIBRARIES ${NUMA_LIBRARY})
endif()

# Mark as advanced
mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY)
