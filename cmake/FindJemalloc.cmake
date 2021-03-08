# This file is used to find jemalloc library in CMake script, modifeid from the
# code from
#
#   https://github.com/BVLC/caffe/blob/master/cmake/Modules/FindGlog.cmake
#
# which is licensed under the 2-Clause BSD License.
#
# - Try to find Jemalloc
#
# The following variables are optionally searched for defaults
#  JEMALLOC_ROOT_DIR:            Base directory where all JEMALLOC components are found
#
# The following are set after configuration is done:
#  JEMALLOC_FOUND
#  JEMALLOC_INCLUDE_DIRS
#  JEMALLOC_LIBRARIES
#  JEMALLOC_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(JEMALLOC_ROOT_DIR "" CACHE PATH "Folder contains libjemalloc")

# We are testing only a couple of files in the include directories
find_path(JEMALLOC_INCLUDE_DIR jemalloc PATHS ${JEMALLOC_ROOT_DIR}/include)

find_library(JEMALLOC_LIBRARY jemalloc PATHS  ${JEMALLOC_ROOT_DIR}/lib)

find_package_handle_standard_args(JEMALLOC DEFAULT_MSG JEMALLOC_INCLUDE_DIR JEMALLOC_LIBRARY)

if(JEMALLOC_FOUND)
    set(JEMALLOC_INCLUDE_DIRS ${JEMALLOC_INCLUDE_DIR})
    set(JEMALLOC_LIBRARIES ${JEMALLOC_LIBRARY})
    message(STATUS "Found jemalloc (include: ${JEMALLOC_INCLUDE_DIRS}, library: ${JEMALLOC_LIBRARIES})")
    mark_as_advanced(JEMALLOC_LIBRARY_DEBUG JEMALLOC_LIBRARY_RELEASE
                     JEMALLOC_LIBRARY JEMALLOC_INCLUDE_DIR JEMALLOC_ROOT_DIR)
endif()
