# This file is used to find fuse3 library in CMake script, based on code
# from
#
#   https://github.com/fkie-cad/pcapFS/blob/master/cmake/Modules/FindFUSE3.cmake
#
# which is licensed under the MIT License.
#
# FindFUSE3.cmake
#
# Finds the FUSE3 library.
#
# This will define the following variables
#
#    FUSE3_FOUND
#    FUSE3_INCLUDE_DIRS
#    FUSE3_LIBRARIES
#    FUSE3_VERSION
#
# and the following imported targets
#
#    FUSE3::FUSE3
#

if(FUSE3_INCLUDE_DIR AND FUSE3_LIBRARY)
    set(FUSE_FIND_QUIETLY TRUE)
endif()

find_package(PkgConfig)
pkg_check_modules(PC_FUSE3 REQUIRED QUIET fuse3)

find_path(FUSE3_INCLUDE_DIR
          NAMES fuse_lowlevel.h
          PATHS "${PC_FUSE3_INCLUDE_DIRS}"
          PATH_SUFFIXES fuse3
)

find_library(FUSE3_LIBRARY
             NAMES fuse3
             PATHS "${PC_FUSE3_LIBDIR}"
)

set(FUSE3_VERSION "${PC_FUSE3_VERSION}")

mark_as_advanced(FUSE3_INCLUDE_DIR FUSE3_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FUSE3
    REQUIRED_VARS FUSE3_INCLUDE_DIR FUSE3_LIBRARY
    VERSION_VAR FUSE3_VERSION_STRING
)

if(FUSE_FOUND)
    set(FUSE_INCLUDE_DIRS ${FUSE_INCLUDE_DIR})
    set(FUSE_LIBRARIES ${FUSE_LIBRARY})
endif()

if(FUSE3_FOUND AND NOT TARGET FUSE::FUSE)
    add_library(FUSE3::FUSE3 INTERFACE IMPORTED)
    set_target_properties(FUSE3::FUSE3 PROPERTIES
                          INTERFACE_INCLUDE_DIRECTORIES "${FUSE3_INCLUDE_DIR}"
                          INTERFACE_LINK_LIBRARIES "${FUSE3_LIBRARY}"
    )
endif()
