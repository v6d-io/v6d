# This file is used to find etcd-cpp-apiv3 library in CMake script, based on code
# from
#
#   https://github.com/BVLC/caffe/blob/master/cmake/Modules/FindGFlags.cmake
#
# which is licensed under the 3-Clause BSD License.
#
# - Try to find the path of etcd-cpp-apiv3
#   (https://github.com/7br/etcd-cpp-apiv3.git)
#
# The following variables are optionally searched for defaults
#  ETCD_CPP_ROOT_DIR:            Base directory where all etcd-cpp-apiv3 components are found
#
# The following are set after configuration is done:
#  ETCD_CPP_FOUND
#  ETCD_CPP_INCLUDE_DIRS
#  ETCD_CPP_LIBRARIES
#  ETCD_CPP_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(ETCD_CPP_ROOT_DIR "" CACHE PATH "Folder contains aliyun etcd-cpp-apiv3")

find_path(ETCD_CPP_INCLUDE_DIR etcd PATHS ${ETCD_CPP_ROOT_DIR}/include)

find_library(ETCD_CPP_LIBRARY etcd-cpp-api PATHS ${ETCD_CPP_ROOT_DIR}/lib)

find_package_handle_standard_args(ETCD_CPP DEFAULT_MSG ETCD_CPP_INCLUDE_DIR ETCD_CPP_LIBRARY)

if(ETCD_CPP_FOUND)
    set(ETCD_CPP_INCLUDE_DIRS ${ETCD_CPP_INCLUDE_DIR} ${ETCD_CPP_INCLUDE_DIR}/proto)
    set(ETCD_CPP_LIBRARIES ${ETCD_CPP_LIBRARY})
    message(STATUS "Found ETCD_CPP (include: ${ETCD_CPP_INCLUDE_DIRS}, library: ${ETCD_CPP_LIBRARIES})")
    mark_as_advanced(ETCD_CPP_LIBRARY_DEBUG ETCD_CPP_LIBRARY_RELEASE
            ETCD_CPP_LIBRARY ETCD_CPP_INCLUDE_DIR ETCD_CPP_ROOT_DIR)
endif()
