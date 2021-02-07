# This file is used to find librdkafka library in CMake script, modifeid from the
# code from
#
#   https://github.com/BVLC/caffe/blob/master/cmake/Modules/FindGlog.cmake
#
# which is licensed under the 2-Clause BSD License.
#
# - Try to find librdkafka
#
# The following variables are optionally searched for defaults
#  Rdkafka_ROOT_DIR:            Base directory where all rdkafka components are found
#
# The following are set after configuration is done:
#  Rdkafka_FOUND
#  Rdkafka_INCLUDE_DIRS
#  Rdkafka_LIBRARIES
#  Rdkafka_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(Rdkafka_ROOT_DIR "" CACHE PATH "Folder contains librdkafka")

# We are testing only a couple of files in the include directories
find_path(Rdkafka_INCLUDE_DIR librdkafka PATHS ${Rdkafka_ROOT_DIR}/include)

find_library(Rdkafka_LIBRARY rdkafka PATHS  ${Rdkafka_ROOT_DIR}/lib)
find_library(Rdkafka++_LIBRARY rdkafka++ PATHS  ${Rdkafka_ROOT_DIR}/lib)

find_package_handle_standard_args(Rdkafka DEFAULT_MSG Rdkafka_INCLUDE_DIR Rdkafka_LIBRARY)


if(Rdkafka_FOUND)
    set(Rdkafka_INCLUDE_DIRS ${Rdkafka_INCLUDE_DIR})
    # The Rdkafka_LIBRARY comes later, since it is depended by the former.
    set(Rdkafka_LIBRARIES ${Rdkafka++_LIBRARY} ${Rdkafka_LIBRARY})
    message(STATUS "Found rdkafka (include: ${Rdkafka_INCLUDE_DIRS}, library: ${Rdkafka_LIBRARIES})")
    mark_as_advanced(Rdkafka_LIBRARY_DEBUG Rdkafka_LIBRARY_RELEASE
                     Rdkafka_LIBRARY Rdkafka_INCLUDE_DIR Rdkafka_ROOT_DIR)
endif()
