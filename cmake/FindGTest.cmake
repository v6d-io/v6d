# FindGTest.cmake

# Try to find the GTest library
# Define the following cached variables:
#  GTest_FOUND - Was GTest found?
#  GTest_INCLUDE_DIRS - Where to find the GTest includes
#  GTest_LIBRARIES - The libraries needed to use GTest

set(GTEST_HOME $ENV{GTEST_HOME})

find_path (GTEST_INCLUDE_DIRS
    NAMES gtest/gtest.h
    PATHS ${GTEST_HOME}/include /usr/local/include /usr/include
)

find_library (GTEST_LIBRARIES
    NAMES gtest gtest_main
    PATHS ${GTEST_HOME}/lib /usr/local/lib /usr/lib
)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTEST DEFAULT_MSG
GTEST_INCLUDE_DIRS GTEST_LIBRARIES)

message("GTest include dirs: ${GTEST_INCLUDE_DIRS} GTest libraries: ${GTEST_LIBRARIES}")

if (GTEST_FOUND)
    add_library(GTEST::gtest SHARED IMPORTED)
    set_target_properties(GTEST::gtest PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${GTEST_LIBRARIES}"
        INTERFACE_COMPILE_DEFINITIONS "CMAKE_INCLUDE"
    )
else()
    message(WARNING "GTest not found.")
    set(GTEST_INCLUDE_DIRS "")
    set(GTEST_LIBRARIES "")
endif()

mark_as_advanced(GTEST_INCLUDE_DIRS GTEST_LIBRARIES)
