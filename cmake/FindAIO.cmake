# FindAIO.cmake

# Try to find the AIO library
# Define the following cached variables:
#  AIO_FOUND - Was AIO found?
#  AIO_INCLUDE_DIRS - Where to find the AIO includes
#  AIO_LIBRARIES - The libraries needed to use AIO

set(AIO_HOME $ENV{AIO_HOME})

find_path (AIO_INCLUDE_DIRS
    NAMES libaio.h
    PATHS ${AIO_HOME}/include /usr/local/include /usr/include
)

find_library (AIO_LIBRARIES
    NAMES aio
    PATHS ${AIO_HOME}/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(AIO DEFAULT_MSG
AIO_INCLUDE_DIRS AIO_LIBRARIES)

if (AIO_FOUND)
    add_library(AIO::aio SHARED IMPORTED)
    set_target_properties(AIO::aio PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${AIO_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${AIO_LIBRARIES}"
        INTERFACE_COMPILE_DEFINITIONS "CMAKE_INCLUDE"
    )
endif()

mark_as_advanced(AIO_INCLUDE_DIRS AIO_LIBRARIES)
