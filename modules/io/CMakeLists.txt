# build vineyard-io
file(GLOB IO_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "io/*.cc")

option(BUILD_VINEYARD_IO_KAFKA "Enable vineyard's IOAdaptor with KAFKA support" OFF)

if(BUILD_VINEYARD_IO_KAFKA)
    include("${PROJECT_SOURCE_DIR}/cmake/FindRdkafka.cmake")
endif()

# force build some thirdparty as static libraries, to make "install" easy
set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")

# restore
set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}" CACHE BOOL "Restore the default value" FORCE)

add_library(vineyard_io ${IO_SRC_FILES})
target_include_directories(vineyard_io PRIVATE ${MPI_CXX_INCLUDE_PATH}
)
target_link_libraries(vineyard_io PUBLIC vineyard_client
                                         vineyard_basic
                                         ${ARROW_SHARED_LIB}
)
target_link_libraries(vineyard_io PRIVATE ${GFLAGS_LIBRARIES})

target_include_directories(vineyard_io PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/libgrape-lite>
    $<INSTALL_INTERFACE:include>
)

if(RDKAFKA_FOUND)
    target_include_directories(vineyard_io PUBLIC ${RDKAFKA_INCLUDE_DIRS})
    target_compile_definitions(vineyard_io PRIVATE -DKAFKA_ENABLED)
    target_link_libraries(vineyard_io PUBLIC ${RDKAFKA_LIBRARIES})
endif()

install_vineyard_target(vineyard_io)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(vineyard_io_adaptors ALL)

# build and install c++ vineyard-io adaptors
file(GLOB CPP_IO_ADAPTORS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/adaptors"
                                   "${CMAKE_CURRENT_SOURCE_DIR}/adaptors/*.cc")
foreach(fname ${CPP_IO_ADAPTORS})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${fname})
    set(IO_BINARY_TOOL ${CMAKE_MATCH_1})
    message(STATUS "Found (cpp) io binary tool - " ${IO_BINARY_TOOL})
    add_executable(${IO_BINARY_TOOL} ${CMAKE_CURRENT_SOURCE_DIR}/adaptors/${IO_BINARY_TOOL}.cc)
    target_link_libraries(${IO_BINARY_TOOL} PRIVATE
                          vineyard_basic
                          vineyard_io
                          ${ARROW_SHARED_LIB}
                          ${CPPNETLIB_LIBRARIES}
                          ${MPI_CXX_LIBRARIES})
    set_target_properties(${IO_BINARY_TOOL} PROPERTIES OUTPUT_NAME "vineyard_${IO_BINARY_TOOL}")
    if(${LIBUNWIND_FOUND})
        target_link_libraries(${IO_BINARY_TOOL} PRIVATE ${LIBUNWIND_LIBRARIES})
    endif()
    install_vineyard_target(${IO_BINARY_TOOL})
    add_dependencies(vineyard_io_adaptors ${IO_BINARY_TOOL})
endforeach()

# install python vineyard-io adaptors
macro(install_python_target python_target rename)
    get_target_property(PYTHON_TARGET_LOCATION ${python_target} IMPORTED_LOCATION)
    install(PROGRAMS "${PYTHON_TARGET_LOCATION}"
            DESTINATION bin
            RENAME "${rename}"
    )
endmacro()

file(GLOB PYTHON_IO_ADAPTORS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/adaptors"
                                      "${CMAKE_CURRENT_SOURCE_DIR}/adaptors/*.py")
foreach(fname ${PYTHON_IO_ADAPTORS})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${fname})
    set(IO_BINARY_TOOL ${CMAKE_MATCH_1})
    message(STATUS "Found (python) io binary tool - " ${IO_BINARY_TOOL})

    add_executable(${IO_BINARY_TOOL} IMPORTED)
    set_target_properties(${IO_BINARY_TOOL} PROPERTIES
                      IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/adaptors/${IO_BINARY_TOOL}.py")
    install_python_target(${IO_BINARY_TOOL} "vineyard_${IO_BINARY_TOOL}")
    add_dependencies(vineyard_io_adaptors ${IO_BINARY_TOOL})
endforeach()

configure_file(setup.cfg.in "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg" @ONLY)
