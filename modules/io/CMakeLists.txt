# build vineyard-io
file(GLOB IO_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "io/*.cc")

option(BUILD_VINEYARD_IO_OSS "Enable vineyard's IOAdaptor with OSS support" OFF)
option(BUILD_VINEYARD_IO_KAFKA "Enable vineyard's IOAdaptor with KAFKA support" OFF)

if(BUILD_VINEYARD_IO_OSS)
    find_package(CURL)
endif()
if(BUILD_VINEYARD_IO_KAFKA)
    include(FindRdkafka)
endif()

# force build some thirdparty as static libraries, to make "install" easy
set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build network-uri as static library" FORCE)

set(Uri_BUILD_DOCS OFF CACHE BOOL "Disable network-uri docs.")
set(Uri_BUILD_TESTS OFF CACHE BOOL "Disable network-uri tests.")
add_subdirectory("${PROJECT_SOURCE_DIR}/thirdparty/uri"
                 "${PROJECT_BINARY_DIR}/thirdparty/uri"
)
set(CPPNETLIB_LIBRARIES network-uri)
set(CPPNETLIB_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/thirdparty/uri/include")

if(BUILD_VINEYARD_IO_OSS)
    set(BUILD_SAMPLE OFF CACHE BOOL "Build aliyun-oss-cpp-sdk samples" FORCE)
    add_subdirectory("${PROJECT_SOURCE_DIR}/thirdparty/aliyun-oss-cpp-sdk"
                     "${PROJECT_BINARY_DIR}/thirdparty/aliyun-oss-cpp-sdk"
    )
    set(OSS_LIBRARIES cpp-sdk)
endif()

# restore
set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}" CACHE BOOL "Restore the default value" FORCE)

add_library(vineyard_io ${IO_SRC_FILES})
target_include_directories(vineyard_io PRIVATE ${CPPNETLIB_INCLUDE_DIRS}
                                               ${MPI_CXX_INCLUDE_PATH}
)
target_link_libraries(vineyard_io PUBLIC vineyard_client
                                         vineyard_basic
                                         ${ARROW_SHARED_LIB}
)
target_link_libraries(vineyard_io PRIVATE ${CPPNETLIB_LIBRARIES})
target_link_libraries(vineyard_io PRIVATE ${GFLAGS_LIBRARIES})

target_include_directories(vineyard_io PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/libgrape-lite>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_VINEYARD_IO_OSS)
    target_include_directories(vineyard_io PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/aliyun-oss-cpp-sdk/sdk/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(vineyard_io PRIVATE -DOSS_ENABLED)
    target_link_libraries(vineyard_io PRIVATE ${OSS_LIBRARIES}
                                              ${CURL_LIBRARIES}
                                              ${OPENSSL_LIBRARIES}
    )
endif()

if(RDKAFKA_FOUND)
    target_include_directories(vineyard_io PUBLIC ${RDKAFKA_INCLUDE_DIRS})
    target_compile_definitions(vineyard_io PRIVATE -DKAFKA_ENABLED)
    target_link_libraries(vineyard_io PUBLIC ${RDKAFKA_LIBRARIES})
endif()

install_vineyard_target(vineyard_io)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

# build vineyard-io adaptors
file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/adaptors" "${CMAKE_CURRENT_SOURCE_DIR}/adaptors/*.cc")
add_custom_target(vineyard_io_adaptors ALL)
foreach(fname ${TEST_FILES})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${fname})
    set(IO_BINARY_TOOL ${CMAKE_MATCH_1})
    message(STATUS "Found io binary tool - " ${IO_BINARY_TOOL})
    add_executable(${IO_BINARY_TOOL} ${CMAKE_CURRENT_SOURCE_DIR}/adaptors/${IO_BINARY_TOOL}.cc)
    target_link_libraries(${IO_BINARY_TOOL}
                          vineyard_basic
                          vineyard_io
                          ${ARROW_SHARED_LIB}
                          ${CPPNETLIB_LIBRARIES}
                          ${MPI_CXX_LIBRARIES})
    set_target_properties(${IO_BINARY_TOOL} PROPERTIES OUTPUT_NAME "vineyard_${IO_BINARY_TOOL}")
    install_vineyard_target(${IO_BINARY_TOOL})
    add_dependencies(vineyard_io_adaptors ${IO_BINARY_TOOL})
endforeach()
