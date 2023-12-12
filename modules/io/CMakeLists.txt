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
target_add_debuginfo(vineyard_io)
target_include_directories(vineyard_io PRIVATE ${MPI_CXX_INCLUDE_PATH}
)
target_link_libraries(vineyard_io PUBLIC vineyard_client
                                         vineyard_basic
                                         ${ARROW_SHARED_LIB}
                                         ${CMAKE_DL_LIBS}
)
target_link_libraries(vineyard_io PRIVATE ${GFLAGS_LIBRARIES})

if(Rdkafka_FOUND)
    target_include_directories(vineyard_io PUBLIC ${Rdkafka_INCLUDE_DIRS})
    target_compile_definitions(vineyard_io PRIVATE -DKAFKA_ENABLED)
    target_link_libraries(vineyard_io PUBLIC ${Rdkafka_LIBRARIES})
endif()

install_export_vineyard_target(vineyard_io)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/io" "include/vineyard/io")
