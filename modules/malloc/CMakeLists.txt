# build vineyard-allocator

file(GLOB_RECURSE BASIC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_malloc ${BASIC_SRC_FILES})
target_link_libraries(vineyard_malloc PUBLIC vineyard_client
                                             ${ARROW_SHARED_LIB}
)

if(WITH_ALLOCATOR STREQUAL "jemalloc")
    target_link_libraries(vineyard_malloc PRIVATE jemalloc ${CMAKE_DL_LIBS})
    target_compile_options(vineyard_malloc PRIVATE -DWITH_JEMALLOC)
endif()

install_vineyard_target(vineyard_malloc)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
