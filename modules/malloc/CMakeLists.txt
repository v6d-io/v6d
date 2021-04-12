# build vineyard-allocator

file(GLOB_RECURSE BASIC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_malloc ${BASIC_SRC_FILES})
target_link_libraries(vineyard_malloc vineyard_client
                                      ${ARROW_SHARED_LIB}
)

if(VINEYARD_MOD_SRCS)
    add_dependencies(vineyard_malloc vineyard_malloc)
endif()

install_vineyard_target(vineyard_malloc)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
