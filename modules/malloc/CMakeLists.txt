# build vineyard-allocator

file(GLOB_RECURSE BASIC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_malloc ${BASIC_SRC_FILES})
target_link_libraries(vineyard_malloc PUBLIC vineyard_client)

# requires mimalloc
if(APPLE)
    target_link_libraries(vineyard_malloc PRIVATE -Wl,-all_load mimalloc-static)
else()
    target_link_libraries(vineyard_malloc PRIVATE -Wl,--whole-archive mimalloc-static -Wl,--no-whole-archive)
endif()

if(BUILD_VINEYARD_MALLOC_OVERRIDE)
    # TODO(not implemented yet)
    target_compile_options(vineyard_malloc PRIVATE -DVINEYARD_MALLOC_OVERRIDE)
endif()

install_vineyard_target(vineyard_malloc)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
