# build vineyard-allocator

file(GLOB_RECURSE BASIC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_malloc ${BASIC_SRC_FILES}
                            # requires mimalloc
                            $<TARGET_OBJECTS:mimalloc-static>)
target_add_debuginfo(vineyard_malloc)
target_link_libraries(vineyard_malloc PUBLIC vineyard_client)
target_link_libraries(vineyard_malloc PRIVATE $<TARGET_PROPERTY:mimalloc-static,INTERFACE_LINK_LIBRARIES>)

if(BUILD_VINEYARD_MALLOC_OVERRIDE)
    # TODO(not implemented yet)
    target_compile_options(vineyard_malloc PRIVATE -DVINEYARD_MALLOC_OVERRIDE)
endif()

install_export_vineyard_target(vineyard_malloc)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
