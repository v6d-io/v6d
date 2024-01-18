set(LZ4_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lz4")

add_custom_target(build_lz4
                  COMMAND make -C ${LZ4_SOURCE_DIR}
                  WORKING_DIRECTORY ${LZ4_SOURCE_DIR})

file(GLOB LZ4_LIBRARIES "${LZ4_SOURCE_DIR}/lib/*.so")

file(GLOB VINEYARD_KV_STATE_CACHE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                "ds/*.cc"
                                "ds/*.h"
                                "radix-tree/*.cc"
                                "radix-tree/*.h"
                                "utils/*.cc"
                                "utils/*.h"
                                "strategy/*.cc"
                                "strategy/*.h"
                                "lz4/lib/*.h"
)

add_library(vineyard_kv_state_cache ${VINEYARD_KV_STATE_CACHE_SRCS})
target_link_libraries(vineyard_kv_state_cache PUBLIC vineyard_client vineyard_basic ${LZ4_LIBRARIES})

install_export_vineyard_target(vineyard_kv_state_cache)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/utils/")
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/ds/")
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/radix-tree/")
