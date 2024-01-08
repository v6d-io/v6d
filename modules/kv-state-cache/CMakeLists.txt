file(GLOB vineyard_kv_state_cache_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                "ds/*.cc"
                                "ds/*.h"
                                "radix-tree/*.cc"
                                "radix-tree/*.h"
                                "utils/*.cc"
                                "utils/*.h"
)

add_library(vineyard_kv_state_cache ${vineyard_kv_state_cache_SRCS})
target_link_libraries(vineyard_kv_state_cache PUBLIC vineyard_client vineyard_basic)
install_export_vineyard_target(vineyard_kv_state_cache)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/utils/")
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/ds/")
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/radix-tree/")
