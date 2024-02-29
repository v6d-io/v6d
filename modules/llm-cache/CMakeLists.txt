file(GLOB VINEYARD_LLM_CACHE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                  "ds/*.cc"
                                  "ds/*.h"
                                  "radix-tree/*.cc"
                                  "radix-tree/*.h"
                                  "${PROJECT_SOURCE_DIR}/thirdparty/rax/*.cc"
                                  "${PROJECT_SOURCE_DIR}/thirdparty/rax/*.h"
)

add_library(vineyard_llm_cache ${VINEYARD_LLM_CACHE_SRCS})
target_link_libraries(vineyard_llm_cache PUBLIC vineyard_client vineyard_basic)

install_export_vineyard_target(vineyard_llm_cache)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/ds/")
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}/radix-tree/")

add_subdirectory(tests)
