file(GLOB RDMA_SRC *.cpp)

add_library(rdma ${RDMA_SRC})

target_link_libraries(rdma ibverbs pthread vineyard_client vineyard_basic)
