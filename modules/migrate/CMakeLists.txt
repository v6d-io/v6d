# build vineyard-migrate
add_library(vineyard_migrate "object_migration.cc" "flags.cc" "protocols.cc")
target_include_directories(vineyard_migrate PUBLIC
                                            ${MPI_CXX_INCLUDE_PATH}
)
target_link_libraries(vineyard_migrate vineyard_client
                                       vineyard_basic
                                       ${ARROW_SHARED_LIB}
                                       ${GFLAGS_LIBRARIES}
                                       ${MPI_CXX_LIBRARIES}
)

install_vineyard_target(vineyard_migrate)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(vineyard_copy "vineyard_copy.cc"
)
target_link_libraries(vineyard_copy vineyard_migrate ${Boost_LIBRARIES}
                                                     ${GFLAGS_LIBRARIES}
)
install_vineyard_target(vineyard_copy)
