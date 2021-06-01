# build vineyard-object-migration
add_library(vineyard_object_migration "object_migration.cc" "flags.cc" "protocols.cc")
target_include_directories(vineyard_object_migration PUBLIC)
target_link_libraries(vineyard_object_migration vineyard_client
                                                ${ARROW_SHARED_LIB}
                                                ${GFLAGS_LIBRARIES}
)

install_vineyard_target(vineyard_object_migration)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

# build vineyard-copy
add_executable(vineyard-copy "vineyard_copy.cc"
)
target_link_libraries(vineyard-copy vineyard_object_migration
                                    ${Boost_LIBRARIES}
                                    ${GFLAGS_LIBRARIES}
)
install_vineyard_target(vineyard-copy)

# build vineyard-migrate
add_executable(vineyard-migrate "vineyard_migrate.cc")
target_link_libraries(vineyard-migrate vineyard_client
                                       ${ARROW_SHARED_LIB}
                                       ${Boost_LIBRARIES}
                                       ${GFLAGS_LIBRARIES}
)
install_vineyard_target(vineyard-migrate)

# build vineyard-migrate-stream
add_executable(vineyard-migrate-stream "vineyard_migrate_stream.cc")
target_link_libraries(vineyard-migrate-stream vineyard_client
                                              ${ARROW_SHARED_LIB}
                                              ${Boost_LIBRARIES}
                                              ${GFLAGS_LIBRARIES}
)
install_vineyard_target(vineyard-migrate-stream)

# install vineyard-migrate-to-local script.
install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/vineyard-migrate-to-local.py"
        DESTINATION bin
        RENAME vineyard-migrate-to-local
)

