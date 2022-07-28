set(ADAPTOR_TEST_SRC)
# list(APPEND FUSE_SRC_FILES "arrow_ipc/arrow_ipc.cc")
# list(APPEND FUSE_SRC_FILES "arrow_ipc/arrow_ipc.h")

list(APPEND ADAPTOR_TEST_SRC "adaptor_test.cc")

list(APPEND ADAPTOR_TEST_SRC "factory/deserializer_registry.h")
# list(APPEND ADAPTOR_TEST_SRC "factory/array_adaptors.h")

add_executable(fuse_test ${ADAPTOR_TEST_SRC})
target_include_directories(fuse_test PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/ctti/include>
$<INSTALL_INTERFACE:include>
)
target_link_libraries(fuse_test PUBLIC vineyard_client
                                           vineyard_basic
                                           ${ARROW_SHARED_LIB}
                                           ${CMAKE_DL_LIBS}
)
# target_link_libraries(fuse_test ${CMAKE_DL_LIBS})