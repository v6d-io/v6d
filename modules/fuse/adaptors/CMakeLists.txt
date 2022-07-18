set(ADAPTOR_TEST_SRC)
# list(APPEND FUSE_SRC_FILES "arrow_ipc/arrow_ipc.cc")
# list(APPEND FUSE_SRC_FILES "arrow_ipc/arrow_ipc.h")
list(APPEND ADAPTOR_TEST_SRC "factory/adaptor_factory.cc")
list(APPEND ADAPTOR_TEST_SRC "factory/adaptor_factory.h")
list(APPEND ADAPTOR_TEST_SRC "factory/detail/fuse_registry.cc")
list(APPEND ADAPTOR_TEST_SRC "adaptor_test.cc")
list(APPEND ADAPTOR_TEST_SRC "factory/mock.cc")
list(APPEND ADAPTOR_TEST_SRC "factory/mock.h")

list(APPEND ADAPTOR_TEST_SRC "factory/i_adaptor.h")
add_executable(fuse_test ${ADAPTOR_TEST_SRC})
target_include_directories(fuse_test PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/ctti/include>
$<INSTALL_INTERFACE:include>
)
target_link_libraries(fuse_test ${CMAKE_DL_LIBS})