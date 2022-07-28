# build vineyard-fuse
set(FUSE_SRC_FILES)

list(APPEND FUSE_SRC_FILES "fuse_impl.cc")

add_library(vineyard_fuse ${FUSE_SRC_FILES})
target_link_libraries(vineyard_fuse PUBLIC vineyard_client
                                           vineyard_basic
                                           ${ARROW_SHARED_LIB}
)
set_target_properties(vineyard_fuse PROPERTIES    CXX_STANDARD 14)
target_link_libraries(vineyard_fuse PUBLIC FUSE3::FUSE3)
target_compile_options(vineyard_fuse PUBLIC -DWITH_ARROW_IPC)
if(BUILD_VINEYARD_FUSE_PARQUET)
    target_compile_options(vineyard_fuse PUBLIC -DWITH_PARQUET)
    if(TARGET parquet_shared)
        target_link_libraries(vineyard_fuse PUBLIC parquet_shared)
    elseif(TARGET parquet_static)
        target_link_libraries(vineyard_fuse PUBLIC parquet_static)
    endif()
endif()
target_include_directories(vineyard_fuse PRIVATE ${CMAKE_SOURCE_DIR})

install_vineyard_target(vineyard_fuse)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")

set(FUSE_MOUNT_SRC_FILES)

# list(APPEND FUSE_MOUNT_SRC_FILES "adaptors/arrow_ipc/deserializer_registry.h")

# message(STATUS "Cmake source dir is at " ${CMAKE_SOURCE_DIR})
add_executable(vineyard-fusermount  fusermount.cc)
target_include_directories(vineyard-fusermount PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(vineyard-fusermount PUBLIC vineyard_fuse)
install_vineyard_target(vineyard-fusermount)

# if(BUILD_VINEYARD_TESTS)
#     enable_testing()
#     file(GLOB TEST_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/test" "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cc")
#     foreach(f ${TEST_FILES})
#         string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${f})
#         set(T_NAME ${CMAKE_MATCH_1})
#         message(STATUS "Found unit_test - " ${T_NAME})
#         if(BUILD_VINEYARD_TESTS_ALL)
#             add_executable(${T_NAME} test/${T_NAME}.cc)
#         else()
#             add_executable(${T_NAME} EXCLUDE_FROM_ALL test/${T_NAME}.cc)
#         endif()
#         target_link_libraries(${T_NAME} PRIVATE
#                               vineyard_fuse
#                               ${ARROW_SHARED_LIB}
#                               ${MPI_CXX_LIBRARIES})
#         if(${LIBUNWIND_FOUND})
#             target_link_libraries(${T_NAME} PRIVATE ${LIBUNWIND_LIBRARIES})
#         endif()
#         add_test(${T_NAME}, ${T_NAME})
#         add_dependencies(vineyard_tests ${T_NAME})
#     endforeach()
# endif()
