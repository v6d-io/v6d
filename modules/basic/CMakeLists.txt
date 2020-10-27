# build vineyard-basic
file(GLOB_RECURSE VINEYARD_MOD_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                    "*.vineyard-mod")

if (VINEYARD_MOD_SRCS)
    vineyard_generate(
        OUT_VAR VINEYARD_GENERATES
        VINEYARD_MODULES ${VINEYARD_MOD_SRCS}
    )
    add_custom_target(vineyard_basic_gen
                      DEPENDS ${VINEYARD_GENERATES}
                      COMMENT "Running code generation for vineyard_basic.")
endif()

file(GLOB_RECURSE BASIC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_basic ${BASIC_SRC_FILES})
target_link_libraries(vineyard_basic vineyard_client
                                     ${ARROW_SHARED_LIB}
                                     ${MPI_CXX_LIBRARIES}
)

target_include_directories(vineyard_basic PUBLIC
                                          ${MPI_CXX_INCLUDE_PATH}
)
target_include_directories(vineyard_basic PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/libgrape-lite>
    $<INSTALL_INTERFACE:include>
)
if(VINEYARD_MOD_SRCS)
    add_dependencies(vineyard_basic vineyard_basic_gen)
endif()

install_vineyard_target(vineyard_basic)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
