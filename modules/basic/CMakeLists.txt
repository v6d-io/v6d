# build vineyard-basic
file(GLOB_RECURSE VINEYARD_MOD_SRCS "${CMAKE_CURRENT_SOURCE_DIR}"
                                    "ds/*.vineyard-mod"
                                    "stream/*.vineyard-mod")

file(GLOB_RECURSE VINEYARD_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}"
                                   "ds/*.h"
                                   "stream/*.h")

if(VINEYARD_MOD_SRCS)
    vineyard_generate(
        OUT_VAR VINEYARD_GENERATES
        VINEYARD_MODULES ${VINEYARD_MOD_SRCS}
    )
else()
    set(VINEYARD_GENERATES)
endif()

if(BUILD_VINEYARD_JAVA AND (VINEYARD_HEADERS OR VINEYARD_GENERATES))
    vineyard_generate_java(
        OUT_VAR VINEYARD_JAVA_GENERATES
        LANGUAGE java
        VINEYARD_MODULES ${VINEYARD_HEADERS} ${VINEYARD_GENERATES}
        DEPENDS ${VINEYARD_GENERATES}
    )
else()
    set(VINEYARD_JAVA_GENERATES)
endif()

add_custom_target(vineyard_basic_gen
                  DEPENDS ${VINEYARD_GENERATES}
                  COMMENT "Running code generation for vineyard_basic."
)

add_custom_target(vineyard_basic_gen_java
                  DEPENDS vineyard_basic_gen
                          ${VINEYARD_JAVA_GENERATES}
                  COMMENT "Running java code generation for vineyard_basic."
)

file(GLOB_RECURSE BASIC_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc")

add_library(vineyard_basic ${BASIC_SRC_FILES})
target_link_libraries(vineyard_basic PUBLIC vineyard_client
                                            ${ARROW_SHARED_LIB}
                                            ${GLOG_LIBRARIES}
)

add_dependencies(vineyard_basic vineyard_basic_gen)

add_dependencies(vineyard_codegen vineyard_basic_gen)
if(VINEYARD_JAVA_GENERATES)
    add_dependencies(vineyard_codegen_java vineyard_basic_gen_java)
endif()

install_vineyard_target(vineyard_basic)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
