cmake_minimum_required(VERSION 3.1)

project(vineyard-java)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(LINUX AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang compiler: ${CMAKE_CXX_COMPILER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fforce-emit-vtables")

    message(STATUS "Try to find ld.lld under same directory..")
    get_filename_component(COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    message(STATUS "Using compiler: ${COMPILER_DIR}")

    set(CMAKE_JNI_LINKER_FLAGS "-fuse-ld=${COMPILER_DIR}/ld.lld -Xlinker -mllvm=-lto-embed-bitcode")
else()
    message(STATUS "Using a non-clang compiler will lose performance evaluation provided by LLVM4JNI. Clang-11 is recommended")
    message(STATUS "Your compiler is ${CMAKE_CXX_COMPILER}")
endif()

find_program(ccache_EXECUTABLE ccache)
if(ccache_EXECUTABLE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${ccache_EXECUTABLE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${ccache_EXECUTABLE})
    add_custom_target(ccache-stats
        COMMAND ${ccache_EXECUTABLE} --show-stats
    )
else()
    add_custom_target(ccache-stats
        COMMAND echo "ccache not found."
    )
endif(ccache_EXECUTABLE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

file(GLOB SOURCES "target/generated-sources/annotations/*.cc"
                  "target/generated-test-sources/test-annotations/*.cc"
                  "src/main/native/*.cc"
)

set(LIBNAME "vineyard-java")

set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)

find_package(Threads)

include_directories(SYSTEM ${JAVA_INCLUDE_PATH})
include_directories(SYSTEM ${JAVA_INCLUDE_PATH2})
include_directories(${PROJECT_SOURCE_DIR}/src/main/native)

include_directories(${PROJECT_SOURCE_DIR}/../src)
include_directories(${PROJECT_SOURCE_DIR}/../modules)
include_directories(${PROJECT_SOURCE_DIR}/../thirdparty)

# /usr/local/opt/openssl@3/include;/usr/local/include;/usr/local/include;/usr/local/include;/Users/linzhu.ht/v6d/src;/Users/linzhu.ht/v6d/modules;/Users/linzhu.ht/v6d/thirdparty;/usr/local/opt/openssl@3/include

# find arrow
find_package(Arrow REQUIRED)

# find vineyard
find_package(vineyard)

add_library(${LIBNAME} SHARED ${SOURCES})

set_target_properties(${LIBNAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${LIBNAME} ${CMAKE_JNI_LINKER_FLAGS}
                                 arrow_shared
                                 vineyard_client
                                 vineyard_basic
)
target_compile_features(${LIBNAME} PRIVATE cxx_std_14)
