# build hosseinmoein-dataframe
file(GLOB DATAFRAME_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}"
                              "*.cc")

set(HMDF_TESTING OFF CACHE BOOL "Build tests of hosseinmoein/dataframe")
set(HMDF_EXAMPLES OFF CACHE BOOL "Build examples of hosseinmoein/dataframe")
set(HMDF_BENCHMARKS OFF CACHE BOOL "Build benchmarks of hosseinmoein/dataframe")
add_subdirectory_static(thirdparty/DataFrame EXCLUDE_FROM_ALL)

add_library(vineyard_hosseinmoein_dataframe ${DATAFRAME_SRC_FILES})
target_add_debuginfo(vineyard_hosseinmoein_dataframe)
set_property(TARGET vineyard_hosseinmoein_dataframe PROPERTY CXX_STANDARD 17)
target_link_libraries(vineyard_hosseinmoein_dataframe PUBLIC vineyard_client
                                                             vineyard_basic)
target_link_libraries(vineyard_hosseinmoein_dataframe PRIVATE DataFrame::DataFrame)
target_include_directories(vineyard_hosseinmoein_dataframe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/DataFrame/include>
    $<INSTALL_INTERFACE:include/vineyard/contrib>
)

# install bundled thirdparty: hosseinmoein/dataframe
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/DataFrame/include/DataFrame
        DESTINATION include/vineyard/contrib    # target directory
        FILES_MATCHING                          # install only matched files
        PATTERN "*.h"                           # select header files
        PATTERN "*.hpp"                         # select C++ template header files
)

install_export_vineyard_target(vineyard_hosseinmoein_dataframe)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
