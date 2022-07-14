set(CMAKE_CXX_STANDARD 17)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/DataFrame/include/
)

add_subdirectory_static(thirdparty/DataFrame)

add_library(hosseinmoein_dataframe STATIC hosseinmoein_dataframe.cc hosseinmoein_dataframe.h)
target_link_libraries(hosseinmoein_dataframe PUBLIC DataFrame::DataFrame vineyard_client vineyard_basic)

install_vineyard_target(hosseinmoein_dataframe)
install_vineyard_headers("${CMAKE_CURRENT_SOURCE_DIR}")
