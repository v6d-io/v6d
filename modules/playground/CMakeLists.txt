set(SS)
list(APPEND SS "playground.cc")
add_executable(register "register.cc")
add_executable(play ${SS} )
target_link_libraries(play vineyard_basic)
target_link_libraries(register vineyard_basic)
target_include_directories(play PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty/ctti/include>
$<INSTALL_INTERFACE:include>
)
target_link_libraries(play ${CMAKE_DL_LIBS})