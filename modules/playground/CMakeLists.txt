set(SS)
list(APPEND SS "playground.cc")
add_executable(play ${SS} )
target_link_libraries(play vineyard_basic)
