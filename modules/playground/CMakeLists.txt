set(SS)
list(APPEND SS "playground.cc")
add_executable(playy ${SS} )
target_link_libraries(playy vineyard_basic)