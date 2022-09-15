function(add_tde_test NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} tde_cpp_objs gtest gtest_main)
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

add_tde_test(bits_op_test bits_op_test.cpp)
