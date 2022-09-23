# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function(add_tde_test NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} tde_cpp_objs gtest gtest_main)
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

add_tde_test(bits_op_test bits_op_test.cpp)
add_tde_test(move_only_function_test move_only_function_test.cpp)
