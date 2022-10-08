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
add_tde_test(naive_id_transformer_test naive_id_transformer_test.cpp)
add_tde_test(random_bits_generator_test random_bits_generator_test.cpp)
add_tde_test(mixed_lfu_lru_strategy_test mixed_lfu_lru_strategy_test.cpp)
