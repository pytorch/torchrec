# Copyright(c) Meta Platforms, Inc.and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD - style license found in the
# LICENSE file in the root directory of this source tree.

function(add_tde_benchmark NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} tde_cpp_objs benchmark::benchmark_main benchmark::benchmark)
endfunction()

add_tde_benchmark(naive_id_transformer_benchmark naive_id_transformer_benchmark.cpp)
