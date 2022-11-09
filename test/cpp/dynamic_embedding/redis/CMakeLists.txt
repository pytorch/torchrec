# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function(add_redis_test NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} redis_io tde_cpp_objs gtest gtest_main)
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

# TODO: Need start a empty redis-server on 127.0.0.1:6379 before run *redis*_test.
add_redis_test(redis_io_test redis_io_test.cpp)
add_redis_test(url_test url_test.cpp)
