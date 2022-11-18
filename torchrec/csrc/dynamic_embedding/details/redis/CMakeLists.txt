# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FetchContent_Declare(
        hiredis
        GIT_REPOSITORY https://github.com/redis/hiredis.git
        GIT_TAG 06be7ff312a78f69237e5963cc7d24bc84104d3b
)

FetchContent_GetProperties(hiredis)
if(NOT hiredis_POPULATED)
    # Do not include hiredis in install targets
    FetchContent_Populate(hiredis)
    set(DISABLE_TESTS ON CACHE BOOL "Disable tests for hiredis")
    add_subdirectory(
        ${hiredis_SOURCE_DIR} ${hiredis_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(redis_io SHARED redis_io.cpp)
target_include_directories(
    redis_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../)
target_include_directories(redis_io PUBLIC ${TORCH_INCLUDE_DIRS})
target_compile_options(redis_io PUBLIC -fPIC)
target_link_libraries(redis_io PUBLIC hiredis::hiredis_static)
