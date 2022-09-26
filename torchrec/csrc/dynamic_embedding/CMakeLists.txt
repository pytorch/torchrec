# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(tde_cpp_objs
            OBJECT
            details/clz_impl.cpp
            details/ctz_impl.cpp)

target_include_directories(tde_cpp_objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
target_include_directories(tde_cpp_objs PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(tde_cpp_objs PUBLIC ${TORCH_LIBRARIES})
target_compile_options(tde_cpp_objs PUBLIC -fPIC)
