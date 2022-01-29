# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(inference)

# dependencies
find_package(Torch REQUIRED)
find_package(Boost REQUIRED)
find_package(folly REQUIRED)
find_package(Threads REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${folly_INCLUDE_DIRS})

# for our own header files
include_directories(.)

# define our library target
add_library(inference SHARED Batching.cpp BatchingQueue.cpp GPUExecutor.cpp)

# c++ 17
target_compile_features(inference PRIVATE cxx_std_17)

# link against LibTorch
target_link_libraries(inference "${TORCH_LIBRARIES}")

# link folly and boost
target_link_libraries(inference ${Boost_LIBRARIES} ${FOLLY_LIBRARIES})
