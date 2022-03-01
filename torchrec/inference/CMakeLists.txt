# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(inference)

if(NOT DEFINED TORCH_DEPLOY_LIB_PATH)
    message(
      FATAL_ERROR
      "You must set TORCH_DEPLOY_LIB_PATH to the path of the torch deploy \
      library."
    )
endif()

# dependencies
find_package(Torch REQUIRED)
find_package(folly REQUIRED)
find_package(gflags REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
include_directories(${folly_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)

# for our own header files
include_directories(include/)

# define our library target
add_library(inference SHARED
  src/Batching.cpp
  src/BatchingQueue.cpp
  src/GPUExecutor.cpp)

target_link_libraries(inference
  ${TORCH_DEPLOY_LIB_PATH}
  "${TORCH_LIBRARIES}"
  ${FOLLY_LIBRARIES})

add_executable(example example.cpp)
target_link_libraries(example inference)
