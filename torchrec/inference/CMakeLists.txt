# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(inference)

# This step is crucial to ensure that the
# _REFLECTION, _GRPC_GRPCPP and _PROTOBUF_LIBPROTOBUF variables are set.
# e.g. ~/gprc/examples/cpp/cmake/common.cmake
include(${GRPC_COMMON_CMAKE_PATH}/common.cmake)

if(NOT DEFINED TORCH_DEPLOY_LIB_PATH)
  message(
    FATAL_ERROR
    "TORCH_DEPLOY_LIB_PATH must be set to the path of the torch deploy library."
  )
endif()

set(CMAKE_C_FLAGS "-Wl,-export-dynamic")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-export-dynamic")

# dependencies
find_package(Torch REQUIRED)
find_package(folly REQUIRED)
find_package(gflags REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
include_directories(${folly_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)

# inference library

# for our own header files
include_directories(include/)

# define our library target
add_library(inference SHARED
  src/Batching.cpp
  src/BatchingQueue.cpp
  src/GPUExecutor.cpp
  src/ResultSplit.cpp
)

# whole archive is needed to link against the static library
target_link_libraries(inference "-Wl,--whole-archive" ${TORCH_DEPLOY_LIB_PATH})
target_link_libraries(
  inference
  "-Wl,--no-whole-archive"
  "${TORCH_LIBRARIES}"
  ${FOLLY_LIBRARIES}
)

# for generated protobuf

# grpc headers. e.g. ~/.local/include
include_directories(${GRPC_HEADER_INCLUDE_PATH})

set(pred_grpc_srcs "include/torchrec/inference/predictor.grpc.pb.cc")
set(pred_grpc_hdrs "include/torchrec/inference/predictor.grpc.pb.h")
set(pred_proto_srcs "include/torchrec/inference/predictor.pb.cc")
set(pred_proto_hdrs "include/torchrec/inference/predictor.pb.h")

add_library(pred_grpc_proto SHARED
  ${pred_grpc_srcs}
  ${pred_grpc_hdrs}
  ${pred_proto_srcs}
  ${pred_proto_hdrs})

target_link_libraries(pred_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# server

add_executable(server server.cpp)
target_link_libraries(inference "-Wl,--whole-archive" ${TORCH_DEPLOY_LIB_PATH})
target_link_libraries(server "-Wl,--no-whole-archive"
  inference
  pred_grpc_proto
  "${TORCH_LIBRARIES}"
  ${FOLLY_LIBRARIES}
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})
