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


# abi and other flags

if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
  endif()
endif()

# keep it static for now since folly-shared version is broken
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# dependencies
find_package(Boost REQUIRED)
find_package(Torch REQUIRED)
find_package(folly REQUIRED)
find_package(gflags REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
include_directories(${folly_INCLUDE_DIRS})
include_directories(${PYTORCH_FMT_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 17)

# torch deploy library
add_library(torch_deploy_internal STATIC
    ${DEPLOY_INTERPRETER_PATH}/libtorch_deployinterpreter.o
    ${DEPLOY_SRC_PATH}/deploy.cpp
    ${DEPLOY_SRC_PATH}/loader.cpp
    ${DEPLOY_SRC_PATH}/path_environment.cpp
    ${DEPLOY_SRC_PATH}/elf_file.cpp)

# For python builtins. caffe2_interface_library properly
# makes use of the --whole-archive option.
target_link_libraries(torch_deploy_internal PRIVATE
  crypt pthread dl util m z ffi lzma readline nsl ncursesw panelw
)
target_link_libraries(torch_deploy_internal
  PUBLIC shm torch ${PYTORCH_LIB_FMT}
)
caffe2_interface_library(torch_deploy_internal torch_deploy)

# inference library

# for our own header files
include_directories(include/)
include_directories(gen/)

# define our library target
add_library(inference STATIC
  src/Batching.cpp
  src/BatchingQueue.cpp
  src/GPUExecutor.cpp
  src/ResultSplit.cpp
  src/Exception.cpp
  src/ResourceManager.cpp
)

# -rdynamic is needed to link against the static library
target_link_libraries(inference "-Wl,--no-as-needed -rdynamic"
  dl torch_deploy "${TORCH_LIBRARIES}" ${FBGEMM_LIB} ${FOLLY_LIBRARIES}
)

# for generated protobuf

# grpc headers. e.g. ~/.local/include
include_directories(${GRPC_HEADER_INCLUDE_PATH})

set(pred_grpc_srcs "gen/torchrec/inference/predictor.grpc.pb.cc")
set(pred_grpc_hdrs "gen/torchrec/inference/predictor.grpc.pb.h")
set(pred_proto_srcs "gen/torchrec/inference/predictor.pb.cc")
set(pred_proto_hdrs "gen/torchrec/inference/predictor.pb.h")

add_library(pred_grpc_proto STATIC
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
target_link_libraries(server
  inference
  torch_deploy
  pred_grpc_proto
  "${TORCH_LIBRARIES}"
  ${FOLLY_LIBRARIES}
  ${PYTORCH_LIB_FMT}
  ${FBGEMM_LIB}
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})
