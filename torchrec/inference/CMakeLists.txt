# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.8)

project(inference C CXX)

include(/home/paulzhan/grpc/examples/cpp/cmake/common.cmake)


# Proto file
get_filename_component(hw_proto "/home/paulzhan/torchrec/torchrec/inference/protos/predictor.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/predictor.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/predictor.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/predictor.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/predictor.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

set(CMAKE_CXX_STANDARD 17)

# Torch + FBGEMM
find_package(Torch REQUIRED)
add_library( fbgemm SHARED IMPORTED GLOBAL )
set_target_properties(fbgemm PROPERTIES IMPORTED_LOCATION ${FBGEMM_LIB})


add_library(hw_grpc_proto STATIC
  ${hw_grpc_srcs}
  ${hw_grpc_hdrs}
  ${hw_proto_srcs}
  ${hw_proto_hdrs})
target_link_libraries(hw_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Targets greeter_[async_](client|server)
add_executable(server server.cpp)
target_link_libraries(server
  "${TORCH_LIBRARIES}"
  fbgemm
  hw_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)
