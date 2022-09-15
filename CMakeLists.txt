cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

project(TorchRec
        VERSION 0.2.0
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" OFF)

add_subdirectory(torchrec/csrc)
add_subdirectory(third_party)

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(test/cpp)
endif()
