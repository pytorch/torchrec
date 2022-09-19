cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

project(TorchRec
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" OFF)

add_subdirectory(torchrec/csrc)

if (BUILD_TEST)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.12.0
    )
    FetchContent_Declare(google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.5.6
    )

    # We will not need to test benchmark lib itself.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
    # We will not need to install benchmark since we link it statically.
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
    FetchContent_MakeAvailable(googletest google_benchmark)

    enable_testing()
    add_subdirectory(test/cpp)
endif()
