include(FetchContent)

if (TDE_WITH_TESTING)
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
endif()

# tcb::span section. A std::span implementation in C++ 11.
FetchContent_Declare(
        tcb_span
        GIT_TAG 836dc6a0efd9849cb194e88e4aa2387436bb079b
        GIT_REPOSITORY https://github.com/tcbrindle/span.git
)

# Do not MakeAvailable so do not enable tcb span unittest
FetchContent_Populate(tcb_span)

add_library(tcb_span INTERFACE)
target_include_directories(tcb_span INTERFACE ${tcb_span_SOURCE_DIR}/include)
add_library(tcb::span ALIAS tcb_span)

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.10.5
)

FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
        hiredis
        GIT_REPOSITORY https://github.com/redis/hiredis.git
        GIT_TAG v1.0.2
)

FetchContent_GetProperties(hiredis)
if(NOT hiredis_POPULATED)
    # Do not include hiredis in install targets
    FetchContent_Populate(hiredis)
    set(DISABLE_TESTS OFF CACHE BOOL "Disable tests for hiredis")
    add_subdirectory(${hiredis_SOURCE_DIR} ${hiredis_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
        foonathan_lexy  # used for parsing redis config string
        GIT_REPOSITORY https://github.com/foonathan/lexy.git
        GIT_TAG v2022.05.1
)

FetchContent_MakeAvailable(foonathan_lexy)
