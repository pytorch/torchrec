include(FetchContent)

if (TDE_WITH_TESTING)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.12.0
            )


    FetchContent_Declare(google_benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.6.2
            )

    FetchContent_MakeAvailable(googletest google_benchmark)
endif()

# tcb::span section. A std::span implementation in C++ 11.
# the tcb::span is enable_testing globally. so we do not add_subdirectory(span)
FetchContent_Declare(
        tcb_span
        GIT_REPOSITORY https://github.com/tcbrindle/span.git
        GIT_TAG 836dc6a0efd9849cb194e88e4aa2387436bb079b
)
FetchContent_MakeAvailable(tcb_span)
