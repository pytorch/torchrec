# tde_cpp_objs section, used by both unittests and tde_cpp.so
add_library(tde_cpp_objs OBJECT id_transformer.cpp details/multithreaded_id_transformer.cpp)
target_include_directories(tde_cpp_objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(tde_cpp_objs PUBLIC ${TORCH_LIBRARIES})
target_include_directories(tde_cpp_objs PUBLIC ${TORCH_INCLUDE_DIRS})
set_property(TARGET tde_cpp_objs PROPERTY CXX_STANDARD 17)
target_link_libraries(tde_cpp_objs PUBLIC span)

# tde_cpp.so section
add_library(tde_cpp MODULE bind.cpp)
target_link_libraries(tde_cpp PRIVATE tde_cpp_objs)
set_target_properties(tde_cpp PROPERTIES PREFIX "")
set_property(TARGET tde_cpp PROPERTY CXX_STANDARD 17)
INSTALL(TARGETS tde_cpp DESTINATION torchrec_dynamic_embedding)


if (TDE_WITH_TESTING)
    # we use single target for a single test. It will make `ctest -j ` faster.
    # multithreaded_id_transformer_test
    add_executable(multithreaded_id_transformer_test details/multithreaded_id_transformer_test.cpp)
    target_link_libraries(multithreaded_id_transformer_test tde_cpp_objs gtest gtest_main)
    add_test(NAME
            multithreaded_id_transformer_test
            COMMAND
            multithreaded_id_transformer_test --skip-benchmarks # skip benchmark for unit test
            )
    set_property(TARGET multithreaded_id_transformer_test PROPERTY CXX_STANDARD 17)


    add_executable(multithreaded_id_transformer_benchmark details/multithreaded_id_transformer_benchmark.cpp)
    target_link_libraries(multithreaded_id_transformer_benchmark
            tde_cpp_objs benchmark::benchmark_main benchmark::benchmark)
    set_property(TARGET multithreaded_id_transformer_benchmark PROPERTY CXX_STANDARD 17)

endif ()
