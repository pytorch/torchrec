# tde_cpp_objs section, used by both unittests and tde_cpp.so
add_library(tde_cpp_objs OBJECT id_transformer.cpp details/multithreaded_id_transformer.cpp
        details/move_only_function.cpp)
target_include_directories(tde_cpp_objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(tde_cpp_objs PUBLIC ${TORCH_LIBRARIES})
target_include_directories(tde_cpp_objs PUBLIC ${TORCH_INCLUDE_DIRS})
target_compile_features(tde_cpp_objs PUBLIC cxx_std_17)
target_link_libraries(tde_cpp_objs PUBLIC tcb::span)

# tde_cpp.so section
add_library(tde_cpp MODULE bind.cpp)
target_link_libraries(tde_cpp PRIVATE tde_cpp_objs)
set_target_properties(tde_cpp PROPERTIES PREFIX "")
INSTALL(TARGETS tde_cpp DESTINATION torchrec_dynamic_embedding)


if (TDE_WITH_TESTING)
    function(add_tde_test NAME)
        add_executable(${NAME} ${ARGN})
        target_link_libraries(${NAME} tde_cpp_objs gtest gtest_main)
        add_test(NAME ${NAME} COMMAND ${NAME})
    endfunction()

    function(add_tde_benchmark NAME)
        add_executable(${NAME} ${ARGN})
        target_link_libraries(${NAME} tde_cpp_objs benchmark::benchmark_main benchmark::benchmark)
    endfunction()

    add_tde_test(multithreaded_id_transformer_test details/multithreaded_id_transformer_test.cpp)
    add_tde_benchmark(multithreaded_id_transformer_benchmark details/multithreaded_id_transformer_benchmark.cpp)

    add_tde_test(move_only_function_test details/move_only_function_test.cpp)

endif ()
