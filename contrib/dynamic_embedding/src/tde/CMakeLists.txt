# tde_cpp_objs section, used by both unittests and tde_cpp.so
add_library(tde_cpp_objs OBJECT id_transformer.cpp ps.cpp
        details/naive_id_transformer.cpp
        details/cacheline_id_transformer.cpp
        details/io.cpp details/io_registry.cpp
        details/move_only_function.cpp
        details/random_bits_generator.cpp details/mixed_lfu_lru_strategy.cpp
        details/clz_impl.cpp details/ctz_impl.cpp
        details/id_transformer_variant.cpp details/redis_io.cpp details/redis_io_v1.cpp
        details/notification.cpp)
target_include_directories(tde_cpp_objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(tde_cpp_objs PUBLIC ${TORCH_LIBRARIES})
target_include_directories(tde_cpp_objs PUBLIC ${TORCH_INCLUDE_DIRS})
target_compile_features(tde_cpp_objs PUBLIC cxx_std_17)
target_link_libraries(tde_cpp_objs PUBLIC tcb::span nlohmann_json hiredis::hiredis_static)
target_link_libraries(tde_cpp_objs PRIVATE foonathan::lexy::core)
target_compile_options(tde_cpp_objs PUBLIC -fPIC)
target_compile_definitions(tde_cpp_objs PUBLIC -DJSON_HAS_CPP_17)
target_link_libraries(tde_cpp_objs PUBLIC ${CMAKE_DL_LIBS})

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
    
    add_tde_test(naive_id_transformer_test details/naive_id_transformer_test.cpp)
    add_tde_benchmark(naive_id_transformer_benchmark details/naive_id_transformer_benchmark.cpp)

    add_tde_test(cacheline_id_transformer_test details/cacheline_id_transformer_test.cpp)
    add_tde_benchmark(cacheline_id_transformer_benchmark details/cacheline_id_transformer_benchmark.cpp)

    add_tde_test(move_only_function_test details/move_only_function_test.cpp)
    add_tde_test(random_bits_generator_test details/random_bits_generator_test.cpp)
    add_tde_test(mixed_lfu_lru_strategy_test details/mixed_lfu_lru_strategy_test.cpp)
    add_tde_test(bits_op_test details/bits_op_test.cpp)
    add_tde_benchmark(mixed_lfu_lru_strategy_benchmark details/mixed_lfu_lru_strategy_benchmark.cpp)
    add_tde_benchmark(random_bits_generator_benchmark details/random_bits_generator_benchmark.cpp)
    add_tde_test(id_transformer_variant_test details/id_transformer_variant_test.cpp)
    # TODO: Need start a empty redis-server on 127.0.0.1:6379 before run *redis*_test.
    add_tde_test(redis_io_v1_test details/redis_io_v1_test.cpp)
    add_tde_test(io_redis_test details/io_redis_test.cpp)
    add_tde_test(url_test details/url_test.cpp)
    target_link_libraries(url_test foonathan::lexy::core)
    add_tde_test(notification_test details/notification_test.cpp)

    add_tde_benchmark(mixed_lfu_lru_strategy_evict_benchmark
            details/mixed_lfu_lru_strategy_evict_benchmark.cpp)
endif ()
