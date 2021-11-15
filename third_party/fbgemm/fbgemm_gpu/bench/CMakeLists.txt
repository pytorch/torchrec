cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if (USE_SANITIZER)
    message(WARNING "USING SANITIZER IN BENCHMARK")
endif()

if (${OpenMP_FOUND})
  message(STATUS "OpenMP_LIBRARIES= ${OpenMP_CXX_LIBRARIES}")
endif()

#benchmarks
macro(add_benchmark BENCHNAME)
  add_executable(${BENCHNAME} ${ARGN})
  set_target_properties(${BENCHNAME} PROPERTIES
          CUDA_SEPARABLE_COMPILATION OFF
          CXX_STANDARD 11
          CXX_EXTENSIONS NO)
  target_link_libraries(${BENCHNAME} fbgemm_gpu -lcurand)
  add_dependencies(${BENCHNAME} fbgemm_gpu)

  if (USE_SANITIZER)
    target_compile_options(${BENCHNAME} PRIVATE
      "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
    target_link_options(${BENCHNAME} PRIVATE "-fsanitize=${USE_SANITIZER}")
  endif()

  if(${OpenMP_FOUND})
    target_link_libraries(${BENCHNAME} "${OpenMP_CXX_LIBRARIES}")
  endif()

  set_target_properties(${BENCHNAME} PROPERTIES FOLDER test)
endmacro()

if(FBGEMMGPU_BUILD_BENCHMARKS)

  set(BENCHMARKS "")

  file(GLOB BENCH_LIST "*_benchmark.cu")
  foreach(BENCH_FILE ${BENCH_LIST})
    get_filename_component(BENCH_NAME "${BENCH_FILE}" NAME_WE)
    get_filename_component(BENCH_FILE_ONLY "${BENCH_FILE}" NAME)
    add_benchmark("${BENCH_NAME}"
      "${BENCH_FILE_ONLY}")
    list(APPEND BENCHMARKS "${BENCH_NAME}")
  endforeach()

  add_custom_target(run_benchmarks
    COMMAND ${BENCHMARKS})

  add_dependencies(run_benchmarks
    ${BENCHMARKS})

endif()
