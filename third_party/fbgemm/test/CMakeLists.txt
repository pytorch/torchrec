cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(FBGEMM_BUILD_TESTS AND NOT TARGET gtest)
  #Download Googletest framework from github if
  #GOOGLETEST_SOURCE_DIR is not specified.
  if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    set(GOOGLETEST_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/third_party/googletest"
      CACHE STRING "googletest source directory from submodules")
  endif()

  #build Googletest framework
  #MSVC needs gtest_for_shared_crt to select right runtime lib
  if (MSVC AND FBGEMM_LIBRARY_TYPE STREQUAL "shared")
    message(WARNING "gtest_force_shared_crt is ON")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  else()
    message(WARNING "gtest_force_shared_crt is OFF")
  endif()
  add_subdirectory("${GOOGLETEST_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/googletest")
endif()

if (USE_SANITIZER)
  message(WARNING "USING SANITIZER IN TEST")
endif()

if(${OpenMP_FOUND})
  message(STATUS "OpenMP_LIBRARIES= ${OpenMP_CXX_LIBRARIES}")
endif()

macro(add_gtest TESTNAME)
  add_executable(${TESTNAME} ${ARGN}
    ../bench/BenchUtils.cc
    EmbeddingSpMDMTestUtils.cc
    QuantizationHelpers.cc
    TestUtils.cc)
  set_target_properties(${TESTNAME} PROPERTIES
          CXX_STANDARD 11
          CXX_EXTENSIONS NO)
  #To compile test files with AVX2 turned on
  #For static build, defining FBGEMM_STATIC to avoid generating
  #functions with _dllimport attributes.
  if(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
      "/arch:AVX2" "/wd4244" "/wd4267" "/wd4305" "/wd4309")
    if (FBGEMM_LIBRARY_TYPE STREQUAL "static")
      target_compile_definitions(${TESTNAME} PRIVATE FBGEMM_STATIC)
    endif()
  else(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
     "-m64" "-mavx2" "-mfma" "-masm=intel")
  endif(MSVC)
  if (USE_SANITIZER)
    target_compile_options(${TESTNAME} PRIVATE
      "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
    target_link_options(${TESTNAME} PRIVATE
      "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  endif()

  target_link_libraries(${TESTNAME} gtest gmock gtest_main fbgemm)

  if(${OpenMP_FOUND})
    target_link_libraries(${TESTNAME} ${OpenMP_CXX_LIBRARIES})
  endif()

  add_dependencies(${TESTNAME} gtest fbgemm)
  add_test(${TESTNAME} ${TESTNAME})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()


file(GLOB TEST_LIST "*Test.cc")

foreach(TEST_FILE ${TEST_LIST})
  get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
  get_filename_component(TEST_FILE_ONLY "${TEST_FILE}" NAME)
  add_gtest("${TEST_NAME}"
    "${TEST_FILE_ONLY}")
endforeach()
