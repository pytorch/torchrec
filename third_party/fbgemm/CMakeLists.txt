cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from defs.bzl file
find_package(PythonInterp)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

project(fbgemm VERSION 0.1 LANGUAGES CXX C)

set(FBGEMM_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (shared, static, or default) to build")
set_property(CACHE FBGEMM_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(FBGEMM_BUILD_TESTS "Build fbgemm unit tests" ON)
option(FBGEMM_BUILD_BENCHMARKS "Build fbgemm benchmarks" ON)
option(FBGEMM_BUILD_DOCS "Build fbgemm documentation" OFF)

if(FBGEMM_BUILD_TESTS)
  enable_testing()
endif()

set(FBGEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FBGEMM_THIRDPARTY_DIR ${FBGEMM_BINARY_DIR}/third_party)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#add address sanitizer
set(USE_SANITIZER "" CACHE STRING "options include address, leak, ...")

#check if compiler supports avx512
include(CheckCXXCompilerFlag)
if(MSVC)
  CHECK_CXX_COMPILER_FLAG(/arch:AVX512 COMPILER_SUPPORTS_AVX512)
  set(MSVC_BOOL True)
else(MSVC)
  CHECK_CXX_COMPILER_FLAG(-mavx512f COMPILER_SUPPORTS_AVX512)
  set(MSVC_BOOL False)
endif(MSVC)
if(NOT COMPILER_SUPPORTS_AVX512)
  message(FATAL_ERROR "A compiler with AVX512 support is required.")
endif()

#We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

#check if compiler supports openmp
find_package(OpenMP)
if (OpenMP_FOUND)
  message(WARNING "OpenMP found! OpenMP_C_INCLUDE_DIRS = ${OpenMP_C_INCLUDE_DIRS}")
  include_directories(${OpenMP_C_INCLUDE_DIRS})
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(WARNING "OpenMP is not supported by the compiler")
endif()

# Define file lists
get_filelist("get_fbgemm_generic_srcs(with_base=True)" FBGEMM_GENERIC_SRCS)
get_filelist("get_fbgemm_avx2_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX2_SRCS)
get_filelist("get_fbgemm_inline_avx2_srcs(msvc=${MSVC_BOOL})"
  FBGEMM_AVX2_INLINE_SRCS)
get_filelist("get_fbgemm_avx512_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX512_SRCS)
get_filelist("get_fbgemm_inline_avx512_srcs(msvc=${MSVC_BOOL})"
  FBGEMM_AVX512_INLINE_SRCS)
get_filelist("get_fbgemm_public_headers()" FBGEMM_PUBLIC_HEADERS)

add_library(fbgemm_generic OBJECT ${FBGEMM_GENERIC_SRCS})
add_library(fbgemm_avx2 OBJECT ${FBGEMM_AVX2_SRCS} ${FBGEMM_AVX2_INLINE_SRCS})
add_library(fbgemm_avx512 OBJECT
  ${FBGEMM_AVX512_SRCS} ${FBGEMM_AVX512_INLINE_SRCS})

# Make libraries depend on defs.bzl
add_custom_target(defs.bzl DEPENDS defs.bzl)
add_dependencies(fbgemm_generic defs.bzl)
add_dependencies(fbgemm_avx2 defs.bzl)
add_dependencies(fbgemm_avx512 defs.bzl)

set_target_properties(fbgemm_generic fbgemm_avx2 fbgemm_avx512 PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      CXX_VISIBILITY_PRESET hidden)

#On Windows:
#1) Adding definition of ASMJIT_STATIC to avoid generating asmjit function
#calls with _dllimport attribute
#2) MSVC uses /MD in default cxx compiling flags,
#need to change it to /MT in static case
if(MSVC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4305 /wd4309")
  if(FBGEMM_LIBRARY_TYPE STREQUAL "static")
    target_compile_definitions(fbgemm_generic PRIVATE ASMJIT_STATIC)
    target_compile_definitions(fbgemm_avx2 PRIVATE ASMJIT_STATIC)
    target_compile_definitions(fbgemm_avx512 PRIVATE ASMJIT_STATIC)
    foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
  target_compile_options(fbgemm_avx2 PRIVATE "/arch:AVX2")
  target_compile_options(fbgemm_avx512 PRIVATE "/arch:AVX512")
else(MSVC)
  target_compile_options(fbgemm_avx2 PRIVATE
    "-m64" "-mavx2" "-mf16c" "-mfma")
  target_compile_options(fbgemm_avx512 PRIVATE
    "-m64" "-mavx2" "-mfma" "-mavx512f" "-mavx512bw" "-mavx512dq"
    "-mavx512vl")
  set_source_files_properties(src/FbgemmFP16UKernelsAvx2.cc
    PROPERTIES COMPILE_FLAGS "-masm=intel")
  set_source_files_properties(src/FbgemmFP16UKernelsAvx512.cc
    PROPERTIES COMPILE_FLAGS "-masm=intel")
  set_source_files_properties(src/FbgemmFP16UKernelsAvx512_256.cc
    PROPERTIES COMPILE_FLAGS "-masm=intel")
endif(MSVC)

if(USE_SANITIZER)
  target_compile_options(fbgemm_generic PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  target_compile_options(fbgemm_avx2 PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  target_compile_options(fbgemm_avx512 PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

message(WARNING "==========")
message(WARNING "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(WARNING "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message(WARNING "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message(WARNING "==========")

if(NOT TARGET asmjit)
  #Download asmjit from github if ASMJIT_SRC_DIR is not specified.
  if(NOT DEFINED ASMJIT_SRC_DIR)
    set(ASMJIT_SRC_DIR "${FBGEMM_SOURCE_DIR}/third_party/asmjit"
      CACHE STRING "asmjit source directory from submodules")
  endif()

  #build asmjit
  #For MSVC shared build, asmjit needs to be shared also.
  if (MSVC AND (FBGEMM_LIBRARY_TYPE STREQUAL "shared"))
    set(ASMJIT_STATIC OFF)
  else()
    set(ASMJIT_STATIC ON)
  endif()

  add_subdirectory("${ASMJIT_SRC_DIR}" "${FBGEMM_BINARY_DIR}/asmjit")
  set_property(TARGET asmjit PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(NOT TARGET cpuinfo)
  #Download cpuinfo from github if CPUINFO_SOURCE_DIR is not specified.
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/third_party/cpuinfo"
      CACHE STRING "cpuinfo source directory from submodules")
  endif()

  #build cpuinfo
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Do not build cpuinfo unit tests")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Do not build cpuinfo mock tests")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Do not build cpuinfo benchmarks")
  set(CPUINFO_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  #Select static runtime, needed for static build for MSVC
  set(CPUINFO_RUNTIME_TYPE static CACHE STRING "Set runtime to static")
  add_subdirectory("${CPUINFO_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/cpuinfo")
  set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(fbgemm_generic BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

target_include_directories(fbgemm_avx2 BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

target_include_directories(fbgemm_avx512 BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

if(FBGEMM_LIBRARY_TYPE STREQUAL "default")
  add_library(fbgemm
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_avx2>
    $<TARGET_OBJECTS:fbgemm_avx512>)
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "shared")
  add_library(fbgemm SHARED
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_avx2>
    $<TARGET_OBJECTS:fbgemm_avx512>)
  set_property(TARGET fbgemm_generic PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fbgemm_avx2 PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fbgemm_avx512 PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(fbgemm PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "static")
  add_library(fbgemm STATIC
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_avx2>
    $<TARGET_OBJECTS:fbgemm_avx512>)
  #MSVC need to define FBGEMM_STATIC for fbgemm_generic also to
  #avoid generating _dllimport functions.
  target_compile_definitions(fbgemm_generic PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm_avx2 PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm_avx512 PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm PRIVATE FBGEMM_STATIC)
else()
  message(FATAL_ERROR "Unsupported library type ${FBGEMM_LIBRARY_TYPE}")
endif()

if(USE_SANITIZER)
  target_link_options(fbgemm PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

target_include_directories(fbgemm BEFORE
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>)

target_link_libraries(fbgemm $<BUILD_INTERFACE:asmjit>
  $<BUILD_INTERFACE:cpuinfo>)
add_dependencies(fbgemm asmjit cpuinfo)

install(TARGETS fbgemm EXPORT fbgemmLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) #For windows

install(FILES ${FBGEMM_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fbgemm")

install(EXPORT fbgemmLibraryConfig DESTINATION share/cmake/fbgemm
  FILE fbgemmLibraryConfig.cmake)

if(MSVC)
  if(FBGEMM_LIBRARY_TYPE STREQUAL "shared")
    install(
      FILES $<TARGET_PDB_FILE:fbgemm> $<TARGET_PDB_FILE:asmjit>
      DESTINATION ${CMAKE_INSTALL_LIBDIR} OPTIONAL)
  endif()
  install(TARGETS fbgemm DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(TARGETS asmjit DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

#Make project importable from the build directory
#export(TARGETS fbgemm asmjit FILE fbgemmLibraryConfig.cmake)

if(FBGEMM_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(FBGEMM_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(FBGEMM_BUILD_DOCS)
  add_subdirectory(docs)
endif()
