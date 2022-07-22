cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

project(torchrec_dynamic_embedding
        VERSION 0.0.1
        LANGUAGES CXX C)

find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(TDE_IS_TOP_LEVEL_PROJECT ON)
else()
    set(TDE_IS_TOP_LEVEL_PROJECT OFF)
endif()

option(TDE_WITH_TESTING "Enable unittest in C++ side" ${TDE_IS_TOP_LEVEL_PROJECT})

if (TDE_WITH_TESTING)
    enable_testing()
endif()
add_subdirectory(src)
