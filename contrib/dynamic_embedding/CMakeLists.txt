cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

project(torchrec_dynamic_embedding
        VERSION 0.0.1
        LANGUAGES CXX C)

option(TDE_TORCH_BASE_DIR "torch python directory" "")

if (NOT TDE_TORCH_BASE_DIR)
    message(FATAL_ERROR "TDE_TORCH_BASE_DIR must set."
        "Use python -c 'import torch;import os.path;print(os.path.dirname(torch.__file__))'"
            " to get this dir.")
else()
    find_package(Torch REQUIRED
            PATHS "${TDE_TORCH_BASE_DIR}"
            NO_DEFAULT_PATH)
endif()


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(TDE_IS_TOP_LEVEL_PROJECT ON)
else()
    set(TDE_IS_TOP_LEVEL_PROJECT OFF)
endif()

option(TDE_WITH_TESTING "Enable unittest in C++ side" ${TDE_IS_TOP_LEVEL_PROJECT})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    option(TDE_WITH_CXX11_ABI "GLIBCXX use c++11 ABI or not. libtorch installed by conda is not use it by default" OFF)
    if (TDE_WITH_CXX11_ABI)
        add_definitions("-D_GLIBCXX_USE_CXX11_ABI=1")
    else()
        add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

if (TDE_WITH_TESTING)
    enable_testing()
endif()
add_subdirectory(src)
