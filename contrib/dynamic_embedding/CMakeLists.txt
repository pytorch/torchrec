cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

project(src/torchrec_dynamic_embedding
        VERSION 0.0.1
        LANGUAGES CXX C)

find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)

add_subdirectory(src)
