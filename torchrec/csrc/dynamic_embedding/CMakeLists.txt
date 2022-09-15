add_library(tde_cpp_objs
            OBJECT
            details/clz_impl.cpp
            details/ctz_impl.cpp)

target_include_directories(tde_cpp_objs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
