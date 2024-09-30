# A dummy ParamServer IO plugin. It just store params into memory
add_library(memory_io SHARED memory_io.cpp)
set_target_properties(memory_io PROPERTIES PREFIX "")
target_compile_features(memory_io PUBLIC cxx_std_17)
