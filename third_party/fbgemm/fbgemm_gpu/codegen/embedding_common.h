#pragma once
#include <stdint.h>

// Keep in sync with split_embedding_configs.py:SparseType
enum class SparseType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
    INVALID = 5,
};
