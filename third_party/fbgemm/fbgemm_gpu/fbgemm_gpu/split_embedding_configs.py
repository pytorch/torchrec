#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Dict


@enum.unique
class EmbOptimType(enum.Enum):
    SGD = "sgd"  # uses non-deterministic updates (atomicAdd(..)) with duplicate ids
    EXACT_SGD = (
        "exact_sgd"  # uses deterministic updates (via sorting + segment reduction)
    )
    LAMB = "lamb"
    ADAM = "adam"
    # exact/dedup: gradients to the same row are applied with coalesce then apply
    # together, instead of applied in sequence (approx).
    EXACT_ADAGRAD = "exact_adagrad"
    EXACT_ROWWISE_ADAGRAD = "exact_row_wise_adagrad"
    LARS_SGD = "lars_sgd"
    PARTIAL_ROWWISE_ADAM = "partial_row_wise_adam"
    PARTIAL_ROWWISE_LAMB = "partial_row_wise_lamb"
    ROWWISE_ADAGRAD = "row_wise_adagrad"
    MADGRAD = "madgrad"

    def __str__(self) -> str:
        return self.value


@enum.unique
class SparseType(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_int(ty: int) -> "SparseType":
        if ty == 0:
            return SparseType("fp32")
        elif ty == 1:
            return SparseType("fp16")
        elif ty == 2:
            return SparseType("int8")
        elif ty == 3:
            return SparseType("int4")
        elif ty == 4:
            return SparseType("int2")
        else:
            raise ValueError(f"Unsupported sparse type: {ty}")

    def as_int(self) -> int:
        return {
            SparseType.FP32.value: 0,
            SparseType.FP16.value: 1,
            SparseType.INT8.value: 2,
            SparseType.INT4.value: 3,
            SparseType.INT2.value: 4,
        }[self.value]

    def bit_rate(self) -> int:
        return {
            SparseType.FP32.value: 32,
            SparseType.FP16.value: 16,
            SparseType.INT8.value: 8,
            SparseType.INT4.value: 4,
            SparseType.INT2.value: 2,
        }[self.value]

    def align_size(self) -> int:
        return {
            SparseType.FP32.value: 1,
            SparseType.FP16.value: 2,
            SparseType.INT8.value: 4,
            SparseType.INT4.value: 8,
            SparseType.INT2.value: 16,
        }[self.value]

    def is_float(self) -> bool:
        if self.value == SparseType.FP32.value or self.value == SparseType.FP16.value:
            return True
        else:
            return False


ELEMENT_SIZE: Dict[SparseType, int] = {
    SparseType.FP32: 4,
    SparseType.FP16: 2,
    SparseType.INT8: 1,
    # SparseType.INT4: 0.5,
}
