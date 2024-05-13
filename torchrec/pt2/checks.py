#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List

import torch


try:
    if torch.jit.is_scripting():
        raise Exception()

    from torch.compiler import (
        is_compiling as is_compiler_compiling,
        is_dynamo_compiling as is_torchdynamo_compiling,
    )

    def is_non_strict_exporting() -> bool:
        return not is_torchdynamo_compiling() and is_compiler_compiling()

except Exception:
    # BC for torch versions without compiler and torch deploy path
    def is_torchdynamo_compiling() -> bool:  # type: ignore[misc]
        return False

    def is_non_strict_exporting() -> bool:
        return False


def pt2_checks_tensor_slice(
    tensor: torch.Tensor, start_offset: int, end_offset: int, dim: int = 0
) -> None:
    if torch.jit.is_scripting() or not is_torchdynamo_compiling():
        return

    torch._check_is_size(start_offset)
    torch._check_is_size(end_offset)
    torch._check_is_size(end_offset - start_offset)
    torch._check(start_offset <= tensor.size(dim))
    torch._check(end_offset <= tensor.size(dim))
    torch._check(end_offset >= start_offset)


def pt2_checks_all_is_size(list: List[int]) -> List[int]:
    if torch.jit.is_scripting() or not is_torchdynamo_compiling():
        return list

    for i in list:
        torch._check_is_size(i)
    return list
