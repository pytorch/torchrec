#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from functools import reduce
from typing import Any, Iterable, Type, Union

import torch

# pyre-ignore[2]
def sharder_name(t: Type[Any]) -> str:
    return t.__module__ + "." + t.__name__


def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes / (1024 * 1024 * 1024))


def bytes_to_mb(num_bytes: Union[float, int]) -> float:
    return float(num_bytes / (1024 * 1024))


def gb_to_bytes(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)


def prod(iterable: Iterable[int]) -> int:
    return reduce(operator.mul, iterable, 1)


def placement(
    compute_device: str,
    rank: int,
    local_size: int,
) -> str:
    """
    Returns placement, formatted as string
    """

    param_device = compute_device
    if compute_device == "cuda":
        param_device = torch.device("cuda", rank % local_size)
    return f"rank:{rank}/{param_device}"
