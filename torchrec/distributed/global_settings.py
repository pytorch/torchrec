#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os

PROPOGATE_DEVICE: bool = False

TORCHREC_CONSTRUCT_SHARDED_TENSOR_FROM_METADATA_ENV = (
    "TORCHREC_CONSTRUCT_SHARDED_TENSOR_FROM_METADATA"
)


def set_propogate_device(val: bool) -> None:
    global PROPOGATE_DEVICE
    PROPOGATE_DEVICE = val


def get_propogate_device() -> bool:
    global PROPOGATE_DEVICE
    return PROPOGATE_DEVICE


def construct_sharded_tensor_from_metadata_enabled() -> bool:
    return (
        os.environ.get(TORCHREC_CONSTRUCT_SHARDED_TENSOR_FROM_METADATA_ENV, "0") == "1"
    )


def enable_construct_sharded_tensor_from_metadata() -> None:
    os.environ[TORCHREC_CONSTRUCT_SHARDED_TENSOR_FROM_METADATA_ENV] = "1"
