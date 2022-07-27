#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import cast, Optional

from fbgemm_gpu.quantize_comm import QuantizedCommCodec as FbgemmQuantizedCommCodec
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.types import QuantizedCommCodec, QuantizedCommCodecs

logger: logging.Logger = logging.getLogger()


@unique
class CommType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"

    def __str__(self) -> str:
        return self.value


def comm_type_to_sparse_type(comm_type: CommType) -> SparseType:
    return {
        CommType.FP32: SparseType.FP32,
        CommType.FP16: SparseType.FP16,
        CommType.BF16: SparseType.BF16,
        CommType.FP8: SparseType.FP8,
    }[comm_type]


@dataclass
class QCommsConfig:
    """
    Quantization configs for the AllToAll and ReduceScatter communication modules used in sharding.
    """

    # Quantization of comm modules in the forward pass
    forward_precision: CommType = CommType.FP32
    # Quantization of comm modules in the backward pass
    backward_precision: CommType = CommType.FP32
    # For supported precisions (currently FP16), scale the gradient of the decoder and
    # divide the gradient of the encoder by this value. In some cases this can provide additional numerical stability.
    forward_loss_scale: Optional[float] = None
    backward_loss_scale: Optional[float] = None


def get_qcomm_codecs(qcomms_config: Optional[QCommsConfig]) -> QuantizedCommCodecs:
    codecs = QuantizedCommCodecs()
    if qcomms_config is not None:
        codecs.forward = cast(
            QuantizedCommCodec,
            FbgemmQuantizedCommCodec(
                comm_precision=comm_type_to_sparse_type(
                    qcomms_config.forward_precision
                ),
                loss_scale=qcomms_config.forward_loss_scale,
            ),
        )
        codecs.backward = cast(
            QuantizedCommCodec,
            FbgemmQuantizedCommCodec(
                comm_precision=comm_type_to_sparse_type(
                    qcomms_config.backward_precision
                ),
                loss_scale=qcomms_config.backward_loss_scale,
            ),
        )
    return codecs
