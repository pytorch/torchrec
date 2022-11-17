#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import copy
import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import cast, Dict, List, Optional

import torch

from fbgemm_gpu.quantize_comm import (
    QuantizationContext,
    QuantizedCommCodec as FbgemmQuantizedCommCodec,
)
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.types import CommOp, QuantizedCommCodec, QuantizedCommCodecs

logger: logging.Logger = logging.getLogger()


@unique
class CommType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"

    def __str__(self) -> str:
        return self.value


def comm_type_to_sparse_type(comm_type: CommType) -> SparseType:
    return {
        CommType.FP32: SparseType.FP32,
        CommType.FP16: SparseType.FP16,
        CommType.BF16: SparseType.BF16,
        CommType.FP8: SparseType.FP8,
        CommType.INT8: SparseType.INT8,
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
    fp8_quantize_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            self.forward_precision != CommType.FP8
            and self.backward_precision != CommType.FP8
            and self.fp8_quantize_dim is not None
        ):
            raise ValueError(
                f"fp8_quantize_dim is set to {self.fp8_quantize_dim} but no FP8 precision is found in forward and backward precisions"
            )


def get_qcomm_codecs(qcomms_config: Optional[QCommsConfig]) -> QuantizedCommCodecs:
    codecs = QuantizedCommCodecs()
    if qcomms_config is not None:
        codecs.forward = cast(
            QuantizedCommCodec[QuantizationContext],
            FbgemmQuantizedCommCodec(
                comm_precision=comm_type_to_sparse_type(
                    qcomms_config.forward_precision
                ),
                loss_scale=qcomms_config.forward_loss_scale,
                is_fwd=True,
                row_dim=qcomms_config.fp8_quantize_dim
                if qcomms_config.forward_precision == CommType.FP8
                else None,
            ),
        )
        codecs.backward = cast(
            QuantizedCommCodec[QuantizationContext],
            FbgemmQuantizedCommCodec(
                comm_precision=comm_type_to_sparse_type(
                    qcomms_config.backward_precision
                ),
                loss_scale=qcomms_config.backward_loss_scale,
                is_fwd=False,
                row_dim=qcomms_config.fp8_quantize_dim
                if qcomms_config.backward_precision == CommType.FP8
                else None,
            ),
        )
    return codecs


def get_qcomm_codecs_registry(
    qcomms_config: QCommsConfig,
    comm_ops: Optional[List[CommOp]] = None,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, QuantizedCommCodecs]]:
    """
     This method constructs QuantizedCommCodecs from a given QCommConfig. It assumes
     that you want to use the same QComm configs for all comm-types passed in.

     Some quantization schemes are not supported for some backends (such as BF16 for gloo/cpu, and FP8 for reduce scatter on nccl).
     This scheme will provide some fallback logic and print a warning.

    Args:
        qcomms_config (QCommsConfig): QCommsConfig to construct FBGEMMQuantizedCommCodecs from
        comm_ops (Optional[List[CommOp]]): List of CommOps to enter into the registry
        device (torch.device): Backend comms will run on.

    Example::
        qcomm_codces_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(forward_precision=FP16, backward_precision=BF16),
            device=torch.device("cuda"))
    """

    if (
        qcomms_config.forward_precision == CommType.FP32
        and qcomms_config.backward_precision == CommType.FP32
    ):
        return None

    if device is None:
        device = torch.device("cuda")

    qcomm_codecs_registry = {}
    if comm_ops is None:
        comm_ops = [
            CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL,
            CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER,
            CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL,
        ]
    for comm_op in comm_ops:
        qcomm_config_copy = copy.deepcopy(qcomms_config)
        # TODO: On H100, FP8 types might be natively supported, in which case we should check for that arch type and not fallback.
        if comm_op == CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER:
            if qcomm_config_copy.forward_precision == CommType.FP8:
                logger.warning(
                    "FP8 is not supported for reduce scatter's forward - falling back to FP16"
                )
                qcomm_config_copy.forward_precision = CommType.FP16
            if qcomm_config_copy.backward_precision == CommType.FP8:
                logger.warning(
                    "FP8 is not supported for reduce scatter's backward - falling back to BF16"
                )
                qcomm_config_copy.backward_precision = CommType.BF16

        if device.type == "cpu":
            if qcomm_config_copy.forward_precision == CommType.BF16:
                logger.warning(
                    "BF16 is not for forward_precision is not supported on GLOO - falling back to FP16."
                )
                qcomm_config_copy.forward_precision = CommType.FP16

            if qcomm_config_copy.backward_precision == CommType.BF16:
                logger.warning(
                    "BF16 is not for backward_precision is not supported on GLOO - falling back to FP16."
                )
                qcomm_config_copy.backward_precision = CommType.FP16

        qcomm_codecs_registry[comm_op.name] = get_qcomm_codecs(qcomm_config_copy)

    return qcomm_codecs_registry
