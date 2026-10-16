#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn
from parameterized import parameterized
from torchrec.distributed.dist_data import _get_recat
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)
from torchrec.distributed.pec_comm_ops import PECGradientApply
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import QuantizedCommCodecs
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def _generate_gradient(
    num_rows: int,
    dim: int,
    rank: int,
    partition: int,
    device: torch.device,
) -> torch.Tensor:
    """Generates distinct fractional gradients for each rank and partition."""
    return torch.sin(
        torch.arange(num_rows * dim, device=device, dtype=torch.float32)
        + 0.37
        + rank
        + partition
    ).view(-1, dim)


def _expected_qcomm_gradient(
    splits: list[list[int]],
    rank: int,
    partition: int,
    dim: int,
    device: torch.device,
    codecs: QuantizedCommCodecs,
) -> torch.Tensor:
    """Computes this receiver's gradients from each sender without communication."""
    expected_chunks = []
    for source, source_splits in enumerate(splits):
        # The reference reconstructs the sender's gradient and passes it through
        # the codec so the expected result includes quantization loss.
        source_grad = _generate_gradient(
            sum(source_splits), dim, source, partition, device
        ).view(-1)
        codec_ctx = codecs.backward.create_context()
        decoded = codecs.backward.decode(
            codecs.backward.encode(source_grad, codec_ctx), codec_ctx
        ).view(-1, dim)

        # The reference extracts this receiver's slice from each sender buffer.
        # AllToAll concatenates these slices in source-rank order on receipt.
        start = sum(source_splits[:rank])
        expected_chunks.append(decoded[start : start + source_splits[rank]])

    return torch.cat(expected_chunks)


@dataclass
class _PendingGradientDist:
    name: str
    applier: PECGradientApply
    expected: torch.Tensor


def _test_qcomm_gradient_dist(
    rank: int,
    world_size: int,
    qcomms_config: QCommsConfig,
) -> None:
    """Checks quantized AllToAll buffers independently of embedding lookup."""
    with MultiProcessContext(rank, world_size, "nccl") as ctx:
        assert ctx.pg is not None
        codecs = get_qcomm_codecs(qcomms_config)
        dim = 32  # Aligns embedding rows with INT8/FP8 rows and MX4 groups.
        # Counts refer to rows sent between peers, not trainer batch sizes.
        cases = [
            ("uneven_splits", [[2, 1], [3, 0]]),
            ("empty_sender", [[0, 0], [1, 2]]),
            ("empty_receiver", [[1, 0], [2, 0]]),
        ]
        pending = []
        for partition, (name, splits) in enumerate(cases):
            send_rows = splits[rank]
            receive_rows = [splits[source][rank] for source in range(world_size)]
            grad = _generate_gradient(sum(send_rows), dim, rank, partition, ctx.device)
            applier = PECGradientApply(grad, send_rows, receive_rows, ctx.pg, codecs)
            applier.dist()
            work, buffer, codec_ctx = (
                applier._work,
                applier._grad_buffer,
                applier._qcomm_ctx,
            )

            applier.dist()
            assert applier._work is work, name
            assert applier._grad_buffer is buffer, name
            assert applier._qcomm_ctx is codec_ctx, name
            pending.append(
                _PendingGradientDist(
                    name=name,
                    applier=applier,
                    expected=_expected_qcomm_gradient(
                        splits, rank, partition, dim, ctx.device, codecs
                    ),
                )
            )

        # Reverse consumption checks that exchanges retain independent buffers
        # and codec contexts. No lookup, recat, or embedding backward runs here.
        for exchange in reversed(pending):
            applier = exchange.applier
            assert applier._work is not None
            assert applier._grad_buffer is not None
            applier._work.wait()
            received = codecs.backward.decode(applier._grad_buffer, applier._qcomm_ctx)
            torch.testing.assert_close(
                received.view(-1, dim),
                exchange.expected,
                msg=lambda message, name=exchange.name: f"{name}: {message}",
            )


def _test_qcomm_gradient_apply(
    rank: int,
    world_size: int,
    qcomms_config: QCommsConfig,
) -> None:
    """Checks decoding, rank-to-feature permutation, and embedding backward."""
    with MultiProcessContext(rank, world_size, "nccl") as ctx:
        assert ctx.pg is not None
        codecs = get_qcomm_codecs(qcomms_config)
        dim = 32
        # Rank 0 contributes two IDs per receiver; rank 1 contributes three.
        splits = [[2, 2], [3, 3]]
        grad = _generate_gradient(sum(splits[rank]), dim, rank, 0, ctx.device)
        applier = PECGradientApply(grad, splits[rank], [2, 3], ctx.pg, codecs)
        applier.dist()  # A real exchange supplies the buffer consumed by apply().

        # Rank 0 has one sample and rank 1 has two samples:
        #                  rank 0    rank 1
        # feature 0          [1]     [1, 0]
        # feature 1          [1]     [1, 1]
        # These bag lengths total two IDs from rank 0 and three from rank 1.
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.arange(5, device=ctx.device),
            lengths=torch.tensor([1, 1, 0, 1, 1, 1], device=ctx.device),
        )
        grad_recat = _get_recat(
            2, world_size, device=ctx.device, batch_size_per_rank=[1, 2]
        )
        assert grad_recat is not None
        # Receive order groups rank 0's features before rank 1's features.
        grad_recat_lengths = torch.tensor([1, 1, 1, 0, 1, 1], device=ctx.device) * dim
        embedding = nn.Embedding(5, dim, device=ctx.device)
        with torch.no_grad():
            applier.apply(
                features,
                lambda kjt: embedding(kjt.values()),
                dim,
                grad_recat=grad_recat,
                grad_recat_lengths=grad_recat_lengths,
            )

        # The independent reference gathers feature 0 from both sources, then
        # feature 1. Unique IDs expose each received row in the weight gradient.
        received = _expected_qcomm_gradient(splits, rank, 0, dim, ctx.device, codecs)
        source_0, source_1 = received.split([2, 3])
        expected = torch.cat([source_0[:1], source_1[:1], source_0[1:], source_1[1:]])
        torch.testing.assert_close(embedding.weight.grad, expected)
        assert applier._work is None
        assert applier._grad_buffer is None
        assert applier._qcomm_ctx is None


def _test_empty_gradient_apply(rank: int, world_size: int) -> None:
    """Checks that a receiver with no rows completes without embedding backward."""
    with MultiProcessContext(rank, world_size, "nccl") as ctx:
        assert ctx.pg is not None
        dim = 32
        codecs = get_qcomm_codecs(QCommsConfig(backward_precision=CommType.BF16))
        splits = [[1, 0], [2, 0]]
        receive_rows = [splits[source][rank] for source in range(world_size)]
        grad = _generate_gradient(sum(splits[rank]), dim, rank, 0, ctx.device)
        applier = PECGradientApply(grad, splits[rank], receive_rows, ctx.pg, codecs)
        applier.dist()

        num_rows = sum(receive_rows)
        embedding = nn.Embedding(max(num_rows, 1), dim, device=ctx.device)
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature"],
            values=torch.arange(num_rows, device=ctx.device),
            lengths=torch.tensor([num_rows], device=ctx.device),
        )
        applier.apply(features, lambda kjt: embedding(kjt.values()), dim)
        if num_rows:
            expected = _expected_qcomm_gradient(
                splits, rank, 0, dim, ctx.device, codecs
            )
            torch.testing.assert_close(embedding.weight.grad, expected)
        else:
            assert embedding.weight.grad is None
        assert applier._work is None
        assert applier._grad_buffer is None
        assert applier._qcomm_ctx is None


@skip_if_asan_class
@unittest.skipIf(
    not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 2,
    "Requires NCCL and two GPUs",
)
class PECCommOpsTest(MultiProcessTestBase):
    @parameterized.expand(
        [
            (
                "fp16_scaled",
                QCommsConfig(backward_precision=CommType.FP16, backward_loss_scale=8.0),
            ),
            ("bf16", QCommsConfig(backward_precision=CommType.BF16)),
            (
                "fp8_rowwise",
                QCommsConfig(backward_precision=CommType.FP8, fp8_quantize_dim_bwd=32),
            ),
            ("int8", QCommsConfig(backward_precision=CommType.INT8)),
            (
                "mx4",
                QCommsConfig(backward_precision=CommType.MX4, mx4_quantize_dim_bwd=32),
            ),
        ]
    )
    def test_gradient_dist(self, name: str, qcomms_config: QCommsConfig) -> None:
        self._run_multi_process_test(
            callable=_test_qcomm_gradient_dist,
            world_size=2,
            qcomms_config=qcomms_config,
        )

    @parameterized.expand(
        [
            (
                "fp16_scaled",
                QCommsConfig(backward_precision=CommType.FP16, backward_loss_scale=8.0),
            ),
            ("bf16", QCommsConfig(backward_precision=CommType.BF16)),
            (
                "fp8_rowwise",
                QCommsConfig(backward_precision=CommType.FP8, fp8_quantize_dim_bwd=32),
            ),
            ("int8", QCommsConfig(backward_precision=CommType.INT8)),
            (
                "mx4",
                QCommsConfig(backward_precision=CommType.MX4, mx4_quantize_dim_bwd=32),
            ),
        ]
    )
    def test_gradient_apply(self, name: str, qcomms_config: QCommsConfig) -> None:
        self._run_multi_process_test(
            callable=_test_qcomm_gradient_apply,
            world_size=2,
            qcomms_config=qcomms_config,
        )

    def test_empty_gradient_receiver(self) -> None:
        self._run_multi_process_test(
            callable=_test_empty_gradient_apply,
            world_size=2,
        )
