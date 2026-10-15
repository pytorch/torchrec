#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
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


def _test_qcomm_gradient_apply(
    rank: int,
    world_size: int,
    qcomms_config: QCommsConfig,
) -> None:
    """Exercises real NCCL exchanges with separate contexts and uneven splits."""
    with MultiProcessContext(rank, world_size, "nccl") as ctx:
        assert ctx.pg is not None
        codecs = get_qcomm_codecs(qcomms_config)

        # This dimension aligns embedding rows with INT8/FP8 rows and MX4 groups.
        dim = 32

        # splits[source][destination] counts embedding rows sent between ranks.
        split_matrices = [
            [[2, 1], [3, 0]],  # The OL case includes uneven and zero peer splits.
            [[0, 0], [1, 2]],  # NOL: rank 0 sends no rows.
            [[1, 0], [2, 0]],  # Additional edge case: rank 1 receives no rows.
        ]
        pending = []

        # The test starts every exchange before applying any gradients. Each
        # applier retains its codec context and receive buffer until apply().
        for partition, splits in enumerate(split_matrices):
            send = splits[rank]
            receive = [splits[source][rank] for source in range(world_size)]

            # Distinct values per rank/partition expose incorrect routing;
            # fractional values also expose accidentally bypassed quantization.
            grad = _generate_gradient(sum(send), dim, rank, partition, ctx.device)

            applier = PECGradientApply(grad, send, receive, ctx.pg, codecs)
            applier.dist()
            applier.dist()  # The second call checks idempotency.

            # The lookup maps each received row to a unique embedding ID. Its
            # weight gradient equals that row without duplicate-ID reduction.
            embedding = nn.Embedding(max(sum(receive), 1), dim, device=ctx.device)
            features = KeyedJaggedTensor.from_lengths_sync(
                keys=["feature"],
                values=torch.arange(sum(receive), device=ctx.device),
                lengths=torch.tensor([sum(receive)], device=ctx.device),
            )

            expected = _expected_qcomm_gradient(
                splits, rank, partition, dim, ctx.device, codecs
            )

            assert embedding.weight.grad is None
            pending.append((applier, features, embedding, expected))

        # The test consumes exchanges in reverse order to exercise independent
        # contexts and delayed application, then checks against the reference.
        for applier, features, embedding, expected in reversed(pending):
            applier.apply(
                features, lambda kjt, module=embedding: module(kjt.values()), dim
            )

            if expected.numel():
                torch.testing.assert_close(embedding.weight.grad, expected)
            else:
                # An empty receiver completes communication without an embedding
                # lookup or backward call.
                assert embedding.weight.grad is None


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
    def test_gradient_apply(self, name: str, qcomms_config: QCommsConfig) -> None:
        self._run_multi_process_test(
            callable=_test_qcomm_gradient_apply,
            world_size=2,
            qcomms_config=qcomms_config,
        )
