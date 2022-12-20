#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import hypothesis.strategies as st
import torch
import torch.distributed as dist
import torch.quantization as quant
from hypothesis import given, settings, Verbosity
from torchrec.distributed.embedding_lookup import (
    BatchedDenseEmbedding,
    BatchedDenseEmbeddingBag,
    BatchedFusedEmbedding,
    BatchedFusedEmbeddingBag,
    GroupedEmbeddingsLookup,
    GroupedPooledEmbeddingsLookup,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torchrec.distributed.quant_embedding_kernel import (
    QuantBatchedEmbedding,
    QuantBatchedEmbeddingBag,
)
from torchrec.modules.embedding_configs import DataType, PoolingType
from torchrec.test_utils import get_free_port


def quantize_sharded_embeddings(
    module: torch.nn.Module, dtype: torch.dtype
) -> torch.nn.Module:
    qconfig = quant.QConfigDynamic(
        activation=quant.PlaceholderObserver,
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
    )
    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            BatchedFusedEmbeddingBag: qconfig,
            BatchedDenseEmbeddingBag: qconfig,
            BatchedDenseEmbedding: qconfig,
            BatchedFusedEmbedding: qconfig,
        },
        mapping={
            BatchedFusedEmbeddingBag: QuantBatchedEmbeddingBag,
            BatchedDenseEmbeddingBag: QuantBatchedEmbeddingBag,
            BatchedDenseEmbedding: QuantBatchedEmbedding,
            BatchedFusedEmbedding: QuantBatchedEmbedding,
        },
        inplace=False,
    )


class QuantizeKernelTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.device = torch.device("cuda:0")
        backend = "nccl"
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend=backend)

    def tearDown(self) -> None:
        dist.destroy_process_group()
        del os.environ["NCCL_SOCKET_IFNAME"]
        super().tearDown()

    def _create_config(
        self, compute_kernel: EmbeddingComputeKernel
    ) -> GroupedEmbeddingConfig:
        num_embedding_tables = 2
        embedding_tables = []
        for i in range(num_embedding_tables):
            rows = (i + 1) * 10
            cols = 16
            local_metadata = ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[rows, cols],
                placement=torch.distributed._remote_device("rank:0/cuda:0"),
            )
            embedding_tables.append(
                ShardedEmbeddingTable(
                    num_embeddings=rows,
                    embedding_dim=cols,
                    name="table_" + str(i),
                    feature_names=["feature_" + str(i)],
                    pooling=PoolingType.MEAN,
                    is_weighted=False,
                    has_feature_processor=False,
                    local_rows=rows,
                    local_cols=cols,
                    compute_kernel=compute_kernel,
                    local_metadata=local_metadata,
                    global_metadata=ShardedTensorMetadata(
                        shards_metadata=[local_metadata],
                        size=torch.Size([rows, cols]),
                    ),
                    weight_init_max=1.0,
                    weight_init_min=0.0,
                )
            )
        return GroupedEmbeddingConfig(
            data_type=DataType.FP32,
            pooling=PoolingType.MEAN,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=compute_kernel,
            embedding_tables=embedding_tables,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore [56]
    @given(
        compute_kernel=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE,
                EmbeddingComputeKernel.FUSED,
            ]
        ),
        dtype=st.sampled_from(
            [
                torch.qint8,
                torch.quint4x2,
                torch.quint2x4,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=6, deadline=None)
    def test_quantize_embedding_bag_kernels(
        self, compute_kernel: EmbeddingComputeKernel, dtype: torch.dtype
    ) -> None:
        config = self._create_config(compute_kernel)
        sharded = GroupedPooledEmbeddingsLookup(
            grouped_configs=[config],
            device=torch.device("cuda:0"),
        )

        quantized = quantize_sharded_embeddings(sharded, dtype=dtype)

        for _, buffer in quantized.named_buffers():
            self.assertEqual(buffer.dtype, torch.uint8)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore [56]
    @given(
        compute_kernel=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE,
                EmbeddingComputeKernel.FUSED,
            ]
        ),
        dtype=st.sampled_from(
            [
                torch.qint8,
                torch.quint4x2,
                torch.quint2x4,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=6, deadline=None)
    def test_quantize_embedding_kernels(
        self, compute_kernel: EmbeddingComputeKernel, dtype: torch.dtype
    ) -> None:
        config = self._create_config(compute_kernel)
        sharded = GroupedEmbeddingsLookup(
            grouped_configs=[config],
            device=torch.device("cuda:0"),
        )

        quantized = quantize_sharded_embeddings(sharded, dtype=dtype)

        for _, buffer in quantized.named_buffers():
            self.assertEqual(buffer.dtype, torch.uint8)
