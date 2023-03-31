#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import copy
import unittest
from typing import List, Tuple

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.shard import _shard_modules

from torchrec.distributed.test_utils.infer_utils import (
    model_input_to_forward_args,
    prep_inputs,
    quantize,
    TestModelInfo,
    TestQuantEBCSharder,
    TorchTypesModelInputWrapper,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    ModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
)
from torchrec.fx.tracer import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig


def assert_close(expected, got) -> None:
    if isinstance(expected, dict):
        for feature, jt_e in expected.items():
            jt_got = got[feature]
            torch.testing.assert_close(jt_e.lengths(), jt_got.lengths())
            torch.testing.assert_close(jt_e.values(), jt_got.values())
            torch.testing.assert_close(jt_e.offsets(), jt_got.offsets())
    else:
        torch.testing.assert_close(expected, got)


def _model(
    num_embeddings: int,
    emb_dim: int,
    world_size: int,
    batch_size: int,
    dense_device: torch.device,
    sparse_device: torch.device,
) -> TestModelInfo:

    mi = TestModelInfo(
        dense_device=dense_device,
        sparse_device=sparse_device,
        num_features=1,
        num_float_features=8,
        num_weighted_features=1,
        topology=Topology(world_size=world_size, compute_device="cuda"),
    )

    mi.planner = EmbeddingShardingPlanner(
        topology=mi.topology,
        batch_size=batch_size,
        enumerator=EmbeddingEnumerator(
            topology=mi.topology,
            batch_size=batch_size,
            estimator=[
                EmbeddingPerfEstimator(topology=mi.topology, is_inference=True),
                EmbeddingStorageEstimator(topology=mi.topology),
            ],
        ),
    )

    mi.tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(mi.num_features)
    ]

    mi.weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(mi.num_weighted_features)
    ]

    mi.model = TorchTypesModelInputWrapper(
        TestSparseNN(
            tables=mi.tables,
            weighted_tables=mi.weighted_tables,
            num_float_features=mi.num_float_features,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
    )
    mi.model.training = False
    mi.quant_model = quantize(mi.model, inplace=False)
    return mi


def _cw_shard_qebc(
    mi: TestModelInfo,
    device: torch.device,
    expected_shards: List[Tuple[int, int, int, int]],
) -> torch.nn.Module:
    sharder = TestQuantEBCSharder(
        sharding_type=ShardingType.COLUMN_WISE.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in mi.tables],
    )
    plan = mi.planner.plan(
        mi.quant_model,
        # pyre-ignore
        [sharder],
    )
    msp: ModuleShardingPlan = plan.plan["_module.sparse.ebc"]
    ps: ParameterSharding = msp["table_0"]
    assert ps.sharding_type == ShardingType.COLUMN_WISE.value
    sharding_spec = ps.sharding_spec
    assert len(sharding_spec.shards) == len(expected_shards)
    for shard, ((offset_r, offset_c, size_r, size_c), placement) in zip(
        sharding_spec.shards, expected_shards
    ):
        assert shard.shard_offsets == [offset_r, offset_c]
        assert shard.shard_sizes == [size_r, size_c]
        assert str(shard.placement) == placement

    # We want to leave quant_model unchanged to compare the results with it
    quant_model_copy = copy.deepcopy(mi.quant_model)
    sharded_model = _shard_modules(
        module=quant_model_copy,
        sharders=[sharder],
        device=device,
        plan=plan,
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )
    return sharded_model


class InferShardingsTest(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_cw(self) -> None:
        num_embeddings = 64
        emb_dim = 512
        world_size = 2
        batch_size = 4
        local_device = torch.device("cuda:0")
        mi = _model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
        )

        non_sharded_qebc_model = mi.quant_model
        sharded_qebc_model = _cw_shard_qebc(
            mi,
            device=local_device,
            expected_shards=[
                ((0, 0, 64, 128), "rank:0/cuda:0"),
                ((0, 128, 64, 128), "rank:1/cuda:1"),
                ((0, 256, 64, 128), "rank:0/cuda:0"),
                ((0, 384, 64, 128), "rank:1/cuda:1"),
            ],
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size)
        ]
        non_sharded_named_buffers = {
            n: b for n, b in non_sharded_qebc_model.named_buffers()
        }

        # Making TBEs weights the same for non sharded and sharded models
        # TODO(ivankobzarev): Remove this hardcoded shards weight access after adding named_buffer/state_dict support for non tw modules in shardedQEBC
        w = non_sharded_named_buffers[
            "_module.sparse.ebc.embedding_bags.table_0.weight"
        ]
        wqss = non_sharded_named_buffers[
            "_module.sparse.ebc.embedding_bags.table_0.weight_qscaleshift"
        ]
        qtbes = [
            sharded_qebc_model._module.sparse.ebc._lookups[0]
            ._embedding_lookups_per_rank[rank]
            ._emb_modules[0]
            .emb_module
            for rank in range(world_size)
        ]

        for (dstw, dstq), srcw in zip(
            qtbes[0].split_embedding_weights(split_scale_shifts=True)
            + qtbes[1].split_embedding_weights(split_scale_shifts=True),
            [w[:, :128], w[:, 256:384]] + [w[:, 128:256], w[:, 384:]],
        ):
            dstq.copy_(wqss)
            dstw.copy_(srcw)

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_qebc_model(*inputs[0])

        non_sharded_output = non_sharded_qebc_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        gm = symbolic_trace(
            sharded_qebc_model,
            leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"],
        )

        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(non_sharded_output, gm_script_output)
