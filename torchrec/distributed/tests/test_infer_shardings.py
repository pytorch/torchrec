#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import copy

import unittest
from typing import Dict, List, Tuple

import torch
from torch.distributed._shard.sharding_spec import ShardingSpec
from torchrec import EmbeddingBagConfig, EmbeddingCollection, EmbeddingConfig
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.quant_state import sharded_tbes_weights_spec, WeightSpec
from torchrec.distributed.shard import _shard_modules

from torchrec.distributed.test_utils.infer_utils import (
    KJTInputWrapper,
    model_input_to_forward_args,
    model_input_to_forward_args_kjt,
    prep_inputs,
    quantize,
    TestModelInfo,
    TestQuantEBCSharder,
    TestQuantECSharder,
    TorchTypesModelInputWrapper,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    ModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
)
from torchrec.fx import symbolic_trace


# pyre-ignore
def assert_close(expected, actual) -> None:
    if isinstance(expected, dict):
        assert list(expected.keys()) == list(actual.keys())
        for feature, jt_e in expected.items():
            jt_got = actual[feature]
            assert_close(jt_e.lengths(), jt_got.lengths())
            assert_close(jt_e.values(), jt_got.values())
            assert_close(jt_e.offsets(), jt_got.offsets())
    else:
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            if actual.device != expected.device:
                actual = actual.to(expected.device)

        torch.testing.assert_close(expected, actual)


def assert_weight_spec(
    weights_spec: Dict[str, WeightSpec],
    expected_shards: List[Tuple[Tuple[int, int, int, int], str]],
    ebc_fqn: str,
    weights_prefix: str,
    table_names: List[str],
    sharding_type: str,
) -> None:
    tbe_table_idxs = [0, 0]
    for table_name in table_names:
        unsharded_weight_fqn = f"{ebc_fqn}.{weights_prefix}.{table_name}.weight"
        for (offset_r, offset_c, size_r, size_c), placement in expected_shards:
            tbe_idx: int = 0
            if "rank:1/cuda:1" == placement:
                tbe_idx = 1
            sharded_weight_fqn: str = f"{ebc_fqn}.tbes.{tbe_idx}.{tbe_table_idxs[tbe_idx]}.{table_name}.weight"
            tbe_table_idxs[tbe_idx] += 1
            assert sharded_weight_fqn in weights_spec
            wspec = weights_spec[sharded_weight_fqn]
            assert wspec.fqn == unsharded_weight_fqn
            assert wspec.shard_sizes == [size_r, size_c]
            assert wspec.shard_offsets == [offset_r, offset_c]
            assert wspec.sharding_type == sharding_type

            for qcomp in ["qscale", "qbias"]:
                sharded_weight_qcomp_fqn: str = f"{sharded_weight_fqn}_{qcomp}"
                assert sharded_weight_qcomp_fqn in weights_spec
                wqcomp_spec = weights_spec[sharded_weight_qcomp_fqn]
                assert wqcomp_spec.fqn == f"{unsharded_weight_fqn}_{qcomp}"
                assert wqcomp_spec.shard_sizes == [size_r, 2]
                assert wqcomp_spec.shard_offsets == [offset_r, 0]
                assert wqcomp_spec.sharding_type == sharding_type


def _model(
    num_embeddings: int,
    emb_dim: int,
    world_size: int,
    batch_size: int,
    dense_device: torch.device,
    sparse_device: torch.device,
    quant_state_dict_split_scale_bias: bool = False,
) -> TestModelInfo:
    topology: Topology = Topology(world_size=world_size, compute_device="cuda")
    mi = TestModelInfo(
        dense_device=dense_device,
        sparse_device=sparse_device,
        num_features=1,
        num_float_features=8,
        num_weighted_features=1,
        topology=topology,
    )

    mi.planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            estimator=[
                EmbeddingPerfEstimator(topology=topology, is_inference=True),
                EmbeddingStorageEstimator(topology=topology),
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
    mi.quant_model = quantize(
        module=mi.model,
        inplace=False,
        quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
    )
    return mi


def _shard_qebc(
    mi: TestModelInfo,
    sharding_type: ShardingType,
    device: torch.device,
    expected_shards: List[Tuple[Tuple[int, int, int, int], str]],
) -> torch.nn.Module:
    sharder = TestQuantEBCSharder(
        sharding_type=sharding_type.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in mi.tables],
    )
    # pyre-ignore
    plan = mi.planner.plan(
        mi.quant_model,
        [sharder],
    )
    msp: ModuleShardingPlan = plan.plan["_module.sparse.ebc"]
    # pyre-ignore
    ps: ParameterSharding = msp["table_0"]
    assert ps.sharding_type == sharding_type.value
    assert ps.sharding_spec is not None
    sharding_spec: ShardingSpec = ps.sharding_spec
    # pyre-ignore
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
        # pyre-ignore
        sharders=[sharder],
        device=device,
        plan=plan,
        # pyre-ignore
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )
    return sharded_model


def _shard_qec(
    mi: TestModelInfo,
    sharding_type: ShardingType,
    device: torch.device,
    expected_shards: List[Tuple[Tuple[int, int, int, int], str]],
) -> torch.nn.Module:
    sharder = TestQuantECSharder(
        sharding_type=sharding_type.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
    )
    # pyre-ignore
    plan = mi.planner.plan(
        mi.quant_model,
        [sharder],
    )
    msp: ModuleShardingPlan = plan.plan["_module_kjt_input.0"]  # TODO: hardcoded
    # pyre-ignore
    ps: ParameterSharding = msp["table_0"]
    assert ps.sharding_type == sharding_type.value
    assert ps.sharding_spec is not None
    sharding_spec: ShardingSpec = ps.sharding_spec
    # pyre-ignore
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
        # pyre-ignore
        sharders=[sharder],
        device=device,
        plan=plan,
        # pyre-ignore
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )
    return sharded_model


class InferShardingsTest(unittest.TestCase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_rw(self) -> None:
        num_embeddings = 256
        emb_dim = 12
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
            quant_state_dict_split_scale_bias=True,
        )

        non_sharded_model = mi.quant_model
        num_emb_half = num_embeddings // 2
        expected_shards = [
            ((0, 0, num_emb_half, emb_dim), "rank:0/cuda:0"),
            ((num_emb_half, 0, num_emb_half, emb_dim), "rank:1/cuda:1"),
        ]
        sharded_model = _shard_qebc(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())

        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0"],
            ShardingType.ROW_WISE.value,
        )

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_cw(self) -> None:
        num_embeddings = 64
        emb_dim = 512
        emb_dim_4 = 512 // 4
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
            quant_state_dict_split_scale_bias=True,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            ((0, 0, num_embeddings, emb_dim_4), "rank:0/cuda:0"),
            ((0, 1 * emb_dim_4, num_embeddings, emb_dim_4), "rank:1/cuda:1"),
            ((0, 2 * emb_dim_4, num_embeddings, emb_dim_4), "rank:0/cuda:0"),
            ((0, 3 * emb_dim_4, num_embeddings, emb_dim_4), "rank:1/cuda:1"),
        ]
        sharded_model = _shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())
        # torchrec.distributed.test_utils.test_sharding.copy_state_dict(sharded_model.state_dict(), non_sharded_model.state_dict()) does not work for CW due to non-trivial qscaleshift copy which is handled in shardedQEBC load_state_dict

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0"],
            ShardingType.COLUMN_WISE.value,
        )

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_cw_sequence(self) -> None:
        num_embeddings = 4
        emb_dim = 512
        emb_dim_4 = emb_dim // 4
        world_size = 2
        batch_size = 2
        local_device = torch.device("cuda:0")

        topology: Topology = Topology(world_size=world_size, compute_device="cuda")
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model, inplace=False, quant_state_dict_split_scale_bias=True
        )
        non_sharded_model = mi.quant_model
        expected_shards = [
            ((0, 0, num_embeddings, emb_dim_4), "rank:0/cuda:0"),
            ((0, 1 * emb_dim_4, num_embeddings, emb_dim_4), "rank:1/cuda:1"),
            ((0, 2 * emb_dim_4, num_embeddings, emb_dim_4), "rank:0/cuda:0"),
            ((0, 3 * emb_dim_4, num_embeddings, emb_dim_4), "rank:1/cuda:1"),
        ]
        sharded_model = _shard_qec(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,  # column wise sharding the model
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module_kjt_input.0",
            "embeddings",
            ["table_0", "table_1"],
            ShardingType.COLUMN_WISE.value,
        )

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_rw_sequence(self) -> None:
        num_embeddings = 10
        emb_dim = 4
        world_size = 2
        batch_size = 2
        local_device = torch.device("cuda:0")

        topology: Topology = Topology(world_size=world_size, compute_device="cuda")
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model, inplace=False, quant_state_dict_split_scale_bias=True
        )
        non_sharded_model = mi.quant_model
        num_emb_half = num_embeddings // 2
        expected_shards = [
            ((0, 0, num_emb_half, emb_dim), "rank:0/cuda:0"),
            ((num_emb_half, 0, num_emb_half, emb_dim), "rank:1/cuda:1"),
        ]
        sharded_model = _shard_qec(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module_kjt_input.0",
            "embeddings",
            ["table_0", "table_1"],
            ShardingType.ROW_WISE.value,
        )
