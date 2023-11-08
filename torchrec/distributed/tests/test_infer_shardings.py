#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import copy

import unittest
from typing import Dict, List, Optional, Tuple

import hypothesis.strategies as st

import torch
from hypothesis import given, settings
from torchrec import EmbeddingBagConfig, EmbeddingCollection, EmbeddingConfig
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.quant_state import sharded_tbes_weights_spec, WeightSpec
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
)
from torchrec.distributed.test_utils.infer_utils import (
    assert_close,
    assert_weight_spec,
    create_cw_min_partition_constraints,
    create_test_model,
    KJTInputWrapper,
    model_input_to_forward_args,
    model_input_to_forward_args_kjt,
    prep_inputs,
    quantize,
    shard_qebc,
    shard_qec,
    TestModelInfo,
    TestQuantEBCSharder,
    TestQuantECSharder,
    TorchTypesModelInputWrapper,
)
from torchrec.distributed.types import (
    ModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.fx import symbolic_trace


class InferShardingsTest(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw(self, weight_dtype: torch.dtype) -> None:
        num_embeddings = 256
        emb_dim = 16
        world_size = 2
        batch_size = 4
        local_device = torch.device("cuda:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        num_emb_half = num_embeddings // 2
        expected_shards = [
            [
                ((0, 0, num_emb_half, emb_dim), "rank:0/cuda:0"),
                ((num_emb_half, 0, num_emb_half, emb_dim), "rank:1/cuda:1"),
            ]
        ]
        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        test_permute=st.booleans(),
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw(self, test_permute: bool, weight_dtype: torch.dtype) -> None:
        test_permute = False
        num_embeddings = 64
        emb_dim = 512
        emb_dim_4 = emb_dim // 4
        local_size = 2
        world_size = 2
        batch_size = 4
        local_device = torch.device("cuda:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim_4),
                    "rank:0/cuda:0" if not test_permute else "rank:1/cuda:1",
                ),
                (
                    (0, 1 * emb_dim_4, num_embeddings, emb_dim_4),
                    "rank:1/cuda:1" if not test_permute else "rank:0/cuda:0",
                ),
                (
                    (0, 2 * emb_dim_4, num_embeddings, emb_dim_4),
                    "rank:0/cuda:0" if not test_permute else "rank:1/cuda:1",
                ),
                (
                    (0, 3 * emb_dim_4, num_embeddings, emb_dim_4),
                    "rank:1/cuda:1" if not test_permute else "rank:0/cuda:0",
                ),
            ]
        ]

        plan = None
        if test_permute:
            sharder = TestQuantEBCSharder(
                sharding_type=ShardingType.COLUMN_WISE.value,
                kernel_type=EmbeddingComputeKernel.QUANT.value,
                shardable_params=[table.name for table in mi.tables],
            )

            module_plan = construct_module_sharding_plan(
                non_sharded_model._module.sparse.ebc,
                per_param_sharding={
                    "table_0": column_wise(ranks=[1, 0, 1, 0]),
                },
                # pyre-ignore
                sharder=sharder,
                local_size=local_size,
                world_size=world_size,
            )

            plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        emb_dim=st.sampled_from([192, 128]),
        test_permute=st.booleans(),
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_with_smaller_emb_dim(
        self, emb_dim: int, test_permute: bool, weight_dtype: torch.dtype
    ) -> None:
        num_embeddings = 64
        emb_dim_4 = emb_dim // 4
        world_size = 2
        batch_size = 4
        local_device = torch.device("cuda:0")
        constraints = create_cw_min_partition_constraints([("table_0", emb_dim_4)])
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            constraints=constraints,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim_4),
                    "rank:0/cuda:0",
                ),
                (
                    (0, 1 * emb_dim_4, num_embeddings, emb_dim_4),
                    "rank:1/cuda:1",
                ),
                (
                    (0, 2 * emb_dim_4, num_embeddings, emb_dim_4),
                    "rank:0/cuda:0",
                ),
                (
                    (0, 3 * emb_dim_4, num_embeddings, emb_dim_4),
                    "rank:1/cuda:1",
                ),
            ]
        ]

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_multiple_tables_with_permute(self, weight_dtype: torch.dtype) -> None:
        num_embeddings = 64
        emb_dim = 512
        emb_dim_2 = 512 // 2
        local_size = 2
        world_size = 2
        batch_size = 4
        local_device = torch.device("cuda:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=2,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                ((0, 0, num_embeddings, emb_dim_2), "rank:1/cuda:1"),
                ((0, 1 * emb_dim_2, num_embeddings, emb_dim_2), "rank:0/cuda:0"),
            ],
            [
                ((0, 0, num_embeddings, emb_dim_2), "rank:0/cuda:0"),
                ((0, 1 * emb_dim_2, num_embeddings, emb_dim_2), "rank:1/cuda:1"),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.COLUMN_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": column_wise(ranks=[1, 0]),
                "table_1": column_wise(ranks=[0, 1]),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
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
            ["table_0", "table_1"],
            ShardingType.COLUMN_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_irregular_shard_placement(self, weight_dtype: torch.dtype) -> None:
        num_embeddings = 64
        emb_dim = 384
        emb_dim_2 = emb_dim // 2
        emb_dim_3 = emb_dim // 3
        local_size = 4
        world_size = 4
        batch_size = 4
        local_device = torch.device("cuda:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=3,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                ((0, 0, num_embeddings, emb_dim_2), "rank:2/cuda:2"),
                ((0, 1 * emb_dim_2, num_embeddings, emb_dim_2), "rank:1/cuda:1"),
            ],
            [
                ((0, 0, num_embeddings, emb_dim_2), "rank:0/cuda:0"),
                ((0, 1 * emb_dim_2, num_embeddings, emb_dim_2), "rank:3/cuda:3"),
            ],
            [
                ((0, 0, num_embeddings, emb_dim_3), "rank:0/cuda:0"),
                ((0, 1 * emb_dim_3, num_embeddings, emb_dim_3), "rank:2/cuda:2"),
                ((0, 2 * emb_dim_3, num_embeddings, emb_dim_3), "rank:3/cuda:3"),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.COLUMN_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": column_wise(ranks=[2, 1]),
                "table_1": column_wise(ranks=[0, 3]),
                "table_2": column_wise(ranks=[0, 2, 3]),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_sequence(self, weight_dtype: torch.dtype) -> None:
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
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                ((0, 0, num_embeddings, emb_dim_4), "rank:0/cuda:0"),
                ((0, 1 * emb_dim_4, num_embeddings, emb_dim_4), "rank:1/cuda:1"),
                ((0, 2 * emb_dim_4, num_embeddings, emb_dim_4), "rank:0/cuda:0"),
                ((0, 3 * emb_dim_4, num_embeddings, emb_dim_4), "rank:1/cuda:1"),
            ],
        ] * 2
        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,  # column wise sharding the model
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_sequence(self, weight_dtype: torch.dtype) -> None:
        num_embeddings = 10
        emb_dim = 16
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
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model
        num_emb_half = num_embeddings // 2
        expected_shards = [
            [
                ((0, 0, num_emb_half, emb_dim), "rank:0/cuda:0"),
                ((num_emb_half, 0, num_emb_half, emb_dim), "rank:1/cuda:1"),
            ],
        ] * 2
        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
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
