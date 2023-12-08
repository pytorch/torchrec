#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import math
import unittest
from typing import Dict, List, Tuple

import hypothesis.strategies as st

import torch
from hypothesis import given, settings
from torchrec import (
    EmbeddingBagConfig,
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.quant_embeddingbag import (
    QuantFeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.quant_state import sharded_tbes_weights_spec, WeightSpec
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    placement,
    row_wise,
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
    quantize_fpebc,
    shard_qebc,
    shard_qec,
    TestModelInfo,
    TestQuantEBCSharder,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import ShardingEnv, ShardingPlan
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import FeatureProcessorsCollection
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection

torch.fx.wrap("len")


class TimeGapPoolingCollectionModule(FeatureProcessorsCollection):
    def __init__(
        self,
        feature_pow: float,
        feature_min: float,
        feature_max: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.feature_pow = feature_pow
        self.device = device

        param = torch.empty(
            [math.ceil(math.pow(feature_max, feature_pow)) + 2],
            device=device,
        )
        self.register_buffer("w", param)

    def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        scores_list = []
        for feature_name in features.keys():
            jt = features[feature_name]
            scores = jt.weights()
            scores = torch.clamp(
                scores,
                min=self.feature_min,
                max=self.feature_max,
            )
            indices = torch.floor(torch.pow(scores, self.feature_pow))
            indices = indices.to(torch.int32)
            scores = torch.index_select(self.w, 0, indices)
            scores_list.append(scores)

        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values(),
            weights=torch.cat(scores_list)
            if scores_list
            else features.weights_or_none(),
            lengths=features.lengths(),
            stride=features.stride(),
        )


def placement_helper(device_type: str, index: int = 0) -> str:
    if device_type == "cpu":
        return f"rank:0/{device_type}"  # cpu only use rank 0

    return f"rank:{index}/{device_type}:{index}"


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

    @unittest.skipIf(
        torch.cuda.device_count() <= 2,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        uneven_shard_pattern=st.sampled_from(
            [
                (512, 256, 128, 128),
                (500, 256, 128, 128),
            ]
        ),
        device=st.sampled_from(["cuda"]),  # TODO: add cpu test when it's fixed
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_uneven_sharding(
        self,
        weight_dtype: torch.dtype,
        uneven_shard_pattern: Tuple[int, int, int, int],
        device: str,
    ) -> None:
        num_embeddings, size0, size1, size2 = uneven_shard_pattern
        size2 = min(size2, num_embeddings - size0 - size1)
        emb_dim = 64
        local_size = 3
        world_size = 3
        batch_size = 4
        local_device = torch.device("cuda:0" if device == "cuda" else device)
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
            num_features=1,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, size0, 64),
                    "rank:0/cpu" if device == "cpu" else "rank:0/cuda:0",
                ),
                (
                    (size0, 0, size1, 64),
                    "rank:0/cpu" if device == "cpu" else "rank:1/cuda:1",
                ),
                (
                    (size0 + size1, 0, size2, 64),
                    "rank:0/cpu" if device == "cpu" else "rank:2/cuda:2",
                ),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.ROW_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": row_wise(([size0, size1, size2], "cuda")),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device=st.sampled_from(["cuda"]),  # TODO: add cpu test when it's fixed
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_uneven_sharding_mutiple_table(
        self,
        weight_dtype: torch.dtype,
        device: str,
    ) -> None:
        num_embeddings = 512
        emb_dim = 64
        local_size = 4
        world_size = 4
        batch_size = 1
        local_device = torch.device("cuda:0" if device == "cuda" else device)
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
            num_features=4,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, 256, 64),
                    placement_helper(device, 0),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (384, 0, 64, 64),
                    placement_helper(device, 2),
                ),
                (
                    (448, 0, 64, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 128, 64),
                    placement_helper(device, 0),
                ),
                (
                    (128, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (384, 0, 128, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 256, 64),
                    placement_helper(device, 0),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (384, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (512, 0, 0, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 0, 64),
                    placement_helper(device, 0),
                ),
                (
                    (0, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (128, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (256, 0, 256, 64),
                    placement_helper(device, 3),
                ),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.ROW_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": row_wise(
                    ([256, 128, 64, 64], device),
                ),
                "table_1": row_wise(([128, 128, 128, 128], device)),
                "table_2": row_wise(([256, 128, 128, 0], device)),
                "table_3": row_wise(([0, 128, 128, 256], device)),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        _ = gm_script(*inputs[0])
        # TODO (drqiangzhang): Add comparison between scripted and nonscripted model outputs

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8]),
    )
    @settings(max_examples=1, deadline=None)
    def test_sharded_quant_fp_ebc_tw(self, weight_dtype: torch.dtype) -> None:
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
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                FeatureProcessedEmbeddingBagCollection(
                    EmbeddingBagCollection(
                        tables=mi.tables,
                        is_weighted=True,
                        device=mi.sparse_device,
                    ),
                    TimeGapPoolingCollectionModule(
                        feature_pow=1.0,
                        feature_min=-1.0,
                        feature_max=1.0,
                        device=mi.sparse_device,
                    ),
                )
            )
        )
        model_inputs: List[ModelInput] = prep_inputs(
            mi, world_size, batch_size, long_indices=False
        )
        inputs = []
        for model_input in model_inputs:
            kjt = model_input.idlist_features
            kjt = kjt.to(local_device)
            weights = torch.rand(
                kjt._values.size(0), dtype=torch.float, device=local_device
            )
            inputs.append(
                (
                    kjt._keys,
                    kjt._values,
                    weights,
                    kjt._lengths,
                    kjt._offsets,
                )
            )

        mi.model(*inputs[0])
        print(f"model:\n{mi.model}")

        mi.quant_model = quantize_fpebc(
            module=mi.model,
            inplace=False,
            register_tbes=True,
            quant_state_dict_split_scale_bias=False,
            weight_dtype=weight_dtype,
        )
        quant_model = mi.quant_model
        print(f"quant_model:\n{quant_model}")
        non_sharded_output = mi.quant_model(*inputs[0])

        topology: Topology = Topology(world_size=world_size, compute_device="cuda")
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
        sharder = QuantFeatureProcessedEmbeddingBagCollectionSharder()
        # pyre-ignore
        plan = mi.planner.plan(
            mi.quant_model,
            [sharder],
        )

        sharded_model = _shard_modules(
            module=quant_model,
            # pyre-ignore
            sharders=[sharder],
            device=local_device,
            plan=plan,
            # pyre-ignore
            env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
        )
        print(f"sharded_model:\n{sharded_model}")
        for n, m in sharded_model.named_modules():
            print(f"sharded_model.MODULE[{n}]:{type(m)}")

        # Check that FP is registered as module
        count_registered_fp: int = 0
        for _, m in sharded_model.named_modules():
            if isinstance(m, TimeGapPoolingCollectionModule):
                count_registered_fp += 1

        assert count_registered_fp == world_size

        sharded_output = sharded_model(*inputs[0])
        # TODO(ivankobzarev): check the correctness of non_sharded vs sharded
        # assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(
            sharded_model,
            leaf_modules=[
                "TimeGapPoolingCollectionModule",
                "IntNBitTableBatchedEmbeddingBagsCodegen",
            ],
        )

        # Check that FP was traced as a call_module
        fp_call_module: int = 0
        for node in gm.graph.nodes:
            if node.op == "call_module":
                m = gm
                for attr in node.target.split("."):
                    m = getattr(m, attr)
                if isinstance(m, TimeGapPoolingCollectionModule):
                    fp_call_module += 1

        assert fp_call_module == world_size
        print(f"fx.graph:\n{gm.graph}")

        gm_script = torch.jit.script(gm)
        print(f"gm_script:\n{gm_script}")
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)
        _ = gm_script(*inputs[0])
        # TODO (drqiangzhang): Add comparison between scripted and nonscripted model outputs
