#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import unittest
from typing import Dict, List, Tuple

import torch
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
    assert_close,
    assert_weight_spec,
    create_test_model,
    model_input_to_forward_args,
    prep_inputs,
    quantize,
    shard_qebc,
    TestQuantEBCSharder,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    MODULE_ATTR_EMB_CONFIG_NAME_TO_PRUNING_INDICES_REMAPPING_DICT,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def prune_and_quantize_model(
    model: torch.nn.Module, pruning_ebc_dict: Dict[str, torch.Tensor]
) -> torch.nn.Module:
    ebc = model.get_submodule("_module.sparse.ebc")
    setattr(
        ebc,
        MODULE_ATTR_EMB_CONFIG_NAME_TO_PRUNING_INDICES_REMAPPING_DICT,
        pruning_ebc_dict,
    )

    quant_state_dict_split_scale_bias = True
    quant_model = quantize(
        module=model,
        inplace=False,
        quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
    )

    return quant_model


class QuantPruneTest(unittest.TestCase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_qebc_pruned_tw(self) -> None:
        batch_size: int = 4
        world_size = 2
        local_device = torch.device("cuda:0")

        num_embedding = 100
        emb_dim = 64
        pruned_entry = 40

        # hash, dim, pruned_hash
        table_specs: List[Tuple[int, int, int]] = [
            (num_embedding, emb_dim, num_embedding),
            (num_embedding, emb_dim, num_embedding - pruned_entry),
        ]

        mi = create_test_model(
            num_embedding,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=len(table_specs),
        )

        pruning_ebc_dict: Dict[str, torch.Tensor] = {}
        table_1_spec = table_specs[1]
        table_1_num_emb: int = table_1_spec[0]
        table_1_pruned_num_emb: int = table_1_spec[2]
        table_1_remapping_indices = torch.full(
            fill_value=-1, size=[table_1_num_emb], dtype=torch.int32
        )
        table_1_remapping_indices[-table_1_pruned_num_emb:] = torch.arange(
            table_1_pruned_num_emb, dtype=torch.int32
        )
        pruning_ebc_dict["table_1"] = table_1_remapping_indices

        quant_model = prune_and_quantize_model(mi.model, pruning_ebc_dict)
        mi.quant_model = quant_model

        expected_shards = [
            [
                (
                    (0, 0, table_specs[0][2], table_specs[0][1]),
                    "rank:0/cuda:0",
                ),
            ],
            [
                (
                    (0, 0, table_specs[1][2], table_specs[1][1]),
                    "rank:1/cuda:1",
                ),
            ],
        ]

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.TABLE_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size)
        ]
        sharded_model.load_state_dict(quant_model.state_dict())
        quant_output = quant_model(*inputs[0])
        sharded_output = sharded_model(*inputs[0])
        assert_close(quant_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        buffer = io.BytesIO()
        torch.jit.save(gm_script, buffer)
        buffer.seek(0)
        loaded_gm_script = torch.jit.load(buffer)
        gm_script_output = loaded_gm_script(*inputs[0])
        assert_close(quant_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0", "table_1"],
            ShardingType.TABLE_WISE.value,
        )

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_qebc_pruned_tw_one_ebc(self) -> None:
        batch_size: int = 1
        # hash, dim, pruned_hash
        table_specs: List[Tuple[int, int, int]] = [
            (200, 10, 100),
        ]
        world_size = 2
        local_device = torch.device("cuda:0")
        topology: Topology = Topology(
            world_size=world_size, compute_device=local_device.type
        )

        tables = [
            EmbeddingBagConfig(
                num_embeddings=num_emb,
                embedding_dim=emb_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i, (num_emb, emb_dim, _) in enumerate(table_specs)
        ]

        # quantize api only work with submodule. Wrap ebc inside nn.Sequential
        model = torch.nn.Sequential(
            EmbeddingBagCollection(
                tables=tables,
                device=local_device,
            )
        )
        model.to(local_device)
        model.training = False

        pruning_ebc_dict: Dict[str, torch.Tensor] = {}
        table_0_spec = table_specs[0]
        table_0_num_emb: int = table_0_spec[0]
        table_0_emb_dim: int = table_0_spec[1]
        table_0_remapping_indices = torch.full(
            fill_value=-1, size=[table_0_num_emb], dtype=torch.int32
        )

        # Prune element at even index position
        for i in range(200):
            if i % 2 == 0:
                continue
            table_0_remapping_indices[i] = i // 2

        pruning_ebc_dict["table_0"] = table_0_remapping_indices

        setattr(
            model[0],
            MODULE_ATTR_EMB_CONFIG_NAME_TO_PRUNING_INDICES_REMAPPING_DICT,
            pruning_ebc_dict,
        )

        quant_state_dict_split_scale_bias = True
        quant_model = quantize(
            module=model,
            inplace=False,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

        quant_model = quant_model[0]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.TABLE_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in tables],
        )

        quant_model_copy = copy.deepcopy(quant_model)
        topology: Topology = Topology(world_size=world_size, compute_device="cuda")

        planner = EmbeddingShardingPlanner(
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
        plan = planner.plan(
            quant_model_copy,
            # pyre-ignore
            [sharder],
        )

        sharded_model = _shard_modules(
            module=quant_model_copy,
            # pyre-ignore
            sharders=[sharder],
            device=local_device,
            plan=plan,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )
        sharded_model.load_state_dict(quant_model.state_dict())

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor([0, 1, 2]).cuda(),
            lengths=torch.LongTensor([1, 1, 1]).cuda(),
            weights=None,
        )

        q_output = quant_model(kjt)
        s_output = sharded_model(kjt)

        assert_close(q_output["feature_0"], s_output["feature_0"])

        assert_close(q_output["feature_0"][0], torch.tensor([0.0] * table_0_emb_dim))
        assert_close(q_output["feature_0"][2], torch.tensor([0.0] * table_0_emb_dim))

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_qebc_pruned_cw(self) -> None:
        batch_size: int = 4
        world_size = 2
        local_device = torch.device("cuda:0")

        num_embedding = 200
        emb_dim = 512
        pruned_entry = 100

        # hash, dim, pruned_hash
        table_specs: List[Tuple[int, int, int]] = [
            (num_embedding, emb_dim, num_embedding - pruned_entry),
        ]

        mi = create_test_model(
            num_embedding,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=len(table_specs),
        )

        pruning_ebc_dict: Dict[str, torch.Tensor] = {}
        table_0_spec = table_specs[0]
        table_0_num_emb: int = table_0_spec[0]
        table_0_pruned_num_emb: int = table_0_spec[2]
        table_0_remapping_indices = torch.full(
            fill_value=-1, size=[table_0_num_emb], dtype=torch.int32
        )
        table_0_remapping_indices[-table_0_pruned_num_emb:] = torch.arange(
            table_0_pruned_num_emb, dtype=torch.int32
        )
        pruning_ebc_dict["table_0"] = table_0_remapping_indices

        quant_model = prune_and_quantize_model(mi.model, pruning_ebc_dict)
        mi.quant_model = quant_model

        expected_shards = [
            [
                (
                    (0, 0, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:0/cuda:0",
                ),
                (
                    (0, 128, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:1/cuda:1",
                ),
                (
                    (0, 256, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:0/cuda:0",
                ),
                (
                    (0, 384, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:1/cuda:1",
                ),
            ],
        ]

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size)
        ]
        sharded_model.load_state_dict(quant_model.state_dict())
        quant_output = quant_model(*inputs[0])
        sharded_output = sharded_model(*inputs[0])
        assert_close(quant_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        buffer = io.BytesIO()
        torch.jit.save(gm_script, buffer)
        buffer.seek(0)
        loaded_gm_script = torch.jit.load(buffer)
        gm_script_output = loaded_gm_script(*inputs[0])
        assert_close(quant_output, gm_script_output)

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
    def test_qebc_pruned_cw_one_ebc(self) -> None:
        batch_size: int = 1
        # hash, dim, pruned_hash
        table_specs: List[Tuple[int, int, int]] = [
            (200, 512, 100),
        ]
        world_size = 2
        local_device = torch.device("cuda:0")
        topology: Topology = Topology(
            world_size=world_size, compute_device=local_device.type
        )

        tables = [
            EmbeddingBagConfig(
                num_embeddings=num_emb,
                embedding_dim=emb_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i, (num_emb, emb_dim, _) in enumerate(table_specs)
        ]

        # quantize api only work with submodule. Wrap ebc inside nn.Sequential
        model = torch.nn.Sequential(
            EmbeddingBagCollection(
                tables=tables,
                device=local_device,
            )
        )
        model.to(local_device)
        model.training = False

        pruning_ebc_dict: Dict[str, torch.Tensor] = {}
        table_0_spec = table_specs[0]
        table_0_num_emb: int = table_0_spec[0]
        table_0_emb_dim: int = table_0_spec[1]
        table_0_remapping_indices = torch.full(
            fill_value=-1, size=[table_0_num_emb], dtype=torch.int32
        )

        # Prune element at even index position
        for i in range(200):
            if i % 2 == 0:
                continue
            table_0_remapping_indices[i] = i // 2

        pruning_ebc_dict["table_0"] = table_0_remapping_indices

        setattr(
            model[0],
            MODULE_ATTR_EMB_CONFIG_NAME_TO_PRUNING_INDICES_REMAPPING_DICT,
            pruning_ebc_dict,
        )

        quant_state_dict_split_scale_bias = True
        quant_model = quantize(
            module=model,
            inplace=False,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

        quant_model = quant_model[0]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.COLUMN_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in tables],
        )

        quant_model_copy = copy.deepcopy(quant_model)
        topology: Topology = Topology(world_size=world_size, compute_device="cuda")

        planner = EmbeddingShardingPlanner(
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
        plan = planner.plan(
            quant_model_copy,
            # pyre-ignore
            [sharder],
        )

        sharded_model = _shard_modules(
            module=quant_model_copy,
            # pyre-ignore
            sharders=[sharder],
            device=local_device,
            plan=plan,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )
        sharded_model.load_state_dict(quant_model.state_dict())

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor([0, 1, 2, 197, 198, 199]).cuda(),
            lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]).cuda(),
            weights=None,
        )

        q_output = quant_model(kjt)
        s_output = sharded_model(kjt)

        assert_close(q_output["feature_0"], s_output["feature_0"])

        assert_close(q_output["feature_0"][0], torch.tensor([0.0] * table_0_emb_dim))
        assert_close(q_output["feature_0"][2], torch.tensor([0.0] * table_0_emb_dim))
        assert_close(q_output["feature_0"][4], torch.tensor([0.0] * table_0_emb_dim))
