#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, List, Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings, Verbosity
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, EmbeddingCollection
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    EmbeddingBagCollectionSharder,
    EmbeddingCollectionSharder,
    ParameterShardingGenerator,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ParameterSharding,
    ShardedTensor,
    ShardingEnv,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class ModuleStateTest(MultiProcessTestBase):
    @staticmethod
    def _test_ebc(
        tables: List[EmbeddingBagConfig],
        rank: int,
        world_size: int,
        backend: str,
        parameter_sharding_plan: Dict[str, ParameterSharding],
        sharder: ModuleSharder[nn.Module],
        local_size: Optional[int] = None,
    ) -> None:
        with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
            model = EmbeddingBagCollection(
                tables=tables,
                device=ctx.device,
            )
            sharded_model = sharder.shard(
                module=model,
                params=parameter_sharding_plan,
                # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
                #  `Optional[ProcessGroup]`.
                env=ShardingEnv.from_process_group(ctx.pg),
                device=ctx.device,
            )
            assert isinstance(sharded_model, ShardedEmbeddingBagCollection)

            state_dict = sharded_model.state_dict()

            for state_dict_key in [
                "embedding_bags.0.weight",
                "embedding_bags.1.weight",
            ]:
                assert (
                    state_dict_key in state_dict
                ), f"Expected '{state_dict_key}' in state_dict"
                assert isinstance(
                    state_dict[state_dict_key], ShardedTensor
                ), "expected state dict to contain ShardedTensor"

            # Check that embedding modules are registered as submodules
            assert "embedding_bags" in sharded_model._modules
            assert isinstance(sharded_model._modules["embedding_bags"], nn.ModuleDict)

            # try loading state dict
            sharded_model.load_state_dict(state_dict)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        per_param_sharding=st.sampled_from(
            [
                {
                    "0": column_wise(ranks=[0, 1]),
                    "1": column_wise(ranks=[1, 0]),
                },
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_module_state_ebc(
        self,
        per_param_sharding: Dict[str, ParameterShardingGenerator],
    ) -> None:

        WORLD_SIZE = 2
        EMBEDDING_DIM = 8
        NUM_EMBEDDINGS = 4

        embedding_bag_configs = [
            EmbeddingBagConfig(
                name=str(idx),
                feature_names=[f"feature_{idx}"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=NUM_EMBEDDINGS,
            )
            for idx in per_param_sharding
        ]
        ebc = EmbeddingBagCollection(tables=embedding_bag_configs)
        sharder = EmbeddingBagCollectionSharder()

        parameter_sharding_plan = construct_module_sharding_plan(
            module=ebc,
            per_param_sharding=per_param_sharding,
            local_size=WORLD_SIZE,
            world_size=WORLD_SIZE,
            # pyre-ignore
            sharder=sharder,
        )

        self._run_multi_process_test(
            callable=self._test_ebc,
            tables=embedding_bag_configs,
            local_size=WORLD_SIZE,
            world_size=WORLD_SIZE,
            backend="nccl"
            if (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
            else "gloo",
            sharder=sharder,
            parameter_sharding_plan=parameter_sharding_plan,
        )

    @staticmethod
    def _test_ec(
        tables: List[EmbeddingConfig],
        rank: int,
        world_size: int,
        backend: str,
        parameter_sharding_plan: Dict[str, ParameterSharding],
        sharder: ModuleSharder[nn.Module],
        local_size: Optional[int] = None,
    ) -> None:
        with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
            model = EmbeddingCollection(
                tables=tables,
                device=ctx.device,
            )
            sharded_model = sharder.shard(
                module=model,
                params=parameter_sharding_plan,
                # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
                #  `Optional[ProcessGroup]`.
                env=ShardingEnv.from_process_group(ctx.pg),
                device=ctx.device,
            )
            assert isinstance(sharded_model, ShardedEmbeddingCollection)

            state_dict = sharded_model.state_dict()

            for state_dict_key in [
                "embeddings.0.weight",
                "embeddings.1.weight",
            ]:
                assert (
                    state_dict_key in state_dict
                ), f"Expected '{state_dict_key}' in state_dict"
                assert isinstance(
                    state_dict[state_dict_key], ShardedTensor
                ), "expected state dict to contain ShardedTensor"

            # Check that embedding modules are registered as submodules
            assert "embeddings" in sharded_model._modules
            assert isinstance(sharded_model._modules["embeddings"], nn.ModuleDict)

            # try loading state dict
            sharded_model.load_state_dict(state_dict)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        per_param_sharding=st.sampled_from(
            [
                {
                    "0": column_wise(ranks=[0, 1]),
                    "1": column_wise(ranks=[1, 0]),
                },
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_module_state_ec(
        self,
        per_param_sharding: Dict[str, ParameterShardingGenerator],
    ) -> None:

        WORLD_SIZE = 2
        EMBEDDING_DIM = 8
        NUM_EMBEDDINGS = 4

        embedding_configs = [
            EmbeddingConfig(
                name=str(idx),
                feature_names=[f"feature_{idx}"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=NUM_EMBEDDINGS,
            )
            for idx in per_param_sharding
        ]
        ebc = EmbeddingCollection(tables=embedding_configs)
        sharder = EmbeddingCollectionSharder()

        parameter_sharding_plan = construct_module_sharding_plan(
            module=ebc,
            per_param_sharding=per_param_sharding,
            local_size=WORLD_SIZE,
            world_size=WORLD_SIZE,
            # pyre-ignore
            sharder=sharder,
        )

        self._run_multi_process_test(
            callable=self._test_ec,
            tables=embedding_configs,
            local_size=WORLD_SIZE,
            world_size=WORLD_SIZE,
            backend="nccl"
            if (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
            else "gloo",
            sharder=sharder,
            parameter_sharding_plan=parameter_sharding_plan,
        )
