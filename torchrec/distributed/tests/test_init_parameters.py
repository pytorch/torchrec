#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List, Optional, Union

import torch
from hypothesis import given, settings, strategies as st, Verbosity
from torch import nn
from torch.distributed._tensor import DTensor
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    data_parallel,
    ParameterShardingGenerator,
    row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.test_utils import skip_if_asan_class


def initialize_and_test_parameters(
    rank: int,
    world_size: int,
    backend: str,
    embedding_tables: Union[EmbeddingCollection, EmbeddingBagCollection],
    sharding_type: str,
    sharders: List[ModuleSharder[nn.Module]],
    table_name: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Set seed again in each process to ensure consistency
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        key = (
            f"embeddings.{table_name}.weight"
            if isinstance(embedding_tables, EmbeddingCollection)
            else f"embedding_bags.{table_name}.weight"
        )

        # Create the same fixed tensor in each process
        fixed_tensor = torch.randn(10, 64, generator=torch.Generator().manual_seed(42))

        # Load the fixed tensor into the embedding_tables to ensure consistency
        embedding_tables.load_state_dict({key: fixed_tensor})

        # Store the original tensor on CPU for comparison BEFORE creating the model
        original_tensor = embedding_tables.state_dict()[key].clone().cpu()

        module_sharding_plan = construct_module_sharding_plan(
            embedding_tables,
            per_param_sharding={
                table_name: _select_sharding_type(sharding_type),
            },
            local_size=ctx.local_size,
            world_size=ctx.world_size,
            device_type=ctx.device.type,
        )

        model = DistributedModelParallel(
            module=embedding_tables,
            plan=ShardingPlan({"": module_sharding_plan}),
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=sharders,
            device=ctx.device,
            init_data_parallel=False,
            init_parameters=False,
        )

        if isinstance(model.state_dict()[key], DTensor):
            if ctx.rank == 0:
                gathered_tensor = torch.empty(model.state_dict()[key].size())
            else:
                gathered_tensor = None

            gathered_tensor = model.state_dict()[key].full_tensor()
            if ctx.rank == 0:
                torch.testing.assert_close(
                    gathered_tensor.cpu(), original_tensor, rtol=1e-5, atol=1e-6
                )
        elif isinstance(model.state_dict()[key], ShardedTensor):
            if ctx.rank == 0:
                gathered_tensor = torch.empty_like(original_tensor, device=ctx.device)
            else:
                gathered_tensor = None

            model.state_dict()[key].gather(dst=0, out=gathered_tensor)

            if ctx.rank == 0:
                torch.testing.assert_close(
                    none_throws(gathered_tensor).cpu(),
                    original_tensor,
                    rtol=1e-5,
                    atol=1e-6,
                )
        elif isinstance(model.state_dict()[key], torch.Tensor):
            torch.testing.assert_close(
                model.state_dict()[key].cpu(), original_tensor, rtol=1e-5, atol=1e-6
            )
        else:
            raise AssertionError(
                f"Model state dict contains unsupported type for key: {key}"
            )


def _select_sharding_type(sharding_type: str) -> ParameterShardingGenerator:
    if sharding_type == "table_wise":
        return table_wise(rank=0)
    elif sharding_type == "column_wise":
        return column_wise(ranks=[0, 1])
    elif sharding_type == "row_wise":
        return row_wise()
    elif sharding_type == "data_parallel":
        return data_parallel()
    else:
        raise AssertionError(f"Invalid sharding type specified: {sharding_type}")


@skip_if_asan_class
class ParameterInitializationTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.TABLE_WISE.value,
            ]
        )
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_initialize_parameters_ec(self, sharding_type: str) -> None:
        world_size = 2
        backend = "nccl"
        table_name = "free_parameters"

        # Set seed for deterministic tensor generation
        torch.manual_seed(42)

        # Initialize embedding table on non-meta device, in this case cuda:0
        embedding_tables = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name=table_name,
                    embedding_dim=64,
                    num_embeddings=10,
                    data_type=DataType.FP32,
                )
            ],
        )

        # Use a fixed tensor with explicit seeding for consistent testing
        fixed_tensor = torch.randn(10, 64, generator=torch.Generator().manual_seed(42))
        embedding_tables.load_state_dict(
            {f"embeddings.{table_name}.weight": fixed_tensor}
        )

        self._run_multi_process_test(
            callable=initialize_and_test_parameters,
            embedding_tables=embedding_tables,
            sharding_type=sharding_type,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingCollectionSharder())
            ],
            world_size=world_size,
            backend=backend,
            table_name=table_name,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.TABLE_WISE.value,
            ]
        )
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_initialize_parameters_ebc(self, sharding_type: str) -> None:
        world_size = 2
        backend = "nccl"
        table_name = "free_parameters"

        # Set seed for deterministic tensor generation
        torch.manual_seed(42)

        # Initialize embedding bag on non-meta device, in this case cuda:0
        embedding_tables = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name=table_name,
                    embedding_dim=64,
                    num_embeddings=10,
                    data_type=DataType.FP32,
                )
            ],
        )

        # Use a fixed tensor with explicit seeding for consistent testing
        fixed_tensor = torch.randn(10, 64, generator=torch.Generator().manual_seed(42))
        embedding_tables.load_state_dict(
            {f"embedding_bags.{table_name}.weight": fixed_tensor}
        )

        self._run_multi_process_test(
            callable=initialize_and_test_parameters,
            embedding_tables=embedding_tables,
            sharding_type=sharding_type,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
            world_size=world_size,
            backend=backend,
            table_name=table_name,
        )
