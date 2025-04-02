#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import copy

import random
import unittest

from typing import Any, Dict, List, Optional, Union

import hypothesis.strategies as st

import torch

from hypothesis import given, settings, Verbosity
from torch import nn

from torchrec import distributed as trec_dist, EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection

from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    get_module_to_default_sharders,
    table_wise,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict

from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import data_type_to_dtype, EmbeddingBagConfig

from torchrec.test_utils import skip_if_asan_class
from torchrec.types import DataType


# Utils:
def table_name(i: int) -> str:
    return "table_" + str(i)


def feature_name(i: int) -> str:
    return "feature_" + str(i)


def generate_input_by_world_size(
    world_size: int,
    num_tables: int,
    num_embeddings: int = 4,
    max_mul: int = 3,
) -> List[KeyedJaggedTensor]:
    # TODO merge with new ModelInput generator in TestUtils
    kjt_input_per_rank = []
    mul = random.randint(1, max_mul)
    total_size = num_tables * mul

    for _ in range(world_size):
        feature_names = [feature_name(i) for i in range(num_tables)]
        lengths = []
        values = []
        counting_l = 0
        for i in range(total_size):
            if i == total_size - 1:
                lengths.append(total_size - counting_l)
                break
            next_l = random.randint(0, total_size - counting_l)
            values.extend(
                [random.randint(0, num_embeddings - 1) for _ in range(next_l)]
            )
            lengths.append(next_l)
            counting_l += next_l

        # for length in lengths:

        kjt_input_per_rank.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=feature_names,
                values=torch.LongTensor(values),
                lengths=torch.LongTensor(lengths),
            )
        )

    return kjt_input_per_rank


def generate_embedding_bag_config(
    data_type: DataType,
    num_tables: int = 3,
    embedding_dim: int = 16,
    num_embeddings: int = 4,
) -> List[EmbeddingBagConfig]:
    embedding_bag_config = []
    for i in range(num_tables):
        embedding_bag_config.append(
            EmbeddingBagConfig(
                name=table_name(i),
                feature_names=[feature_name(i)],
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                data_type=data_type,
            ),
        )
    return embedding_bag_config


def create_test_initial_state_dict(
    sharded_module_type: nn.Module,
    num_tables: int,
    data_type: DataType,
    embedding_dim: int = 16,
    num_embeddings: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Helpful for debugging:

    initial_state_dict = {
        "embedding_bags.table_0.weight": torch.tensor(
            [
                [1] * 16,
                [2] * 16,
                [3] * 16,
                [4] * 16,
            ],
        ),
        "embedding_bags.table_1.weight": torch.tensor(
            [
                [101] * 16,
                [102] * 16,
                [103] * 16,
                [104] * 16,
            ],
            dtype=data_type_to_dtype(data_type),
        ),
        ...
    }
    """

    initial_state_dict = {}
    for i in range(num_tables):
        # pyre-ignore
        extended_name = sharded_module_type.extend_shard_name(table_name(i))
        initial_state_dict[extended_name] = torch.tensor(
            [[j + (i * 100)] * embedding_dim for j in range(num_embeddings)],
            dtype=data_type_to_dtype(data_type),
        )

    return initial_state_dict


def are_modules_identical(
    module1: Union[EmbeddingBagCollection, ShardedEmbeddingBagCollection],
    module2: Union[EmbeddingBagCollection, ShardedEmbeddingBagCollection],
) -> None:
    # Check if both modules have the same type
    assert type(module1) is type(module2)

    # Check if both modules have the same parameters
    params1 = list(module1.named_parameters())
    params2 = list(module2.named_parameters())

    assert len(params1) == len(params2)

    for param1, param2 in zip(params1, params2):
        # Check parameter names
        assert param1[0] == param2[0]
        # Check parameter values
        assert torch.allclose(param1[1], param2[1])

    # Check if both modules have the same buffers
    buffers1 = list(module1.named_buffers())
    buffers2 = list(module2.named_buffers())

    assert len(buffers1) == len(buffers2)

    for buffer1, buffer2 in zip(buffers1, buffers2):
        assert buffer1[0] == buffer2[0]  # Check buffer names
        assert torch.allclose(buffer1[1], buffer2[1])  # Check buffer values


def output_sharding_plan_delta(
    old_plan: EmbeddingModuleShardingPlan, new_plan: EmbeddingModuleShardingPlan
) -> EmbeddingModuleShardingPlan:
    assert len(old_plan) == len(new_plan)
    return_plan = copy.deepcopy(new_plan)
    for shard_name, old_param in old_plan.items():
        if shard_name not in return_plan:
            raise ValueError(f"Shard {shard_name} not found in new plan")
        new_param = return_plan[shard_name]
        old_ranks = old_param.ranks
        new_ranks = new_param.ranks
        if old_ranks == new_ranks:
            del return_plan[shard_name]

    return return_plan


def _test_ebc_resharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    module_sharding_plan: EmbeddingModuleShardingPlan,
    new_module_sharding_plan: EmbeddingModuleShardingPlan,
    local_size: Optional[int] = None,
) -> None:
    """
    Distributed call to test resharding for ebc by creating 2 models with identical config and
    states:
        m1 sharded with new_module_sharding_plan
        m2 sharded with module_sharding_plan, then resharded with new_module_sharding_plan

    Expects m1 and resharded m2 to be the same, and predictions outputted from the same KJT
    inputs to be the same.

    TODO: modify to include other modules once dynamic sharding is built out.
    """
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]

        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }
        m1 = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        m2 = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        # Load initial State - making sure models are identical
        m1.load_state_dict(initial_state_dict)
        copy_state_dict(
            loc=m1.state_dict(),
            glob=copy.deepcopy(initial_state_dict),
        )

        m2.load_state_dict(initial_state_dict)
        copy_state_dict(
            loc=m2.state_dict(),
            glob=copy.deepcopy(initial_state_dict),
        )

        sharder = get_module_to_default_sharders()[type(m1)]

        # pyre-ignore
        env = ShardingEnv.from_process_group(ctx.pg)

        sharded_m1 = sharder.shard(
            module=m1,
            params=new_module_sharding_plan,
            env=env,
            device=ctx.device,
        )

        sharded_m2 = sharder.shard(
            module=m1,
            params=module_sharding_plan,
            env=env,
            device=ctx.device,
        )

        new_module_sharding_plan_delta = output_sharding_plan_delta(
            module_sharding_plan, new_module_sharding_plan
        )

        # pyre-ignore
        resharded_m2 = sharder.reshard(
            sharded_module=sharded_m2,
            changed_shard_to_params=new_module_sharding_plan_delta,
            env=env,
            device=ctx.device,
        )

        are_modules_identical(sharded_m1, resharded_m2)

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        # For current test model and inputs, the prediction should be the exact same
        rtol = 0
        atol = 0

        for _ in range(world_size):
            # sharded model
            # each rank gets a subbatch
            sharded_m1_pred_kt_no_dict = sharded_m1(kjt_input_per_rank[ctx.rank])
            resharded_m2_pred_kt_no_dict = resharded_m2(kjt_input_per_rank[ctx.rank])

            sharded_m1_pred_kt = sharded_m1_pred_kt_no_dict.to_dict()
            resharded_m2_pred_kt = resharded_m2_pred_kt_no_dict.to_dict()
            sharded_m1_pred = torch.stack(
                [sharded_m1_pred_kt[feature] for feature in feature_keys]
            )

            resharded_m2_pred = torch.stack(
                [resharded_m2_pred_kt[feature] for feature in feature_keys]
            )
            # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
            # in normal author modelling code this won't be an issue because each rank would individually create
            # their model. output from sharded_pred is correctly on the correct device.

            # Compare predictions of sharded vs unsharded models.
            torch.testing.assert_close(
                sharded_m1_pred.cpu(), resharded_m2_pred.cpu(), rtol=rtol, atol=atol
            )

            sharded_m1_pred.sum().backward()
            resharded_m2_pred.sum().backward()


@skip_if_asan_class
class MultiRankDynamicShardingTest(MultiProcessTestBase):
    def _run_ebc_resharding_test(
        self,
        per_param_sharding: Dict[str, ParameterSharding],
        new_per_param_sharding: Dict[str, ParameterSharding],
        num_tables: int,
        world_size: int,
        data_type: DataType,
        embedding_dim: int = 16,
        num_embeddings: int = 4,
    ) -> None:
        embedding_bag_config = generate_embedding_bag_config(
            data_type, num_tables, embedding_dim, num_embeddings
        )

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            # pyre-ignore
            per_param_sharding=per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        new_module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            # pyre-ignore
            per_param_sharding=new_per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Row-wise not supported on gloo
        if (
            not torch.cuda.is_available()
            and new_module_sharding_plan["table_0"].sharding_type
            == ShardingType.ROW_WISE.value
        ):
            return

        kjt_input_per_rank = generate_input_by_world_size(
            world_size, num_tables, num_embeddings
        )

        # initial_state_dict filled with deterministic dummy values
        initial_state_dict = create_test_initial_state_dict(
            ShardedEmbeddingBagCollection,  # pyre-ignore
            num_tables,
            data_type,
            embedding_dim,
            num_embeddings,
        )

        self._run_multi_process_test(
            callable=_test_ebc_resharding,
            world_size=world_size,
            tables=embedding_bag_config,
            initial_state_dict=initial_state_dict,
            kjt_input_per_rank=kjt_input_per_rank,
            backend="nccl" if torch.cuda.is_available() else "gloo",
            module_sharding_plan=module_sharding_plan,
            new_module_sharding_plan=new_module_sharding_plan,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(  # pyre-ignore
        num_tables=st.sampled_from([2, 3, 4]),
        data_type=st.sampled_from([DataType.FP32, DataType.FP16]),
        world_size=st.sampled_from([2, 4]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_dynamic_sharding_ebc_tw(
        self,
        num_tables: int,
        data_type: DataType,
        world_size: int,
    ) -> None:
        # Tests EBC dynamic sharding implementation for TW

        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
        old_ranks = [random.randint(0, world_size - 1) for _ in range(num_tables)]
        new_ranks = [random.randint(0, world_size - 1) for _ in range(num_tables)]

        if new_ranks == old_ranks:
            return
        per_param_sharding = {}
        new_per_param_sharding = {}

        # Construct parameter shardings
        for i in range(num_tables):
            per_param_sharding[table_name(i)] = table_wise(rank=old_ranks[i])
            new_per_param_sharding[table_name(i)] = table_wise(rank=new_ranks[i])

        self._run_ebc_resharding_test(
            per_param_sharding,
            new_per_param_sharding,
            num_tables,
            world_size,
            data_type,
        )
