#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import Any, Dict, List, Optional, OrderedDict, Type

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings, Verbosity
from torchrec import distributed as trec_dist
from torchrec.distributed.modular_fused_embeddingbag import (
    FusedEmbeddingBagCollectionSharder,
    ShardedFusedEmbeddingBagCollection,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)

from torchrec.distributed.shard_embedding_modules import shard_embedding_modules

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    ModuleSharder,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_model_characteristics, skip_if_asan_class


def test_sharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    optimizer_type: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    is_data_parallel: bool = False,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:

        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        model = FusedEmbeddingBagCollection(
            tables=tables,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=ctx.device,
        )
        plan: ShardingPlan = planner.collective_plan(model, [sharder], ctx.pg)
        sharded_model, sharded_parameter_names = shard_embedding_modules(
            module=model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan,
            sharders=[sharder],
            device=ctx.device,
        )

        unsharded_model = FusedEmbeddingBagCollection(
            tables=tables,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=ctx.device,
        )

        assert isinstance(sharded_model, ShardedFusedEmbeddingBagCollection)

        unsharded_model.load_state_dict(copy.deepcopy(initial_state_dict))
        copy_state_dict(sharded_model.state_dict(), copy.deepcopy(initial_state_dict))

        unsharded_model_optimizer = unsharded_model.fused_optimizer()
        sharded_model_optimizer = sharded_model.fused_optimizer()

        for it in range(5):
            unsharded_model_optimizer.zero_grad()

            unsharded_model_pred_jt = []
            for rank in range(ctx.world_size):
                # simulate the unsharded model run on the entire batch
                unsharded_model_pred_jt.append(
                    unsharded_model(kjt_input_per_rank[rank])
                )

            # sharded model
            # each rank gets a subbatch
            sharded_model_optimizer.zero_grad()
            sharded_model_pred_kt = sharded_model(kjt_input_per_rank[ctx.rank])

            if not is_data_parallel or it < 1:
                # I think there's a numerical bug with data_parallel._register_fused_optim
                torch.testing.assert_allclose(
                    unsharded_model_pred_jt[ctx.rank].values().detach().clone().cpu(),
                    sharded_model_pred_kt.values().detach().clone().cpu(),
                )

            for jt in unsharded_model_pred_jt:
                jt.values().sum().backward()
            sharded_model_pred_kt.values().sum().backward()
            for param in sharded_model.parameters():
                # TODO
                # For model paralllel: this is correct for the wrong reasons,
                # ShardedTensor.grad is always None, which is what we expect in the fused case.
                # In the data_parallel case this is incorrect, as it needs to be None
                assert param.grad is None

            unsharded_model_optimizer.step()
            sharded_model_optimizer.step()

        # check nn.Module APIs look the same
        model_characteristics = {}
        model_characteristics["unsharded_model"] = get_model_characteristics(
            unsharded_model
        )
        model_characteristics["sharded_model"] = get_model_characteristics(
            sharded_model
        )

        assert (
            model_characteristics["unsharded_model"]["named_buffers"].keys()
            == model_characteristics["sharded_model"]["named_buffers"].keys()
        )

        assert (
            model_characteristics["unsharded_model"]["named_parameters"].keys()
            == model_characteristics["sharded_model"]["named_parameters"].keys()
        )

        assert (
            model_characteristics["unsharded_model"]["state_dict"].keys()
            == model_characteristics["sharded_model"]["state_dict"].keys()
        )

        for fqn in unsharded_model.state_dict():
            unsharded_state = unsharded_model.state_dict()[fqn]
            sharded_state = sharded_model.state_dict()[fqn]

            if is_data_parallel:
                continue
            else:
                out = (
                    torch.zeros(size=unsharded_state.shape, device=ctx.device)
                    if ctx.rank == 0
                    else None
                )
                sharded_state.gather(out=out)
                if ctx.rank == 0:
                    torch.testing.assert_allclose(
                        unsharded_state,
                        out,
                    )

        # ShardedTensor needs to support copy:
        # While copying the parameter named "embedding_bags.table_0.weight", whose dimensions in the model are torch.Size([4, 4])
        # and whose dimensions in the checkpoint are torch.Size([4, 4]), an exception occurred :
        # ("torch function 'copy_', with args: (ShardedTensor(ShardedTensorMetadata(shards_metadat ... None not supported for ShardedTensor!",).
        # sharded_model.load_state_dict(sharded_model.state_dict())
        sharded_model_optimizer.load_state_dict(sharded_model_optimizer.state_dict())


class TestFusedEmbeddingBagCollectionSharder(FusedEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


@skip_if_asan_class
class ShardedEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.DATA_PARALLEL.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding_fused_ebc(
        self,
        sharding_type: str,
    ) -> None:

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0", "feature_1"],
                embedding_dim=4,
                num_embeddings=4,
            )
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]        [0, 1,2,3]
        # "feature_1"   [2,3]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([0, 1, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1]),
            ),
        ]

        self._run_multi_process_test(
            callable=test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            # pyre-ignore
            initial_state_dict=OrderedDict(
                [
                    (
                        "embedding_bags.table_0.weight",
                        torch.Tensor(
                            [
                                [1, 1, 1, 1],
                                [2, 2, 2, 2],
                                [4, 4, 4, 4],
                                [8, 8, 8, 8],
                            ]
                        ),
                    )
                ]
            ),
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01},
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=TestFusedEmbeddingBagCollectionSharder(sharding_type=sharding_type),
            backend="nccl",
            is_data_parallel=(sharding_type == ShardingType.DATA_PARALLEL.value),
        )
