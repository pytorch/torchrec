#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import cast, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.mc_embeddingbag import (
    ManagedCollisionEmbeddingBagCollectionSharder,
    ShardedManagedCollisionEmbeddingBagCollection,
)
from torchrec.distributed.mc_module import ShardedManagedCollisionCollection
from torchrec.distributed.shard import _shard_modules

from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.managed_collision_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection

from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        device: torch.device,
        return_remapped: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        self._mc_ebc: ManagedCollisionEmbeddingBagCollection = (
            ManagedCollisionEmbeddingBagCollection(
                EmbeddingBagCollection(
                    tables=tables,
                    device=device,
                ),
                cast(
                    Dict[str, ManagedCollisionModule],
                    {
                        "table_0": MCHManagedCollisionModule(
                            max_output_id=16,
                            device=device,
                            is_train=True,
                            max_history_size=8,
                            zch_size=8,
                            hash_func=lambda input_ids, hash_size: torch.remainder(
                                input_ids, hash_size
                            ),
                            eviction_policy=DistanceLFU_EvictionPolicy(),
                        ),
                        "table_1": MCHManagedCollisionModule(
                            max_output_id=32,
                            device=device,
                            is_train=True,
                            max_history_size=16,
                            zch_size=16,
                            hash_func=lambda input_ids, hash_size: torch.remainder(
                                input_ids, hash_size
                            ),
                            eviction_policy=DistanceLFU_EvictionPolicy(),
                        ),
                    },
                ),
                return_remapped_features=self._return_remapped,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, JaggedTensor]]]:
        if self._return_remapped:
            ebc_out, remapped_ids_out = self._mc_ebc(kjt)
        else:
            ebc_out = self._mc_ebc(kjt)
            remapped_ids_out = None
        pred = torch.cat(
            [ebc_out[key] for key in ["feature_0", "feature_1"]],
            dim=1,
        )
        loss = pred.mean()
        return loss, pred, remapped_ids_out


def _test_sharding(  # noqa C901
    tables: List[EmbeddingBagConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]

        return_remapped: bool = True
        sparse_arch = SparseArch(
            tables, torch.device("meta"), return_remapped=return_remapped
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ebc,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ebc": module_sharding_plan}),
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ebc, ShardedManagedCollisionEmbeddingBagCollection
        )
        assert isinstance(
            sharded_sparse_arch._mc_ebc._managed_collision_collection,
            ShardedManagedCollisionCollection,
        )

        test_state_dict = sharded_sparse_arch.state_dict()
        sharded_sparse_arch.load_state_dict(test_state_dict)

        # sharded model
        # each rank gets a subbatch
        sharded_model_output = sharded_sparse_arch(kjt_input_per_rank[ctx.rank])
        sharded_model_pred = sharded_model_output[0]
        remapped_ids_out = sharded_model_output[2]
        if return_remapped:
            assert remapped_ids_out
        sharded_model_pred.mean().backward()

        if ctx.rank == 0:
            param = (
                sharded_sparse_arch._mc_ebc._embedding_bag_collection.embedding_bags[
                    "table_0"
                ].weight
            )
            param.fill_(100.0)
            sharded_sparse_arch._mc_ebc._evict({"table_0": torch.tensor([0, 2, 4, 6])})
            # these indices will retain their original value of 100.0
            torch.testing.assert_close(
                param[[1, 3, 5, 7]], 100.0 * torch.ones_like(param[[1, 3, 5, 7]])
            )
            # these indices will be reset to values that are in uniform(-1, 1)
            assert torch.all((param[[0, 2, 4, 6]] < 1.0)).item()


@skip_if_asan_class
class ShardedMCEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_sharding_mc_ebc(self) -> None:

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1
        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]       [0,1,2,3]
        # "feature_1"   [2, 3]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    [10000000, 10000001, 10000002, 1000000, 1000001, 20000002],
                ),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    [
                        200003,
                        2000002,
                        200001,
                        2000002,
                        2000000,
                        200001,
                        2000002,
                        2000003,
                        3141592,
                        65358979,
                        323846,
                    ],
                ),
                lengths=torch.LongTensor([2, 2, 4, 1, 1, 1]),
                weights=None,
            ),
        ]

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=ManagedCollisionEmbeddingBagCollectionSharder(),
            backend="nccl",
        )
