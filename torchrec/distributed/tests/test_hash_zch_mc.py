#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#!/usr/bin/env python3

# pyre-strict

import copy
import multiprocessing
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pyre_extensions import none_throws
from torch import nn
from torchrec import (
    EmbeddingCollection,
    EmbeddingConfig,
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)
from torchrec.distributed import ModuleSharder, ShardingEnv
from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder

from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    EmbeddingCollectionSharder,
    ManagedCollisionEmbeddingCollectionSharder,
    row_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingPlan
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import ManagedCollisionCollection

BASE_LEAF_MODULES = [
    "IntNBitTableBatchedEmbeddingBagsCodegen",
    "HashZchManagedCollisionModule",
]


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        buckets: int,
        return_remapped: bool = False,
        input_hash_size: int = 4000,
        is_inference: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        mc_modules = {}
        mc_modules["table_0"] = HashZchManagedCollisionModule(
            is_inference=is_inference,
            zch_size=(tables[0].num_embeddings),
            input_hash_size=input_hash_size,
            device=device,
            total_num_buckets=buckets,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=["feature_0"],
                single_ttl=1,
            ),
        )

        mc_modules["table_1"] = HashZchManagedCollisionModule(
            is_inference=is_inference,
            zch_size=(tables[1].num_embeddings),
            device=device,
            input_hash_size=input_hash_size,
            total_num_buckets=buckets,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=["feature_1"],
                single_ttl=1,
            ),
        )

        self._mc_ec: ManagedCollisionEmbeddingCollection = (
            ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(
                    tables=tables,
                    device=device,
                ),
                ManagedCollisionCollection(
                    managed_collision_modules=mc_modules,
                    embedding_configs=tables,
                ),
                return_remapped_features=self._return_remapped,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:
        return self._mc_ec(kjt)


class TestHashZchMcEmbedding(MultiProcessTestBase):
    # pyre-ignore
    @unittest.skipIf(torch.cuda.device_count() <= 1, "Not enough GPUs, skipping")
    def test_hash_zch_mc_ec(self) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        train_input_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    list(range(1000, 1025)),
                ),
                lengths=torch.LongTensor([1] * 8 + [2] * 8),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    list(range(25000, 25025)),
                ),
                lengths=torch.LongTensor([1] * 8 + [2] * 8),
                weights=None,
            ),
        ]
        train_state_dict = multiprocessing.Manager().dict()

        # Train Model with ZCH on GPU
        self._run_multi_process_test(
            callable=_train_model,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            num_buckets=2,
            kjt_input_per_rank=train_input_per_rank,
            sharder=ManagedCollisionEmbeddingCollectionSharder(
                EmbeddingCollectionSharder(),
                ManagedCollisionCollectionSharder(),
            ),
            return_dict=train_state_dict,
            backend="nccl",
        )


def _train_model(
    tables: List[EmbeddingConfig],
    num_buckets: int,
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    return_dict: Dict[str, Any],
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input = kjt_input_per_rank[rank].to(ctx.device)

        train_model = SparseArch(
            tables=tables,
            device=torch.device("cuda"),
            input_hash_size=0,
            return_remapped=True,
            buckets=num_buckets,
        )
        train_sharding_plan = construct_module_sharding_plan(
            train_model._mc_ec,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda",
            sharder=sharder,
        )
        print(f"train_sharding_plan: {train_sharding_plan}")
        sharded_train_model = _shard_modules(
            module=copy.deepcopy(train_model),
            plan=ShardingPlan({"_mc_ec": train_sharding_plan}),
            env=ShardingEnv.from_process_group(none_throws(ctx.pg)),
            sharders=[sharder],
            device=ctx.device,
        )
        # train
        sharded_train_model(kjt_input.to(ctx.device))

        for (
            key,
            value,
        ) in (
            # pyre-ignore
            sharded_train_model._mc_ec._managed_collision_collection._managed_collision_modules.state_dict().items()
        ):
            return_dict[f"mc_{key}_{rank}"] = value.cpu()
        for (
            key,
            value,
            # pyre-ignore
        ) in sharded_train_model._mc_ec._embedding_collection.state_dict().items():
            tensors = []
            for i in range(len(value.local_shards())):
                tensors.append(value.local_shards()[i].tensor.cpu())
            return_dict[f"ec_{key}_{rank}"] = torch.cat(tensors, dim=0)
