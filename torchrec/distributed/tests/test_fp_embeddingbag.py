#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from operator import xor
from typing import List, Optional

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st, Verbosity
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import distributed as trec_dist
from torchrec.distributed.fp_embeddingbag import (
    FeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.shard import _shard_modules

from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    data_parallel,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.tests.test_fp_embeddingbag_utils import (
    create_module_and_freeze,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def _test_sharding(  # noqa C901
    tables: List[EmbeddingBagConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    set_gradient_division: bool,
    local_size: Optional[int] = None,
    use_dmp: bool = False,
    use_fp_collection: bool = False,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        trec_dist.comm_ops.set_gradient_division(set_gradient_division)

        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]

        sparse_arch = create_module_and_freeze(
            tables, use_fp_collection=use_fp_collection, device=ctx.device
        )

        apply_optimizer_in_backward(
            torch.optim.SGD,
            sparse_arch._fp_ebc._embedding_bag_collection.embedding_bags.parameters(),
            {"lr": 1.0},
        )

        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._fp_ebc,
            per_param_sharding={
                "table_0": column_wise(ranks=[0, 1]),
                "table_1": table_wise(rank=0),
                "table_2": data_parallel(),
                "table_3": column_wise(ranks=[0, 0, 1]),
            },
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        if use_dmp:
            sharded_sparse_arch = DistributedModelParallel(
                module=copy.deepcopy(sparse_arch),
                plan=ShardingPlan({"_fp_ebc": module_sharding_plan}),
                env=ShardingEnv.from_process_group(ctx.pg),
                sharders=[sharder],
                device=ctx.device,
            )
        else:
            sharded_sparse_arch = _shard_modules(
                module=copy.deepcopy(sparse_arch),
                plan=ShardingPlan({"._fp_ebc": module_sharding_plan}),
                env=ShardingEnv.from_process_group(ctx.pg),
                sharders=[sharder],
                device=ctx.device,
            )
            from torch.distributed._composable.replicate import replicate

            replicate(
                sharded_sparse_arch,
                # pyre-ignore
                ignored_modules=[sharded_sparse_arch._fp_ebc._embedding_bag_collection],
                process_group=ctx.pg,
                gradient_as_bucket_view=True,
                device_ids=None if ctx.device.type == "cpu" else [ctx.device],
                broadcast_buffers=False,
            )

        copy_state_dict(
            sharded_sparse_arch.state_dict(),
            copy.deepcopy(sparse_arch.state_dict()),
        )

        unsharded_model_preds = []
        for unsharded_rank in range(ctx.world_size):
            # simulate the unsharded model run on the entire batch
            unsharded_model_preds.append(
                sparse_arch(kjt_input_per_rank[unsharded_rank])[0]
            )

        unsharded_model_pred_this_rank = unsharded_model_preds[ctx.rank]

        # sharded model
        # each rank gets a subbatch
        sharded_model_pred = sharded_sparse_arch(kjt_input_per_rank[ctx.rank])[0]

        torch.testing.assert_close(
            sharded_model_pred.cpu(), unsharded_model_pred_this_rank.cpu()
        )

        torch.stack(unsharded_model_preds).mean().backward()
        sharded_model_pred.mean().backward()

        unsharded_named_parameters = dict(sparse_arch.named_parameters())
        sharded_named_parameters = dict(sharded_sparse_arch.named_parameters())

        for (fqn, param) in unsharded_named_parameters.items():
            if "_feature_processors" not in fqn:
                continue

            replicated_param = sharded_named_parameters[fqn]

            torch.testing.assert_close(
                # pyre-ignore
                param.grad.cpu(),
                replicated_param.grad.cpu(),
                msg=f"Did not match for {fqn} {param.grad=} {replicated_param.grad=}",
            )

        assert (
            sparse_arch.state_dict().keys() == sharded_sparse_arch.state_dict().keys()
        ), "State dict keys are not the same"


@skip_if_asan_class
class ShardedEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    # pyre-ignore
    @given(
        set_gradient_division=st.booleans(),
        use_dmp=st.booleans(),
        use_fp_collection=st.booleans(),
    )
    def test_sharding_ebc(
        self, set_gradient_division: bool, use_dmp: bool, use_fp_collection: bool
    ) -> None:

        import hypothesis

        # don't need to test entire matrix
        hypothesis.assume(not (set_gradient_division and use_dmp))
        hypothesis.assume(not xor(use_dmp, use_fp_collection))

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=3 * 16,
                num_embeddings=16,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingBagConfig(
                name="table_2",
                feature_names=["feature_2"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingBagConfig(
                name="table_3",
                feature_names=["feature_3"],
                embedding_dim=3 * 16,
                num_embeddings=16,
            ),
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]
        # "feature_2"   [3, 1]       [4,1]        [5]
        # "feature_3"   [1]       [6,1,8]        [0,3,3]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]       [0,1,2,3]
        # "feature_1"   [2, 3]       None        [2]
        # "feature_2"   [2, 7]       [1,8,2]        [8,1]
        # "feature_3"   [9]       [8]        [7]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2", "feature_3"],
                values=torch.LongTensor(
                    [0, 1, 2, 0, 1, 2, 3, 1, 4, 1, 5, 1, 6, 1, 8, 0, 3, 3]
                ),
                lengths=torch.LongTensor(
                    [
                        2,
                        0,
                        1,
                        2,
                        0,
                        1,
                        2,
                        2,
                        1,
                        1,
                        3,
                        3,
                    ]
                ),
                weights=torch.FloatTensor(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2", "feature_3"],
                values=torch.LongTensor(
                    [3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2, 2, 7, 1, 8, 2, 8, 1, 9, 8, 7]
                ),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1, 2, 3, 2, 1, 1, 1]),
                weights=torch.FloatTensor(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ),
            ),
        ]

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=FeatureProcessedEmbeddingBagCollectionSharder(),
            backend="nccl"
            if (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
            else "gloo",
            set_gradient_division=set_gradient_division,
            use_dmp=use_dmp,
            use_fp_collection=use_fp_collection,
        )
