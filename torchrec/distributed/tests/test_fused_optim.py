#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import List, Optional, Dict, cast, Union

import hypothesis.strategies as st
import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import Verbosity, given, settings
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
)
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.test_model import (
    TestSparseNN,
    TestSparseNNBase,
    TestEBCSharder,
    TestEBSharder,
)
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelTestBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingType,
    ShardingEnv,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import (
    skip_if_asan_class,
    init_distributed_single_host,
    seed_and_log,
)


def create_test_sharder(
    sharding_type: str, kernel_type: str, optim: EmbOptimType
) -> Union[TestEBSharder, TestEBCSharder]:
    fused_params = {}
    fused_params["optimizer"] = optim
    if optim == EmbOptimType.EXACT_SGD:
        fused_params["learning_rate"] = 0.1
    else:
        fused_params["learning_rate"] = 0.01
    return TestEBCSharder(sharding_type, kernel_type, fused_params)


@skip_if_asan_class
class ModelParallelTest(ModelParallelTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
        optim_type=st.sampled_from(
            [
                EmbOptimType.EXACT_SGD,
                EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_nccl_rw(
        self,
        sharding_type: str,
        kernel_type: str,
        optim_type: EmbOptimType,
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharding_type, kernel_type, optim_type),
            ],
            backend="nccl",
            optim=optim_type,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
        optim_type=st.sampled_from(
            [
                EmbOptimType.EXACT_SGD,
                EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_nccl_tw(
        self,
        sharding_type: str,
        kernel_type: str,
        optim_type: EmbOptimType,
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharding_type, kernel_type, optim_type),
            ],
            backend="nccl",
            optim=optim_type,
        )

    @seed_and_log
    def setUp(self) -> None:
        super().setUp()
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False

        num_features = 4
        num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 2) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 2) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]

        self.embedding_groups = {
            "group_0": ["feature_" + str(i) for i in range(num_features)]
        }

    def _test_sharding(
        self,
        sharders: List[ModuleSharder[nn.Module]],
        optim: EmbOptimType,
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None:
        self._run_multi_process_test(
            # pyre-ignore [6]
            callable=self._test_optim_single_rank,
            world_size=world_size,
            local_size=local_size,
            model_class=cast(TestSparseNNBase, TestSparseNN),
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            backend=backend,
            optim=optim,
            constraints=constraints,
        )

    @classmethod
    def _test_optim_single_rank(
        cls,
        rank: int,
        world_size: int,
        model_class: TestSparseNNBase,
        embedding_groups: Dict[str, List[str]],
        tables: List[EmbeddingTableConfig],
        sharders: List[ModuleSharder[nn.Module]],
        backend: str,
        optim: EmbOptimType,
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        local_size: Optional[int] = None,
    ) -> None:
        # Override local_size after pg construction because unit test device count
        # is larger than local_size setup. This can be problematic for twrw because
        # we have ShardedTensor placement check.
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        if backend == "nccl":
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        pg = init_distributed_single_host(
            rank=rank,
            world_size=world_size,
            backend=backend,
            local_size=local_size,
        )
        if rank == 0:
            global_pg = dist.new_group(ranks=[0], backend=backend)
            dist.new_group(ranks=[1], backend=backend)
        else:
            dist.new_group(ranks=[0], backend=backend)
            global_pg = dist.new_group(ranks=[1], backend=backend)

        # Generate model & inputs.
        (global_model, inputs) = cls._gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=world_size,
            num_float_features=16,
        )
        global_model = global_model.cuda(0)
        global_model = DistributedModelParallel(
            global_model,
            env=ShardingEnv.from_process_group(global_pg),
            sharders=sharders,
            device=torch.device("cuda:0"),
            init_data_parallel=False,
        )
        global_input = inputs[0][0].to(torch.device("cuda:0"))
        local_input = inputs[0][1][rank].to(device)

        # Run single step of unsharded model to populate optimizer states.
        global_opt = torch.optim.SGD(global_model.parameters(), lr=0.1)
        cls._gen_full_pred_after_one_step(global_model, global_opt, global_input)

        # Shard model.
        local_model = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            dense_device=device,
            sparse_device=torch.device("meta"),
            num_float_features=16,
        )
        local_model = DistributedModelParallel(
            local_model,
            env=ShardingEnv.from_process_group(pg),
            sharders=sharders,
            device=device,
        )
        local_opt = torch.optim.SGD(local_model.parameters(), lr=0.1)

        # Load model & optimizer states from the global model.
        cls._copy_state_dict(local_model.state_dict(), global_model.state_dict())
        # pyre-ignore [16]
        for param_name, local_state in local_model.fused_optimizer.state_dict()[
            "state"
        ].items():
            global_state = global_model.fused_optimizer.state_dict()["state"][
                param_name
            ]
            cls._copy_state_dict(local_state, global_state)

        # Run a single training step of the sharded model.
        local_pred = cls._gen_full_pred_after_one_step(
            local_model, local_opt, local_input
        )
        all_local_pred = []
        for _ in range(world_size):
            all_local_pred.append(torch.empty_like(local_pred))
        dist.all_gather(all_local_pred, local_pred, group=pg)

        # Run second training step of the unsharded model.
        global_opt = torch.optim.SGD(global_model.parameters(), lr=0.1)
        global_pred = cls._gen_full_pred_after_one_step(
            global_model, global_opt, global_input
        )

        # Compare predictions of sharded vs unsharded models.
        torch.testing.assert_allclose(
            global_pred.cpu(), torch.cat(all_local_pred).cpu()
        )
