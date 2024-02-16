#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest
from typing import cast, Dict, List, Optional, OrderedDict, Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torch import distributed as dist, nn
from torchrec import distributed as trec_dist
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.test_utils import get_free_port


class SequenceModelParallelStateDictTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())

        self.backend = "nccl"
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        dist.init_process_group(backend=self.backend)

        num_features = 4
        self.num_float_features = 16
        self.batch_size = 3
        shared_features = 2

        initial_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        shared_features_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i + num_features),
                feature_names=["feature_" + str(i)],
            )
            for i in range(shared_features)
        ]

        self.tables = initial_tables + shared_features_tables
        self.shared_features = [f"feature_{i}" for i in range(shared_features)]

        self.embedding_groups = {
            "group_0": [
                f"{feature}@{table.name}"
                if feature in self.shared_features
                else feature
                for table in self.tables
                for feature in table.feature_names
            ]
        }

    def tearDown(self) -> None:
        dist.destroy_process_group()

    def _generate_dmps_and_batch(
        self,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        constraints: Optional[Dict[str, trec_dist.planner.ParameterConstraints]] = None,
    ) -> Tuple[List[DistributedModelParallel], ModelInput]:
        """
        Generate two DMPs based on Sequence Sparse NN and one batch of data.
        """
        if constraints is None:
            constraints = {}
        if sharders is None:
            sharders = get_default_sharders()

        _, local_batch = ModelInput.generate(
            batch_size=self.batch_size,
            world_size=1,
            tables=self.tables,
            num_float_features=self.num_float_features,
            weighted_tables=[],
        )
        batch = local_batch[0].to(self.device)

        dmps = []
        pg = dist.GroupMember.WORLD
        assert pg is not None, "Process group is not initialized"
        env = ShardingEnv.from_process_group(pg)

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=trec_dist.comm.get_local_size(env.world_size),
                world_size=env.world_size,
                compute_device=self.device.type,
            ),
            constraints=constraints,
        )

        for _ in range(2):
            # Create two TestSparseNN modules, wrap both in DMP
            m = TestSequenceSparseNN(
                tables=self.tables,
                num_float_features=self.num_float_features,
                embedding_groups=self.embedding_groups,
                dense_device=self.device,
                sparse_device=torch.device("meta"),
            )
            if pg is not None:
                plan = planner.collective_plan(m, sharders, pg)
            else:
                plan = planner.plan(m, sharders)

            dmp = DistributedModelParallel(
                module=m,
                init_data_parallel=False,
                device=self.device,
                sharders=sharders,
                plan=plan,
            )

            with torch.no_grad():
                dmp(batch)
                dmp.init_data_parallel()
            dmps.append(dmp)
        return (dmps, batch)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ]
        ),
        is_training=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_load_state_dict(
        self,
        sharding_type: str,
        kernel_type: str,
        is_training: bool,
    ) -> None:
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                ),
            ),
        ]
        models, batch = self._generate_dmps_and_batch(sharders)
        m1, m2 = models

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # validate the models are equivalent
        if is_training:
            for _ in range(2):
                loss1, pred1 = m1(batch)
                loss2, pred2 = m2(batch)
                loss1.backward()
                loss2.backward()
                self.assertTrue(torch.equal(loss1, loss2))
                self.assertTrue(torch.equal(pred1, pred2))
        else:
            with torch.no_grad():
                loss1, pred1 = m1(batch)
                loss2, pred2 = m2(batch)
                self.assertTrue(torch.equal(loss1, loss2))
                self.assertTrue(torch.equal(pred1, pred2))

        sd1 = m1.state_dict()
        for key, value in m2.state_dict().items():
            v2 = sd1[key]
            if isinstance(value, ShardedTensor):
                assert len(value.local_shards()) == 1
                dst = value.local_shards()[0].tensor
            else:
                dst = value
            if isinstance(v2, ShardedTensor):
                assert len(v2.local_shards()) == 1
                src = v2.local_shards()[0].tensor
            else:
                src = v2
            self.assertTrue(torch.equal(src, dst))
