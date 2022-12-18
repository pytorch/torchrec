#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import os
import tempfile
import unittest
import uuid

import torch
from torch import distributed as dist, nn
from torch.distributed._composable import fully_shard
from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec.distributed.shard import (
    shard as trec_shard,
    shard_modules as trec_shard_modules,
)
from torchrec.distributed.sharding_plan import (
    apply_to_all,
    construct_module_sharding_plan,
    row_wise,
)
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.types import ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import skip_if_asan


class FSDPTest(unittest.TestCase):
    @classmethod
    def _run(cls, path: str) -> None:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            device: torch.device = torch.device(f"cuda:{rank}")
            backend = "nccl"
            torch.cuda.set_device(device)
        else:
            device: torch.device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        num_float_features = 32

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(2)
        ]
        m = TestSparseNN(
            tables=tables,
            num_float_features=num_float_features,
            weighted_tables=weighted_tables,
            dense_device=device,
        )
        plan = ShardingPlan(
            plan={
                "sparse.ebc": construct_module_sharding_plan(
                    m.sparse.ebc,
                    apply_to_all(m.sparse.ebc, row_wise()),
                ),
                "sparse.weighted_ebc": construct_module_sharding_plan(
                    m.sparse.weighted_ebc,
                    apply_to_all(m.sparse.weighted_ebc, row_wise()),
                ),
            }
        )
        trec_shard_modules(
            module=m,
            device=device,
            plan=plan,
        )
        sharded_m = FullyShardedDataParallel(
            module=m,
            device_id=rank,
            ignored_modules=[m.sparse],
        )

        ######## run one iteration ########
        _, local_batch = ModelInput.generate(
            batch_size=8,
            world_size=world_size,
            num_float_features=num_float_features,
            tables=tables,
            weighted_tables=weighted_tables,
        )
        batch = local_batch[0].to(device)
        sharded_m(batch)[1].sum().backward()

        writer = FileSystemWriter(path=path)
        reader = FileSystemReader(path=path)
        with FullyShardedDataParallel.state_dict_type(
            sharded_m, StateDictType.SHARDED_STATE_DICT
        ):
            state_dict = sharded_m.state_dict()

        save_state_dict(state_dict, writer)

        p_sum = torch.zeros(1, device=device)
        for p in sharded_m.parameters():
            with torch.no_grad():
                if isinstance(p, ShardedTensor):
                    if not p.local_shards():
                        continue
                    p = p.local_tensor()
                p_sum += p.sum()
                p.zero_()
                assert p.sum() == 0

        with FullyShardedDataParallel.state_dict_type(
            sharded_m, StateDictType.SHARDED_STATE_DICT
        ):
            state_dict = sharded_m.state_dict()
            load_state_dict(state_dict, reader)
            sharded_m.load_state_dict(state_dict)

        p_sum_loaded = torch.zeros(1, device=device)
        for p in sharded_m.parameters():
            with torch.no_grad():
                if isinstance(p, ShardedTensor):
                    if not p.local_shards():
                        continue
                    p = p.local_tensor()
                p_sum_loaded += p.sum()
        assert p_sum.allclose(p_sum_loaded)

    @skip_if_asan
    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as path:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )
            elastic_launch(config=lc, entrypoint=self._run)(path)


class FSDPTestComposable(unittest.TestCase):
    @classmethod
    def _run(cls) -> None:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device: torch.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
        num_float_features = 32

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(2)
        ]
        m = TestSparseNN(
            tables=tables,
            num_float_features=num_float_features,
            weighted_tables=weighted_tables,
            dense_device=device,
        )
        m.sparse.ebc = trec_shard(
            module=m.sparse.ebc,
            device=device,
            plan=construct_module_sharding_plan(
                m.sparse.ebc,
                apply_to_all(m.sparse.ebc, row_wise()),
            ),
        )
        m.sparse.weighted_ebc = trec_shard(
            module=m.sparse.weighted_ebc,
            device=device,
            plan=construct_module_sharding_plan(
                m.sparse.weighted_ebc,
                apply_to_all(m.sparse.weighted_ebc, row_wise()),
            ),
        )
        m.dense = fully_shard(
            m.dense,
            device_id=device.index,
            policy=ModuleWrapPolicy({nn.Linear}),
        )
        m.over = fully_shard(
            m.over,
            device_id=device.index,
            policy=ModuleWrapPolicy({nn.Linear}),
        )

        ######## run one iteration ########
        _, local_batch = ModelInput.generate(
            batch_size=8,
            world_size=world_size,
            num_float_features=num_float_features,
            tables=tables,
            weighted_tables=weighted_tables,
        )
        batch = local_batch[0].to(device)
        m(batch)[1].sum().backward()
        # TODO add checkpointing test once fully_shard supports
        # TODO add apply_optimizer_in_backward() API and optimizer state checkpoint

    @skip_if_asan
    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_composable_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )
            elastic_launch(config=lc, entrypoint=self._run)()
