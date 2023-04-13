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
from typing import List, Tuple
from copy import deepcopy

import torch
from torch import distributed as dist
from torch.distributed._composable import replicate
from torch.distributed._shard.api import ShardedTensor
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec.distributed.shard import shard as trec_shard, shard_modules
from torchrec.distributed.sharding_plan import column_wise
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import skip_if_asan

SEED = 42


class DDPTest(unittest.TestCase):
    @classmethod
    def _get_module(
        cls, path: str
    ) -> Tuple[
        TestSparseNN,
        torch.device,
        int,
        int,
        List[EmbeddingBagConfig],
        List[EmbeddingBagConfig],
    ]:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        if torch.cuda.is_available():
            device: torch.device = torch.device(f"cuda:{rank}")
            backend = "nccl"
            torch.cuda.set_device(device)
        else:
            device: torch.device = torch.device("cpu")
            backend = "gloo"

        if not torch.distributed.is_initialized():
            dist.init_process_group(backend=backend)
        num_float_features = 32

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4 * world_size,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4 * world_size,
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
        return m, device, world_size, num_float_features, tables, weighted_tables

    @classmethod
    def _run_compose_test(cls, path: str) -> None:
        (
            m,
            device,
            world_size,
            num_float_features,
            tables,
            weighted_tables,
        ) = cls._get_module(path)
        shard_modules(m)
        replicate(m, ignored_modules=[m.sparse])
        ######## run one iteration ########
        _, local_batch = ModelInput.generate(
            batch_size=8,
            world_size=world_size,
            num_float_features=num_float_features,
            tables=tables,
            weighted_tables=weighted_tables,
        )
        batch = local_batch[0].to(device)
        batch_local = deepcopy(batch)
        m(batch)[1].sum().backward()

        # run local model
        (
            m_local,
            device,
            world_size,
            num_float_features,
            tables,
            weighted_tables,
        ) = cls._get_module(path)
        m_local.cuda()
        m_local(batch_local)[1].sum().backward()
        # Verify grads are equal
        for p, p_local in zip(m.parameters(), m_local.parameters()):
            assert p.sum() == p_local.sum()

    @classmethod
    def _run_checkpoint_test(cls, path: str) -> None:
        (
            m,
            device,
            world_size,
            num_float_features,
            tables,
            weighted_tables,
        ) = cls._get_module(path)
        m.sparse.ebc = trec_shard(
            module=m.sparse.ebc,
            device=device,
            plan=column_wise(ranks=list(range(world_size))),
        )
        m.sparse.weighted_ebc = trec_shard(
            module=m.sparse.weighted_ebc,
            device=device,
            plan=column_wise(ranks=list(range(world_size))),
        )
        m.over = replicate(m.over)
        m.dense = replicate(m.dense)

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

        state_dict = m.state_dict()
        writer = FileSystemWriter(path=path)
        reader = FileSystemReader(path=path)
        save_state_dict(state_dict, writer)

        p_sum = torch.zeros(1, device=device)
        for p in m.parameters():
            with torch.no_grad():
                if isinstance(p, ShardedTensor):
                    if not p.local_shards():
                        continue
                    p = p.local_tensor()
                p_sum += p.sum()
                p.zero_()
                assert p.sum() == 0
        load_state_dict(state_dict, reader)
        m.load_state_dict(state_dict)

        p_sum_loaded = torch.zeros(1, device=device)
        for p in m.parameters():
            with torch.no_grad():
                if isinstance(p, ShardedTensor):
                    if not p.local_shards():
                        continue
                    p = p.local_tensor()
                p_sum_loaded += p.sum()
        # TODO: debug why failing on OSS
        # assert p_sum.allclose(p_sum_loaded)

    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.cuda.device_count() <= 1` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def _test_checkpoint(self) -> None:
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
            elastic_launch(config=lc, entrypoint=self._run_checkpoint_test)(path)

    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.cuda.device_count() <= 1` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_replicate_trec_shard_composes(self) -> None:
        """
        Ensure that trec_shard followed by replicate results in sparse layers being sharded and
        dense layers being replicated.
        """
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
            elastic_launch(config=lc, entrypoint=self._run_compose_test)(path)
