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
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
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
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.optim.warmup import WarmupOptimizer, WarmupPolicy, WarmupStage
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
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            m.sparse.parameters(),
            {"lr": 0.01},
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
            # TODO enable once works
            # use_orig_params=True,
        )
        dense_opt = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(m.named_parameters(), include=False)),
            lambda params: torch.optim.Adam(
                params,
                lr=0.01,
                betas=(0.9, 0.999),
                eps=1e-5,
                weight_decay=1e-05,
            ),
        )
        optims = []
        sparse_grad_parameter_names = set()
        for name, p in in_backward_optimizer_filter(m.named_parameters(), include=True):
            # Add learning rate scheduler
            warmup = WarmupOptimizer(
                # pyre-ignore
                p._in_backward_optimizers[0],
                [
                    WarmupStage(
                        policy=WarmupPolicy.LINEAR,
                        value=0.1,
                        lr_scale=1.0,
                    )
                ],
                lr=0.01,  # initial learning rate
                param_name="__sparse_warmup",
            )
            optims.append((name, warmup))
            sparse_grad_parameter_names.add(name)
        fused_opt_scheduled = CombinedOptimizer(optims)
        dense_opt_scheduled = WarmupOptimizer(
            dense_opt,
            [
                WarmupStage(
                    policy=WarmupPolicy.LINEAR,
                    value=0.15,
                    lr_scale=1.0,
                )
            ],
            lr=0.01,
            param_name="__dense_warmup",
        )
        opt: CombinedOptimizer = CombinedOptimizer(
            [fused_opt_scheduled, (dense_opt_scheduled)]
        )
        # Runs a dummy optimizer step, which allows to initialize
        #   optimizer state, which is typically lazy.
        # This allows us to do in-place loading of optimizer state from a checkpoint.
        # Remark that fused optimizer needs speical case as its states are ShardedTensors.
        # This is the reason we need to pass the sparse_grad_parameter_names as parameters.
        opt.init_state(sparse_grad_parameter_names)
        opt.save_param_groups(True)

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
        opt.step()

        # TODO uncomment after fixing
        # buffer = io.BytesIO()
        # # Use FSDP state_dict() API instead of default
        # opt_state_dict = FullyShardedDataParallel._optim_state_dict(sharded_m, opt)
        # torch.save(opt_state_dict, buffer)
        # buffer.seek(0)

        writer = FileSystemWriter(path=path)
        reader = FileSystemReader(path=path)
        # TODO add StateDictType.SHARDED_STATE_DICT test
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
        o_sum = torch.zeros(1, device=device)
        for p_v in opt.state_dict()["state"].values():
            for t in p_v.values():
                if isinstance(t, ShardedTensor):
                    if not t.local_shards():
                        continue
                    t = t.local_tensor()
                o_sum += t.sum()
                t.zero_()
                assert t.sum() == 0

        state_dict = sharded_m.state_dict()
        load_state_dict(state_dict, reader)
        missing, unexpected = sharded_m.load_state_dict(state_dict)
        assert len(missing) == 0 and len(unexpected) == 0

        p_sum_loaded = torch.zeros(1, device=device)
        for p in sharded_m.parameters():
            with torch.no_grad():
                if isinstance(p, ShardedTensor):
                    if not p.local_shards():
                        continue
                    p = p.local_tensor()
                p_sum_loaded += p.sum()
        assert p_sum.allclose(p_sum_loaded)

        # Use FSDP load_state_dict() API instead of default
        # TODO uncomment after fixing
        # FullyShardedDataParallel._load_optim_state_dict_pre_hook(sharded_m, opt, torch.load(buffer))
        # o_sum_loaded = torch.zeros(1, device=device)
        # for p_v in opt.state_dict()["state"].values():
        #     for t in p_v.values():
        #         if isinstance(t, ShardedTensor):
        #             if not t.local_shards():
        #                 continue
        #             t = t.local_tensor()
        #         o_sum_loaded += t.sum()
        # assert o_sum.allclose(o_sum_loaded)

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
            plan=row_wise(),
        )
        m.sparse.weighted_ebc = trec_shard(
            module=m.sparse.weighted_ebc,
            device=device,
            plan=row_wise(),
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
