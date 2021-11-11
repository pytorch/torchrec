#!/usr/bin/env python3

import multiprocessing
import os
import unittest
from typing import cast, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterHints
from torchrec.distributed.tests.test_model import (
    ModelInput,
    TestSparseNNBase,
)
from torchrec.distributed.types import (
    ShardedTensor,
    ModuleSharder,
    ShardingEnv,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.tests.utils import (
    get_free_port,
    seed_and_log,
    init_distributed_single_host,
)


class ModelParallelTestBase(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"

    @classmethod
    def _test_sharding_single_rank(
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
        hints: Optional[Dict[str, ParameterHints]] = None,
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

        # Generate model & inputs.
        (global_model, inputs) = cls._gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=world_size,
            num_float_features=16,
        )
        global_model = global_model.to(device)
        global_input = inputs[0][0].to(device)
        local_input = inputs[0][1][rank].to(device)

        # Shard model.
        local_model = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=device,
            sparse_device=torch.device("meta"),
            num_float_features=16,
        )

        planner = EmbeddingShardingPlanner(world_size, device.type, hints)
        plan = planner.collective_plan(local_model, sharders, pg)

        local_model = DistributedModelParallel(
            local_model,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            sharders=sharders,
            device=device,
        )

        dense_optim = KeyedOptimizerWrapper(
            dict(local_model.named_parameters()),
            lambda params: torch.optim.SGD(params, lr=0.1),
        )
        local_opt = CombinedOptimizer([local_model.fused_optimizer, dense_optim])

        # Load model state from the global model.
        cls._copy_state_dict(local_model.state_dict(), global_model.state_dict())

        # Run a single training step of the sharded model.
        local_pred = cls._gen_full_pred_after_one_step(
            local_model, local_opt, local_input
        )
        all_local_pred = []
        for _ in range(world_size):
            all_local_pred.append(torch.empty_like(local_pred))
        dist.all_gather(all_local_pred, local_pred, group=pg)

        # Run second training step of the unsharded model.
        assert optim == EmbOptimType.EXACT_SGD
        global_opt = torch.optim.SGD(global_model.parameters(), lr=0.1)
        global_pred = cls._gen_full_pred_after_one_step(
            global_model, global_opt, global_input
        )

        # Compare predictions of sharded vs unsharded models.
        torch.testing.assert_allclose(global_pred, torch.cat(all_local_pred))

    def _run_multi_process_test(
        self,
        callable: Callable[
            [int, int, List[ModuleSharder[nn.Module]], List[torch.Tensor]], None
        ],
        world_size: int,
        sharders: List[ModuleSharder[nn.Module]],
        tables: List[EmbeddingTableConfig],
        backend: str,
        optim: EmbOptimType,
        model_class: TestSparseNNBase,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        hints: Optional[Dict[str, ParameterHints]] = None,
        local_size: Optional[int] = None,
    ) -> None:
        ctx = multiprocessing.get_context("spawn")
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=callable,
                args=(
                    rank,
                    world_size,
                    model_class,
                    embedding_groups,
                    tables,
                    sharders,
                    backend,
                    optim,
                    weighted_tables,
                    hints,
                ),
                kwargs={
                    "local_size": local_size,
                },
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)

    @staticmethod
    def _generate_inputs(
        world_size: int,
        tables: List[EmbeddingTableConfig],
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        batch_size: int = 4,
        num_float_features: int = 16,
    ) -> Tuple[ModelInput, List[ModelInput]]:
        return ModelInput.generate(
            batch_size=batch_size,
            world_size=world_size,
            num_float_features=num_float_features,
            tables=tables,
            weighted_tables=weighted_tables or [],
        )

    @classmethod
    def _gen_model_and_input(
        cls,
        model_class: TestSparseNNBase,
        tables: List[EmbeddingTableConfig],
        embedding_groups: Dict[str, List[str]],
        world_size: int,
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        num_float_features: int = 16,
    ) -> Tuple[nn.Module, List[Tuple[ModelInput, List[ModelInput]]]]:
        torch.manual_seed(0)

        model = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            num_float_features=num_float_features,
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
        )
        inputs = [
            cls._generate_inputs(
                world_size=world_size,
                tables=tables,
                weighted_tables=weighted_tables,
                num_float_features=num_float_features,
            )
        ]
        return (model, inputs)

    @classmethod
    def _gen_full_pred_after_one_step(
        cls,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        input: ModelInput,
    ) -> torch.Tensor:
        # Run a single training step of the global model.
        opt.zero_grad()
        model.train(True)
        loss, _ = model(input)
        loss.backward()
        opt.step()

        # Run a forward pass of the global model.
        with torch.no_grad():
            model.train(False)
            full_pred = model(input)
            return full_pred

    @classmethod
    def _copy_state_dict(
        cls,
        loc: Dict[str, Union[torch.Tensor, ShardedTensor]],
        glob: Dict[str, torch.Tensor],
    ) -> None:
        for name, tensor in loc.items():
            assert name in glob
            global_tensor = glob[name]
            if isinstance(global_tensor, ShardedTensor):
                global_tensor = global_tensor.local_shards()[0].tensor
            if isinstance(tensor, ShardedTensor):
                for local_shard in tensor.local_shards():
                    assert global_tensor.ndim == local_shard.tensor.ndim
                    shard_meta = local_shard.metadata
                    t = global_tensor.detach()
                    if t.ndim == 1:
                        t = t[
                            shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                            + local_shard.tensor.shape[0]
                        ]
                    elif t.ndim == 2:
                        t = t[
                            shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                            + local_shard.tensor.shape[0],
                            shard_meta.shard_offsets[1] : shard_meta.shard_offsets[1]
                            + local_shard.tensor.shape[1],
                        ]
                    else:
                        raise ValueError("Tensors with ndim > 2 are not supported")
                    local_shard.tensor.copy_(t)
            else:
                tensor.copy_(global_tensor)
