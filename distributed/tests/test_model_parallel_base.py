#!/usr/bin/env python3

import multiprocessing
import os
import unittest
from typing import cast, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
        outputs: List[torch.Tensor],
        backend: str,
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        hints: Optional[Dict[str, ParameterHints]] = None,
        local_size: Optional[int] = None,
    ) -> None:
        # Generate model & inputs.
        (global_model, inputs) = cls._gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=world_size,
        )
        local_input = inputs[0][1][rank]
        # Instantiate lazy modules.
        with torch.no_grad():
            global_model(local_input)

        if backend == "nccl":
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
            local_input = local_input.to(device, True)
        else:
            device = torch.device("cpu")

        pg = init_distributed_single_host(
            rank=rank,
            world_size=world_size,
            backend=backend,
            local_size=local_size,
        )
        # Override local_size after pg construction because unit test device count
        # is larger than local_size setup. This can be problematic for twrw because
        # we have ShardedTensor placement check.
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        # Shard model.
        local_model = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )

        planner = EmbeddingShardingPlanner(pg, device, hints)
        plan = planner.collective_plan(local_model, sharders)

        local_model = DistributedModelParallel(
            local_model,
            pg=pg,
            plan=plan,
            sharders=sharders,
            init_data_parallel=False,
            device=device,
        )
        # Instantiate lazy modules.
        with torch.no_grad():
            local_model(local_input)
        local_model.init_data_parallel()

        # Load state from the global model.
        global_state_dict = global_model.state_dict()
        for name, tensor in local_model.state_dict().items():
            assert name in global_state_dict
            global_tensor = global_state_dict[name]
            if isinstance(tensor, ShardedTensor):
                for local_shard in tensor.local_shards():
                    assert global_tensor.ndim == local_shard.tensor.ndim
                    shard_meta = local_shard.metadata
                    t = global_tensor.detach()
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0],
                        shard_meta.shard_offsets[1] : shard_meta.shard_offsets[1]
                        + local_shard.tensor.shape[1],
                    ]
                    local_shard.tensor.copy_(t)
            else:
                tensor.copy_(global_tensor)

        # Run a single training step of the sharded model.
        dense_optim = KeyedOptimizerWrapper(
            dict(local_model.named_parameters()),
            lambda params: torch.optim.SGD(params, lr=0.1),
        )
        opt = CombinedOptimizer([local_model.fused_optimizer, dense_optim])
        opt.zero_grad()
        loss, _ = local_model(local_input)
        loss.backward()
        opt.step()

        # Run a forward pass the the sharded model.
        with torch.no_grad():
            local_model.train(False)
            pred = local_model(local_input)
            outputs[rank] = pred.cpu()

        # Make sure that optimizer params FQN match model params FQN.
        opt_keys = set()
        for param_group in opt.state_dict()["param_groups"]:
            for key in param_group["params"]:
                opt_keys.add(key)
        model_keys = set()
        for key in local_model.state_dict().keys():
            model_keys.add(key)
        np.testing.assert_array_equal(sorted(opt_keys), sorted(model_keys))
        # Make sure that named params FQN match model params FQN.
        for key, _ in local_model.named_parameters():
            assert key in model_keys

    def _run_multi_process_test(
        self,
        callable: Callable[
            [int, int, List[ModuleSharder[nn.Module]], List[torch.Tensor]], None
        ],
        world_size: int,
        sharders: List[ModuleSharder[nn.Module]],
        tables: List[EmbeddingTableConfig],
        backend: str,
        model_class: TestSparseNNBase,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        hints: Optional[Dict[str, ParameterHints]] = None,
        local_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        mgr = multiprocessing.Manager()
        outputs = mgr.list([torch.tensor([])] * world_size)
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
                    outputs,
                    backend,
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
        return list(outputs)

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
    ) -> Tuple[nn.Module, List[Tuple[ModelInput, List[ModelInput]]]]:
        torch.manual_seed(0)

        model = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
        )
        inputs = [
            cls._generate_inputs(
                world_size=world_size,
                tables=tables,
                weighted_tables=weighted_tables,
            )
        ]
        return (model, inputs)

    @classmethod
    def _gen_full_pred_after_one_step(
        cls,
        global_model: nn.Module,
        inputs: List[Tuple[ModelInput, List[ModelInput]]],
    ) -> torch.Tensor:
        global_input = inputs[0][0]

        # Instantiate lazy modules.
        with torch.no_grad():
            global_model(global_input)

        # Run a single training step of the global model.
        opt = torch.optim.SGD(global_model.parameters(), lr=0.1)
        opt.zero_grad()
        loss, _ = global_model(global_input)
        loss.backward()
        opt.step()

        # Run a forward pass of the global model.
        with torch.no_grad():
            global_model.train(False)
            full_pred = global_model(global_input)
            return full_pred
