#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import unittest
from typing import cast, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.comm import _INTRA_PG, _CROSS_PG
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestSparseNNBase,
)
from torchrec.distributed.types import (
    ShardingType,
    ShardingPlan,
    ShardedTensor,
    ModuleSharder,
    ShardingEnv,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.test_utils import (
    get_free_port,
    seed_and_log,
    init_distributed_single_host,
)


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


def _gen_model_and_input(
    model_class: TestSparseNNBase,
    tables: List[EmbeddingTableConfig],
    embedding_groups: Dict[str, List[str]],
    world_size: int,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    num_float_features: int = 16,
    dense_device: Optional[torch.device] = None,
    sparse_device: Optional[torch.device] = None,
) -> Tuple[nn.Module, List[Tuple[ModelInput, List[ModelInput]]]]:
    torch.manual_seed(0)

    model = model_class(
        tables=cast(List[BaseEmbeddingConfig], tables),
        num_float_features=num_float_features,
        weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
        embedding_groups=embedding_groups,
        dense_device=dense_device,
        sparse_device=sparse_device,
    )
    inputs = [
        _generate_inputs(
            world_size=world_size,
            tables=tables,
            weighted_tables=weighted_tables,
            num_float_features=num_float_features,
        )
    ]
    return (model, inputs)


def _copy_state_dict(
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


class ModelParallelTestBase(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def tearDown(self) -> None:
        torch.use_deterministic_algorithms(False)
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        if torch.cuda.is_available():
            os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
        super().tearDown()

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
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        local_size: Optional[int] = None,
    ) -> None:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False

        """
        Override local_size after pg construction because unit test device count is
        larger than local_size setup. This can be problematic for twrw because we have
        ShardedTensor placement check.

        TODO (T108556130) Mock out functions in comm.py instead of overriding env vars
        """
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size or world_size)
        if local_size is not None:
            os.environ["LOCAL_RANK"] = str(rank % local_size)
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
        (global_model, inputs) = _gen_model_and_input(
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

        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size, device.type, local_world_size=local_size),
            constraints=constraints,
        )
        plan: ShardingPlan = planner.collective_plan(local_model, sharders, pg)
        """
        Simulating multiple nodes on a single node. However, metadata information and
        tensor placement must still be consistent. Here we overwrite this to do so.

        NOTE:
            inter/intra process groups should still behave as expected.

        TODO: may need to add some checks that only does this if we're running on a
        single GPU (which should be most cases).
        """
        for group in plan.plan:
            for _, parameter_sharding in plan.plan[group].items():
                if (
                    parameter_sharding.sharding_type
                    in {
                        ShardingType.TABLE_ROW_WISE.value,
                        ShardingType.TABLE_COLUMN_WISE.value,
                    }
                    and device.type != "cpu"
                ):
                    sharding_spec = parameter_sharding.sharding_spec
                    if sharding_spec is not None:
                        # pyre-ignore
                        for shard in sharding_spec.shards:
                            placement = shard.placement
                            rank: Optional[int] = placement.rank()
                            assert rank is not None
                            shard.placement = torch.distributed._remote_device(
                                f"rank:{rank}/cuda:{rank}"
                            )

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
        _copy_state_dict(local_model.state_dict(), global_model.state_dict())

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

        if _INTRA_PG is not None:
            dist.destroy_process_group(_INTRA_PG)
        if _CROSS_PG is not None:
            dist.destroy_process_group(_CROSS_PG)
        dist.destroy_process_group(pg)

        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

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
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        local_size: Optional[int] = None,
    ) -> None:
        ctx = multiprocessing.get_context("forkserver")
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
                    constraints,
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
        # pyre-fixme[20]: Argument `closure` expected.
        opt.step()

        # Run a forward pass of the global model.
        with torch.no_grad():
            model.train(False)
            full_pred = model(input)
            return full_pred


class InferenceModelParallelTestBase(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def tearDown(self) -> None:
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
        super().tearDown()

    def _test_sharded_forward(
        self,
        world_size: int,
        model_class: TestSparseNNBase,
        embedding_groups: Dict[str, List[str]],
        tables: List[EmbeddingTableConfig],
        sharders: List[ModuleSharder[nn.Module]],
        quantize_callable: Callable[[nn.Module], nn.Module],
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None:
        default_rank = 0
        cuda_device = torch.device(f"cuda:{default_rank}")
        torch.cuda.set_device(cuda_device)

        # Generate model & inputs.
        (global_model, inputs) = _gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=1,  # generate only one copy of feature for inference
            num_float_features=16,
            dense_device=cuda_device,
            sparse_device=cuda_device,
        )
        global_model = quantize_callable(global_model)
        local_input = inputs[0][1][default_rank].to(cuda_device)

        # Shard model.
        local_model = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=cuda_device,
            sparse_device=torch.device("meta"),
            num_float_features=16,
        )
        local_model = quantize_callable(local_model)

        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size, "cuda"),
            constraints=constraints,
        )
        plan: ShardingPlan = planner.plan(local_model, sharders)

        # Generate a sharded model on a default rank.
        local_model = DistributedModelParallel(
            local_model,
            env=ShardingEnv.from_local(world_size, default_rank),
            plan=plan,
            sharders=sharders,
            init_data_parallel=False,
        )

        # Load model state from the global model.
        _copy_state_dict(local_model.state_dict(), global_model.state_dict())

        # Run a single training step of the sharded model.
        with torch.inference_mode():
            shard_pred = local_model(local_input)

        # Run second training step of the unsharded model.
        with torch.inference_mode():
            global_pred = global_model(local_input)

        # Compare predictions of sharded vs unsharded models.
        torch.testing.assert_allclose(global_pred, shard_pred)
