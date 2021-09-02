#!/usr/bin/env python3

import multiprocessing
import os
import unittest
from typing import List, Tuple, Optional, Callable

import hypothesis.strategies as st
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from hypothesis import Verbosity, given, settings
from torchrec.distributed.collective_utils import (
    invoke_on_rank_and_broadcast_result,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.tests.test_model import (
    TestSparseNN,
    TestEBCSharder,
    ModelInput,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingPlan,
    ShardingPlanner,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.tests.utils import (
    get_free_port,
    skip_if_asan_class,
    init_distributed_single_host,
    seed_and_log,
)


@skip_if_asan_class
class ModelParallelTest(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.SPARSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_nccl_rw(self, sharding_type: str, kernel_type: str) -> None:
        self._test_sharding(
            sharders=[
                TestEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
                # TODO debug and enable full test for sparse kernel
                # EmbeddingComputeKernel.SPARSE.value,
                # EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_nccl_dp(self, sharding_type: str, kernel_type: str) -> None:
        self._test_sharding(
            sharders=[
                TestEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
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
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.SPARSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_nccl_tw(self, sharding_type: str, kernel_type: str) -> None:
        self._test_sharding(
            sharders=[
                TestEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.SPARSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_sharding_nccl_twrw(
        self,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
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
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.SPARSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_sharding_gloo_tw(
        self,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="gloo",
        )

    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
                # TODO dp+batch_fused is numerically buggy in cpu
                # EmbeddingComputeKernel.SPARSE.value,
                # EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_sharding_gloo_dp(self, sharding_type: str, kernel_type: str) -> None:
        self._test_sharding(
            sharders=[
                TestEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="gloo",
        )

    def test_parameter_init(self) -> None:
        class MyModel(nn.Module):
            def __init__(self, device: str, val: float) -> None:
                super().__init__()
                self.p = nn.Parameter(
                    torch.empty(3, dtype=torch.float32, device=device)
                )
                self.val = val
                self.reset_parameters()

            def reset_parameters(self) -> None:
                nn.init.constant_(self.p, self.val)

        pg = init_distributed_single_host(rank=0, world_size=1, backend="gloo")

        # Check that already allocated parameters are left 'as is'.
        cpu_model = MyModel(device="cpu", val=3.2)
        sharded_model = DistributedModelParallel(
            cpu_model,
            pg=pg,
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([3.2, 3.2, 3.2], dtype=np.float32), sharded_param.detach().numpy()
        )

        # Check that parameters over 'meta' device are allocated and initialized.
        meta_model = MyModel(device="meta", val=7.5)
        sharded_model = DistributedModelParallel(
            meta_model,
            pg=pg,
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([7.5, 7.5, 7.5], dtype=np.float32), sharded_param.detach().numpy()
        )

    @seed_and_log
    def setUp(self) -> None:
        torch.use_deterministic_algorithms(True)
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())

        num_features = 4
        num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]

    def _run_multi_process_test(
        self,
        callable: Callable[[int, int, List[TestEBCSharder], List[torch.Tensor]], None],
        world_size: int,
        sharders: List[ModuleSharder[nn.Module]],
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        backend: str,
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
                    tables,
                    weighted_tables,
                    sharders,
                    outputs,
                    backend,
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
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        batch_size: int = 4,
        num_float_features: int = 16,
    ) -> Tuple[ModelInput, List[ModelInput]]:
        return ModelInput.generate(
            batch_size=batch_size,
            world_size=world_size,
            num_float_features=num_float_features,
            tables=tables,
            weighted_tables=weighted_tables,
        )

    def _test_sharding(
        self,
        sharders: List[TestEBCSharder],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
    ) -> None:

        # Run distributed training and collect predictions.
        local_pred = self._run_multi_process_test(
            # pyre-ignore [6]
            callable=self._test_sharding_single_rank,
            world_size=world_size,
            local_size=local_size,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            # pyre-fixme[6]: Expected `List[ModuleSharder[nn.Module]]` for 4th param
            #  but got `List[TestEBCSharder]`.
            sharders=sharders,
            backend=backend,
        )

        full_pred = self._gen_full_pred_after_one_step(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            world_size=world_size,
        )
        if full_pred is not None:
            torch.testing.assert_allclose(full_pred, torch.cat(local_pred))

    @classmethod
    def _test_sharding_single_rank(
        cls,
        rank: int,
        world_size: int,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        sharders: List[ModuleSharder[nn.Module]],
        outputs: List[torch.Tensor],
        backend: str,
        planner: Optional[ShardingPlanner] = None,
        local_size: Optional[int] = None,
    ) -> None:
        # Generate model & inputs.
        (global_model, inputs) = cls._gen_model_and_input(
            tables=tables, weighted_tables=weighted_tables, world_size=world_size
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

        # Shard model.
        local_model = TestSparseNN(
            sparse_device=torch.device("meta"),
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=device,
        )
        plan: Optional[ShardingPlan]
        if planner:
            plan = invoke_on_rank_and_broadcast_result(
                pg if pg else dist.ProcessGroup.WORLD,
                0,
                planner.plan,
                local_model,
                sharders,
            )
        else:
            plan = None
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
                assert global_tensor.ndim == tensor.local_shard.ndim
                shard_meta = tensor.sharding_metadata.shards[rank]
                t = global_tensor.detach()
                t = t[
                    shard_meta.offsets[0] : shard_meta.offsets[0]
                    + tensor.local_shard.shape[0],
                    shard_meta.offsets[1] : shard_meta.offsets[1]
                    + tensor.local_shard.shape[1],
                ]
                tensor.local_shard.copy_(t)
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

    @classmethod
    def _gen_model_and_input(
        cls,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        world_size: int,
    ) -> Tuple[nn.Module, List[Tuple[ModelInput, List[ModelInput]]]]:
        torch.manual_seed(0)
        model = TestSparseNN(tables=tables, weighted_tables=weighted_tables)
        inputs = [
            cls._generate_inputs(
                world_size=world_size, tables=tables, weighted_tables=weighted_tables
            )
        ]
        return (model, inputs)

    @classmethod
    def _gen_full_pred_after_one_step(
        cls,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        world_size: int,
    ) -> torch.Tensor:
        (global_model, inputs) = cls._gen_model_and_input(
            tables=tables, weighted_tables=weighted_tables, world_size=world_size
        )
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
