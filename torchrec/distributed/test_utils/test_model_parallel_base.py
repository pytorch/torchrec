#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from collections import defaultdict
from typing import Callable, cast, Dict, List, Optional, OrderedDict, Tuple

import numpy as np

import torch
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import given, settings, strategies as st, Verbosity
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec import distributed as trec_dist
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    EmbeddingTableConfig,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    EmbeddingBagSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.fused_embeddingbag import ShardedFusedEmbeddingBagCollection
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.test_utils.test_model import (
    _get_default_rtol_and_atol,
    ModelInput,
    TestSparseNN,
    TestSparseNNBase,
)
from torchrec.distributed.test_utils.test_sharding import (
    copy_state_dict,
    create_test_sharder,
    gen_model_and_input,
    ModelInputCallable,
    SharderType,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    EmbeddingBagConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torchrec.test_utils import get_free_port, seed_and_log


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
        dedup_features_names: Optional[List[str]] = None,
        dedup_tables: Optional[List[EmbeddingTableConfig]] = None,
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        generate: ModelInputCallable = ModelInput.generate,
    ) -> None:
        default_rank = 0
        cuda_device = torch.device(f"cuda:{default_rank}")
        torch.cuda.set_device(cuda_device)

        # Generate model & inputs.
        (global_model, _inputs) = gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            dedup_tables=dedup_tables,
            dedup_feature_names=dedup_features_names,
            embedding_groups=embedding_groups,
            world_size=1,  # generate only one copy of feature for inference
            num_float_features=16,
            dense_device=cuda_device,
            sparse_device=cuda_device,
            generate=generate,
        )
        global_model = quantize_callable(global_model)
        local_input = _inputs[0][1][default_rank].to(cuda_device)

        # Shard model.
        if dedup_features_names:
            local_model = model_class(
                tables=cast(
                    List[BaseEmbeddingConfig],
                    tables + dedup_tables if dedup_tables else tables,
                ),
                weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
                dedup_feature_names=dedup_features_names,
                embedding_groups=embedding_groups,
                dense_device=cuda_device,
                sparse_device=torch.device("meta"),
                num_float_features=16,
            )
        else:
            local_model = model_class(
                tables=cast(
                    List[BaseEmbeddingConfig],
                    tables,
                ),
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

        # materialize inference sharded model on one device for dense part
        local_model = local_model.copy(cuda_device)

        # Load model state from the global model.
        copy_state_dict(local_model.state_dict(), global_model.state_dict())

        # Run a single training step of the sharded model.
        with torch.inference_mode():
            shard_pred = local_model(local_input)

        # Run second training step of the unsharded model.
        with torch.inference_mode():
            global_pred = global_model(local_input)

        # Compare predictions of sharded vs unsharded models.
        rtol, atol = _get_default_rtol_and_atol(global_pred, shard_pred)
        torch.testing.assert_close(global_pred, shard_pred, rtol=rtol, atol=atol)


class ModelParallelSparseOnlyBase(unittest.TestCase):
    def tearDown(self) -> None:
        dist.destroy_process_group()

    def test_sharding_ebc_as_top_level(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        if torch.cuda.is_available():
            curr_device = torch.device("cuda:0")
            torch.cuda.set_device(curr_device)
            backend = "nccl"
        else:
            curr_device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)

        embedding_dim = 128
        num_embeddings = 256
        ebc = EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingBagConfig(
                    name="large_table",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    feature_names=["my_feature"],
                    pooling=PoolingType.SUM,
                ),
            ],
        )

        model = DistributedModelParallel(ebc, device=curr_device)

        self.assertTrue(isinstance(model.module, ShardedEmbeddingBagCollection))

    def test_sharding_fused_ebc_as_top_level(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        if torch.cuda.is_available():
            curr_device = torch.device("cuda:0")
            torch.cuda.set_device(curr_device)
            backend = "nccl"
        else:
            curr_device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)

        embedding_dim = 128
        num_embeddings = 256
        ebc = FusedEmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingBagConfig(
                    name="large_table",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    feature_names=["my_feature"],
                    pooling=PoolingType.SUM,
                ),
            ],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
        )

        model = DistributedModelParallel(ebc, device=curr_device)

        self.assertTrue(isinstance(model.module, ShardedFusedEmbeddingBagCollection))


class ModelParallelStateDictBase(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            backend = "nccl"
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)

        num_features = 4
        num_weighted_features = 2
        self.batch_size = 3
        self.num_float_features = 10

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

    def tearDown(self) -> None:
        dist.destroy_process_group()
        del os.environ["NCCL_SOCKET_IFNAME"]
        super().tearDown()

    def _generate_dmps_and_batch(
        self,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        constraints: Optional[Dict[str, trec_dist.planner.ParameterConstraints]] = None,
    ) -> Tuple[List[DistributedModelParallel], ModelInput]:

        if constraints is None:
            constraints = {}
        if sharders is None:
            sharders = get_default_sharders()

        _, local_batch = ModelInput.generate(
            batch_size=self.batch_size,
            world_size=1,
            num_float_features=self.num_float_features,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
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
            m = TestSparseNN(
                tables=self.tables,
                num_float_features=self.num_float_features,
                weighted_tables=self.weighted_tables,
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

        dist.destroy_process_group()
        dist.init_process_group(backend="gloo")

        # Check that already allocated parameters are left 'as is'.
        cpu_model = MyModel(device="cpu", val=3.2)
        sharded_model = DistributedModelParallel(
            cpu_model,
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([3.2, 3.2, 3.2], dtype=np.float32), sharded_param.detach().numpy()
        )

        # Check that parameters over 'meta' device are allocated and initialized.
        meta_model = MyModel(device="meta", val=7.5)
        sharded_model = DistributedModelParallel(
            meta_model,
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([7.5, 7.5, 7.5], dtype=np.float32), sharded_param.detach().numpy()
        )

    def test_meta_device_dmp_state_dict(self) -> None:
        # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
        #  `Optional[ProcessGroup]`.
        env = ShardingEnv.from_process_group(dist.GroupMember.WORLD)

        m1 = TestSparseNN(
            tables=self.tables,
            num_float_features=self.num_float_features,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
        )
        # dmp with real device
        dmp1 = DistributedModelParallel(
            module=m1,
            init_data_parallel=False,
            init_parameters=False,
            sharders=get_default_sharders(),
            device=self.device,
            env=env,
            plan=EmbeddingShardingPlanner(
                topology=Topology(
                    world_size=env.world_size, compute_device=self.device.type
                )
            ).plan(m1, get_default_sharders()),
        )

        m2 = TestSparseNN(
            tables=self.tables,
            num_float_features=self.num_float_features,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
        )
        # dmp with meta device
        dmp2 = DistributedModelParallel(
            module=m2,
            init_data_parallel=False,
            init_parameters=False,
            sharders=get_default_sharders(),
            device=torch.device("meta"),
            env=env,
            plan=EmbeddingShardingPlanner(
                topology=Topology(
                    world_size=env.world_size, compute_device=self.device.type
                )
            ).plan(m2, get_default_sharders()),
        )

        sd1 = dmp1.state_dict()
        for key, v2 in dmp2.state_dict().items():
            v1 = sd1[key]
            if isinstance(v2, nn.parameter.UninitializedParameter) and isinstance(
                v1, nn.parameter.UninitializedParameter
            ):
                continue
            if isinstance(v2, ShardedTensor):
                self.assertTrue(isinstance(v1, ShardedTensor))
                assert len(v2.local_shards()) == 1
                dst = v2.local_shards()[0].tensor
            else:
                dst = v2
            if isinstance(v1, ShardedTensor):
                assert len(v1.local_shards()) == 1
                src = v1.local_shards()[0].tensor
            else:
                src = v1
            self.assertEqual(src.size(), dst.size())

    # pyre-ignore[56]
    @given(
        sharders=st.sampled_from(
            [
                [EmbeddingBagCollectionSharder()],
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_load_state_dict(self, sharders: List[ModuleSharder[nn.Module]]) -> None:
        models, batch = self._generate_dmps_and_batch(sharders)
        m1, m2 = models

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # validate the models are equivalent
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

    # pyre-ignore[56]
    @given(
        sharders=st.sampled_from(
            [
                [EmbeddingBagCollectionSharder()],
                [EmbeddingBagSharder()],
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_load_state_dict_prefix(
        self, sharders: List[ModuleSharder[nn.Module]]
    ) -> None:
        (m1, m2), batch = self._generate_dmps_and_batch(sharders)

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(
            cast("OrderedDict[str, torch.Tensor]", m1.state_dict(prefix="alpha")),
            prefix="alpha",
        )

        # validate the models are equivalent
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

    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_params_and_buffers(
        self, sharder_type: str, sharding_type: str, kernel_type: str
    ) -> None:
        sharders = [
            create_test_sharder(sharder_type, sharding_type, kernel_type),
        ]
        # pyre-ignore[6]
        (m, _), batch = self._generate_dmps_and_batch(sharders=sharders)
        print(f"Sharding Plan: {m._plan}")
        state_dict_keys = set(m.state_dict().keys())
        param_keys = {key for (key, _) in m.named_parameters()}
        buffer_keys = {key for (key, _) in m.named_buffers()}
        self.assertEqual(state_dict_keys, {*param_keys, *buffer_keys})

    # pyre-ignore
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_load_state_dict_cw_multiple_shards(
        self, sharder_type: str, sharding_type: str, kernel_type: str
    ) -> None:
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    fused_params={
                        "learning_rate": 0.2,
                        "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                    },
                ),
            ),
        ]

        constraints = defaultdict(lambda: trec_dist.planner.ParameterConstraints())
        num_cw_shards_per_table = {}
        for table in self.tables + self.weighted_tables:
            constraints[table.name].min_partition = 4
            num_cw_shards_per_table[table.name] = table.embedding_dim // 4

        (m1, m2), batch = self._generate_dmps_and_batch(
            sharders, constraints=constraints
        )

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # load optimizer state dict

        # Check to see that we can load optimizer state
        src_optimizer = m1.fused_optimizer
        dst_optimizer = m2.fused_optimizer

        src_optimizer_state_dict = src_optimizer.state_dict()
        dst_optimizer_state_dict = dst_optimizer.state_dict()
        m2.fused_optimizer.load_state_dict(src_optimizer_state_dict)

        # validate the models are equivalent
        loss1, pred1 = m1(batch)
        loss2, pred2 = m2(batch)
        self.assertTrue(torch.equal(loss1, loss2))
        self.assertTrue(torch.equal(pred1, pred2))

        sd1 = m1.state_dict()
        for key, value in m2.state_dict().items():
            table_name = key.split(".")[-2]
            v2 = sd1[key]
            if isinstance(value, ShardedTensor):
                self.assertEqual(
                    len(value.local_shards()), num_cw_shards_per_table[table_name]
                )
                dst = value.local_shards()[0].tensor
            else:
                dst = value

            if isinstance(v2, ShardedTensor):
                self.assertEqual(
                    len(value.local_shards()), num_cw_shards_per_table[table_name]
                )

                for src_local_shard, dst_local_shard in zip(
                    value.local_shards(), v2.local_shards()
                ):
                    self.assertTrue(
                        torch.equal(src_local_shard.tensor, dst_local_shard.tensor)
                    )
            else:
                src = v2
                self.assertTrue(torch.equal(src, dst))

        for param_name, dst_param_group in dst_optimizer_state_dict.items():
            src_param_group = src_optimizer_state_dict[param_name]

            for state_key, dst_opt_state in dst_param_group.items():
                table_name = state_key.split(".")[-2]
                src_opt_state = src_param_group[state_key]
                if isinstance(dst_opt_state, ShardedTensor):
                    self.assertIsInstance(src_param_group[state_key], ShardedTensor)

                    self.assertEqual(
                        len(dst_opt_state.local_shards()),
                        num_cw_shards_per_table[table_name],
                    )

                    self.assertEqual(
                        len(src_opt_state.local_shards()),
                        num_cw_shards_per_table[table_name],
                    )

                    for src_local_shard, dst_local_shard in zip(
                        src_opt_state.local_shards(), dst_opt_state.local_shards()
                    ):
                        self.assertTrue(
                            torch.equal(src_local_shard.tensor, dst_local_shard.tensor)
                        )
                elif isinstance(dst_opt_state, torch.Tensor):
                    self.assertIsInstance(src_opt_state, torch.Tensor)
