#!/usr/bin/env python3

import os
import unittest
from collections import OrderedDict
from enum import Enum
from typing import List, Tuple, Optional, Dict, cast, Union

import hypothesis.strategies as st
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import Verbosity, given, settings
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagSharder
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterHints
from torchrec.distributed.tests.test_model import (
    TestSparseNN,
    TestSparseNNBase,
    TestEBCSharder,
    TestEBSharder,
    ModelInput,
)
from torchrec.distributed.tests.test_model_parallel_base import ModelParallelTestBase
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingType,
    ShardingEnv,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.tests.utils import (
    get_free_port,
    skip_if_asan_class,
    init_distributed_single_host,
    seed_and_log,
)


class SharderType(Enum):
    EMBEDDING_BAG = "embedding_bag"
    EMBEDDING_BAG_COLLECTION = "embedding_bag_collection"


def create_test_sharder(
    sharder_type: str, sharding_type: str, kernel_type: str
) -> Union[TestEBSharder, TestEBCSharder]:
    if sharder_type == SharderType.EMBEDDING_BAG.value:
        return TestEBSharder(sharding_type, kernel_type, {"learning_rate": 0.1})
    elif sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value:
        return TestEBCSharder(sharding_type, kernel_type, {"learning_rate": 0.1})
    else:
        raise ValueError(f"Sharder not supported {sharder_type}")


@skip_if_asan_class
class ModelParallelTest(ModelParallelTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
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
    def test_sharding_nccl_rw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ],
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_nccl_dp(
        self, sharder_type: str, sharding_type: str, kernel_type: str
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ],
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
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
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.SPARSE.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_nccl_tw(
        self, sharder_type: str, sharding_type: str, kernel_type: str
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ],
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
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
        local_size=st.sampled_from([2]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_sharding_nccl_twrw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        local_size: int,
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ],
            backend="nccl",
            world_size=2,
            local_size=local_size,
        )

    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
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
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ],
            backend="gloo",
        )

    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
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
    def test_sharding_gloo_dp(
        self, sharder_type: str, sharding_type: str, kernel_type: str
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
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
            env=ShardingEnv.from_process_group(pg),
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([3.2, 3.2, 3.2], dtype=np.float32), sharded_param.detach().numpy()
        )

        # Check that parameters over 'meta' device are allocated and initialized.
        meta_model = MyModel(device="meta", val=7.5)
        sharded_model = DistributedModelParallel(
            meta_model,
            env=ShardingEnv.from_process_group(pg),
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([7.5, 7.5, 7.5], dtype=np.float32), sharded_param.detach().numpy()
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
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        hints: Optional[Dict[str, ParameterHints]] = None,
    ) -> None:
        self._run_multi_process_test(
            # pyre-ignore [6]
            callable=self._test_sharding_single_rank,
            world_size=world_size,
            local_size=local_size,
            model_class=cast(TestSparseNNBase, TestSparseNN),
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            backend=backend,
            optim=EmbOptimType.EXACT_SGD,
            hints=hints,
        )


class ModelParallelStateDictTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            backend = "nccl"
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
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

    def _generate_dmps_and_batch(
        self, sharders: List[ModuleSharder[nn.Module]] = default_sharders
    ) -> Tuple[List[DistributedModelParallel], ModelInput]:
        _, local_batch = ModelInput.generate(
            batch_size=self.batch_size,
            world_size=1,
            num_float_features=self.num_float_features,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )
        batch = local_batch[0].to(self.device)

        # Create two TestSparseNN modules, wrap both in DMP
        dmps = []
        for _ in range(2):
            m = TestSparseNN(
                tables=self.tables,
                num_float_features=self.num_float_features,
                weighted_tables=self.weighted_tables,
                dense_device=self.device,
                sparse_device=torch.device("meta"),
            )
            dmp = DistributedModelParallel(
                module=m,
                init_data_parallel=False,
                device=self.device,
                sharders=sharders,
            )

            with torch.no_grad():
                dmp(batch)
                dmp.init_data_parallel()
            dmps.append(dmp)
        return (dmps, batch)

    def test_meta_device_dmp_state_dict(self) -> None:
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
            sharders=default_sharders,
            device=self.device,
            env=env,
            plan=EmbeddingShardingPlanner(
                world_size=env.world_size,
                compute_device_type=self.device.type,
            ).plan(m1, default_sharders),
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
            sharders=default_sharders,
            device=torch.device("meta"),
            env=env,
            plan=EmbeddingShardingPlanner(
                world_size=env.world_size,
                compute_device_type=self.device.type,
            ).plan(m2, default_sharders),
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
                [EmbeddingBagSharder()],
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_load_state_dict(self, sharders: List[ModuleSharder[nn.Module]]) -> None:
        models, batch = self._generate_dmps_and_batch(sharders)
        m1, m2 = models

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(
            cast("OrderedDict[str, torch.Tensor]", m1.state_dict()), strict=False
        )
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
            strict=False,
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
                SharderType.EMBEDDING_BAG.value,
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
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.SPARSE.value,
                # EmbeddingComputeKernel.BATCHED_DENSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED.value,
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
        state_dict_keys = set(m.state_dict().keys())
        param_keys = {key for (key, _) in m.named_parameters()}
        buffer_keys = {key for (key, _) in m.named_buffers()}
        self.assertEqual(state_dict_keys, {*param_keys, *buffer_keys})
