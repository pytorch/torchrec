#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

from functools import partial
from typing import Any, Dict, List, Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn

from hypothesis import assume, given, settings, Verbosity
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.utils.data import IterableDataset
from torchrec import distributed as trec_dist
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)

from torchrec.distributed.shard import shard
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
    ModuleSharder,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import (
    assert_state_buffers_parameters_equal,
    skip_if_asan_class,
)


def _optional_equals(t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]) -> bool:
    if t1 is None:
        return t2 is None
    return t2 is not None and torch.equal(t1, t2)


def _test_sharding(  # noqa C901
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    is_data_parallel: bool = False,
    use_apply_optimizer_in_backward: bool = False,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        model = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )
        unsharded_model = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        if use_apply_optimizer_in_backward:
            apply_optimizer_in_backward(
                torch.optim.SGD,
                model.embedding_bags["table_0"].parameters(),
                {"lr": 1.0},
            )
            apply_optimizer_in_backward(
                torch.optim.SGD,
                model.embedding_bags["table_1"].parameters(),
                {"lr": 4.0},
            )
            apply_optimizer_in_backward(
                torch.optim.SGD,
                unsharded_model.embedding_bags["table_0"].parameters(),
                {"lr": 1.0},
            )
            apply_optimizer_in_backward(
                torch.optim.SGD,
                unsharded_model.embedding_bags["table_1"].parameters(),
                {"lr": 4.0},
            )
        plan: ShardingPlan = planner.collective_plan(model, [sharder], ctx.pg)
        sharded_model = shard(
            module=model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan.get_plan_for_module(""),
            sharder=sharder,
            device=ctx.device,
        )

        if not use_apply_optimizer_in_backward:
            unsharded_model_optimizer = torch.optim.SGD(
                unsharded_model.parameters(), lr=0.01
            )
            sharded_model_optimizer = torch.optim.SGD(
                sharded_model.parameters(), lr=0.01
            )

        assert isinstance(sharded_model, ShardedEmbeddingBagCollection)

        unsharded_model.load_state_dict(copy.deepcopy(initial_state_dict))
        copy_state_dict(sharded_model.state_dict(), copy.deepcopy(initial_state_dict))

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        unsharded_model_params = dict(unsharded_model.named_parameters())

        if not use_apply_optimizer_in_backward:
            unsharded_model_optimizer.zero_grad()
            sharded_model_optimizer.zero_grad()

        if is_data_parallel:
            for fqn, param in sharded_model.named_parameters():
                assert _optional_equals(param.grad, unsharded_model_params[fqn].grad)

        unsharded_model_pred_kt = []
        for unsharded_rank in range(ctx.world_size):
            # simulate the unsharded model run on the entire batch
            unsharded_model_pred_kt.append(
                unsharded_model(kjt_input_per_rank[unsharded_rank])
            )

        all_unsharded_preds = []
        for unsharded_rank in range(ctx.world_size):
            unsharded_model_pred_kt_mini_batch = unsharded_model_pred_kt[
                unsharded_rank
            ].to_dict()

            all_unsharded_preds.extend(
                [
                    unsharded_model_pred_kt_mini_batch[feature]
                    for feature in feature_keys
                ]
            )
            if unsharded_rank == ctx.rank:
                unsharded_model_pred = torch.stack(
                    [
                        unsharded_model_pred_kt_mini_batch[feature]
                        for feature in feature_keys
                    ]
                )
        # sharded model
        # each rank gets a subbatch
        sharded_model_pred_kt = sharded_model(kjt_input_per_rank[ctx.rank]).to_dict()

        sharded_model_pred = torch.stack(
            [sharded_model_pred_kt[feature] for feature in feature_keys]
        )

        print("sharded model pred before manifesting", sharded_model_pred)
        # DO ANY DENSE COMPUTATION HERE?!?!?!?!?
        # there's a bug here, the 2* doesn't get reflecting in gradient..., e.g. it's not floying to the PACT
        # sharded_model_pred = 2 * sharded_model_pred.manifest()
        # unsharded_model_pred = 2 * unsharded_model_pred

        # TODO hide this somewhere else
        sharded_model_pred = sharded_model_pred.manifest()

        # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
        # in normal author modelling code this won't be an issue because each rank would individually create
        # their model. output from sharded_pred is correctly on the correct device.
        # Compare predictions of sharded vs unsharded models.
        print("sharded model pred", sharded_model_pred)
        print("unsharded model pred", unsharded_model_pred)

        torch.testing.assert_close(sharded_model_pred.cpu(), unsharded_model_pred.cpu())

        sharded_model_pred.sum().backward()

        all_unsharded_preds = torch.stack(all_unsharded_preds)
        _sum = all_unsharded_preds.sum()
        if is_data_parallel:
            _sum /= world_size
        _sum.backward()

        if is_data_parallel:
            for fqn, param in sharded_model.named_parameters():
                assert _optional_equals(param.grad, unsharded_model_params[fqn].grad)

        if not use_apply_optimizer_in_backward:
            unsharded_model_optimizer.step()
            sharded_model_optimizer.step()

        # check nn.Module APIs look the same
        assert_state_buffers_parameters_equal(unsharded_model, sharded_model)

        for fqn in unsharded_model.state_dict():
            unsharded_state = unsharded_model.state_dict()[fqn]
            sharded_state = sharded_model.state_dict()[fqn]
            if is_data_parallel:
                torch.testing.assert_close(unsharded_state, sharded_state)
            else:
                out = (
                    torch.zeros(size=unsharded_state.shape, device=ctx.device)
                    if ctx.rank == 0
                    else None
                )
                sharded_state.gather(out=out)
                if ctx.rank == 0:
                    print("unsharded state", unsharded_state)
                    print("sharded state", out)
                    torch.testing.assert_close(
                        unsharded_state,
                        out,
                    )


class TestEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


@skip_if_asan_class
class ShardedEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                # ShardingType.ROW_WISE.value,
                # ShardingType.COLUMN_WISE.value,
                # ShardingType.DATA_PARALLEL.value,
            ]
        ),
        use_apply_optimizer_in_backward=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_ebc(
        self,
        sharding_type: str,
        use_apply_optimizer_in_backward: bool,
    ) -> None:

        # TODO DistributedDataParallel needs full support of registering fused optims before we can enable this.
        assume(
            not (
                use_apply_optimizer_in_backward
                and sharding_type == ShardingType.DATA_PARALLEL.value
            ),
        )
        assume(not use_apply_optimizer_in_backward)

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=4,
                num_embeddings=4,
                init_fn=partial(torch.nn.init.normal_, mean=0.0, std=1.5),
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=4,
                num_embeddings=4,
                init_fn=partial(torch.nn.init.uniform_, a=-0.036, b=0.036),
            ),
        ]

        constraints = {
            "table_0": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LRU,
                    load_factor=0.1,
                    reserved_memory=8.0,
                    precision=DataType.FP16,
                ),
                enforce_hbm=True,
                stochastic_rounding=False,
                bounds_check_mode=BoundsCheckMode.IGNORE,
            ),
            "table_1": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.2,
                    reserved_memory=0.0,
                    precision=DataType.FP16,
                ),
                enforce_hbm=False,
                stochastic_rounding=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            ),
        }

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]       [0,1,2,3]
        # "feature_1"   [2, 3]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([0, 1, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1]),
            ),
        ]
        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            initial_state_dict={
                "embedding_bags.table_0.weight": torch.Tensor(
                    [
                        [1, 1, 1, 1],
                        [2, 2, 2, 2],
                        [4, 4, 4, 4],
                        [8, 8, 8, 8],
                    ]
                ),
                "embedding_bags.table_1.weight": torch.Tensor(
                    [
                        [101, 101, 101, 101],
                        [102, 102, 102, 102],
                        [104, 104, 104, 104],
                        [108, 108, 108, 108],
                    ]
                ),
            },
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=TestEmbeddingBagCollectionSharder(sharding_type=sharding_type),
            backend="nccl"
            if (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
            else "gloo",
            constraints=constraints,
            is_data_parallel=(sharding_type == ShardingType.DATA_PARALLEL.value),
            use_apply_optimizer_in_backward=use_apply_optimizer_in_backward,
        )


####


def _get_random_dataset(
    num_embeddings: int,
    batch_size: int = 32,
) -> IterableDataset:
    return RandomRecDataset(
        keys=DEFAULT_CAT_NAMES,
        batch_size=batch_size,
        hash_size=num_embeddings,
        ids_per_feature=1,
        num_dense=len(DEFAULT_INT_NAMES),
    )


def train_dlrm(rank: int, world_size: int, backend: str, local_size: int) -> None:
    num_embeddings: int = 1024**2
    embedding_dim: int = 128
    dense_arch_layer_sizes = None
    over_arch_layer_sizes = None
    learning_rate: float = 0.1
    num_iterations: int = 1000
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        """
        Constructs and trains a DLRM model (using random dummy data). Each script is run on each process (rank) in SPMD fashion.
        The embedding layers will be sharded across available ranks

        qcomm_forward_precision: Compression used in forwards pass. FP16 is the recommended usage. INT8 and FP8 are in development, but feel free to try them out.
        qcomm_backward_precision: Compression used in backwards pass. We recommend using BF16 to ensure training stability.

        The effects of quantized comms will be most apparent in large training jobs across multiple nodes where inter host communication is expensive.
        """
        if dense_arch_layer_sizes is None:
            dense_arch_layer_sizes = [64, embedding_dim]
        if over_arch_layer_sizes is None:
            over_arch_layer_sizes = [64, 1]

        # Init process_group , device, rank, backend
        rank = ctx.rank
        device = ctx.device
        # Construct DLRM module
        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]
        dlrm_model = DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=device,
        )
        train_model = DLRMTrain(dlrm_model)

        apply_optimizer_in_backward(
            RowWiseAdagrad,
            train_model.model.sparse_arch.parameters(),
            {"lr": learning_rate},
        )
        # qcomm_codecs_registry = (
        #     get_qcomm_codecs_registry(
        #         qcomms_config=QCommsConfig(
        #             # pyre-ignore
        #             forward_precision=qcomm_forward_precision,
        #             # pyre-ignore
        #             backward_precision=qcomm_backward_precision,
        #         )
        #     )
        #     if backend == "nccl"
        #     else None
        # )
        sharder = EmbeddingBagCollectionSharder()
        # qcomm_codecs_registry=qcomm_codecs_registry)

        model = DistributedModelParallel(
            module=train_model,
            device=ctx.device,
            # pyre-ignore
            sharders=[sharder],
        )

        non_fused_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(model.named_parameters())),
            lambda params: torch.optim.Adagrad(params, lr=learning_rate),
        )
        # # Overlap comm/compute/device transfer during training through train_pipeline
        # train_pipeline = TrainPipelineSparseDist(
        #     model,
        #     non_fused_optimizer,
        #     device,
        # )

        # train model
        train_iterator = iter(
            _get_random_dataset(
                num_embeddings=num_embeddings,
            )
        )

        device = ctx.device
        print(model(next(train_iterator).to(device))[0])  # warmup, input dists
        # train_model.forward = torch.compile(fullgraph=True, backend="eager")(train_model.forward)
        # print(model(next(train_iterator).to(device)))
        # print(model(next(train_iterator).to(device)))
        # print(model(next(train_iterator).to(device)))
        # # for _ in tqdm(range(int(num_iterations)), mininterval=5.0):
        #     train_pipeline.progress(train_iterator)


@skip_if_asan_class
class DLRMTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    def test_dlrm(
        self,
    ) -> None:

        WORLD_SIZE = 2
        self._run_multi_process_test(
            callable=train_dlrm,
            world_size=WORLD_SIZE,
            local_size=WORLD_SIZE,
            backend="nccl",
        )
