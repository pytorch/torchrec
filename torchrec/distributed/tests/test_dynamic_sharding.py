#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import copy
import random
import unittest
from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Tuple, Type

import hypothesis.strategies as st

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType

from hypothesis import assume, given, settings, Verbosity

from torch import nn, optim

from torchrec import (
    distributed as trec_dist,
    EmbeddingBagCollection,
    KeyedJaggedTensor,
    optim as trec_optim,
)

from torchrec.distributed.benchmark.benchmark_pipeline_utils import (
    BaseModelConfig,
    create_model_config,
    generate_data,
    generate_pipeline,
    generate_planner,
    generate_sharded_model_and_optimizer,
    generate_tables,
)
from torchrec.distributed.benchmark.benchmark_train_pipeline import (
    PipelineConfig,
    RunOptions,
)
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import Topology
from torchrec.distributed.sharding.dynamic_sharding import output_sharding_plan_delta

from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    get_module_to_default_sharders,
    table_wise,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model import TestOverArchLarge
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    copy_state_dict,
    create_test_sharder,
    generate_rank_placements,
    SharderType,
)

from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import data_type_to_dtype, EmbeddingBagConfig

from torchrec.test_utils import skip_if_asan_class
from torchrec.types import DataType


# Utils:
def table_name(i: int) -> str:
    return "table_" + str(i)


def feature_name(i: int) -> str:
    return "feature_" + str(i)


def generate_embedding_bag_config(
    data_type: DataType,
    num_tables: int = 3,
    embedding_dim: int = 16,
    num_embeddings: int = 4,
) -> List[EmbeddingBagConfig]:
    embedding_bag_config = []
    for i in range(num_tables):
        embedding_bag_config.append(
            EmbeddingBagConfig(
                name=table_name(i),
                feature_names=[feature_name(i)],
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                data_type=data_type,
            ),
        )
    return embedding_bag_config


def create_test_initial_state_dict(
    sharded_module_type: nn.Module,
    num_tables: int,
    data_type: DataType,
    embedding_dim: int = 16,
    num_embeddings: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Helpful for debugging:

    initial_state_dict = {
        "embedding_bags.table_0.weight": torch.tensor(
            [
                [1] * 16,
                [2] * 16,
                [3] * 16,
                [4] * 16,
            ],
        ),
        "embedding_bags.table_1.weight": torch.tensor(
            [
                [101] * 16,
                [102] * 16,
                [103] * 16,
                [104] * 16,
            ],
            dtype=data_type_to_dtype(data_type),
        ),
        ...
    }
    """

    initial_state_dict = {}
    for i in range(num_tables):
        # pyre-ignore
        extended_name = sharded_module_type.extend_shard_name(table_name(i))
        initial_state_dict[extended_name] = torch.tensor(
            [[j + (i * 100)] * embedding_dim for j in range(num_embeddings)],
            dtype=data_type_to_dtype(data_type),
        )

    return initial_state_dict


def are_sharded_ebc_modules_identical(
    module1: ShardedEmbeddingBagCollection,
    module2: ShardedEmbeddingBagCollection,
) -> None:
    # Check if both modules have the same parameters
    params1 = list(module1.named_parameters())
    params2 = list(module2.named_parameters())

    assert len(params1) == len(params2)

    for param1, param2 in zip(params1, params2):
        # Check parameter names
        assert param1[0] == param2[0]
        # Check parameter values
        assert torch.allclose(param1[1], param2[1])

    # Check if both modules have the same buffers
    buffers1 = list(module1.named_buffers())
    buffers2 = list(module2.named_buffers())

    assert len(buffers1) == len(buffers2)

    for buffer1, buffer2 in zip(buffers1, buffers2):
        assert buffer1[0] == buffer2[0]  # Check buffer names
        assert torch.allclose(buffer1[1], buffer2[1])  # Check buffer values

    # Hard-coded attributes for EmbeddingBagCollection
    attribute_list = [
        "_module_fqn",
        "_table_names",
        "_pooling_type_to_rs_features",
        "_output_dtensor",
        "_sharding_types",
        "_is_weighted",
        "_embedding_names",
        "_embedding_dims",
        "_feature_splits",
        "_features_order",
        "_uncombined_embedding_names",
        "_uncombined_embedding_dims",
        "_has_mean_pooling_callback",
        "_kjt_key_indices",
        "_has_uninitialized_input_dist",
        "_has_features_permute",
        "_dim_per_key",  # Tensor
        "_inverse_indices_permute_indices",  # Tensor
        "_kjt_inverse_order",  # Tensor
        "_kt_key_ordering",  # Tensor
        # Non-primitive types which can be compared
        "module_sharding_plan",
        "_table_name_to_config",
        # Excluding the non-primitive types that cannot be compared
        # "sharding_type_to_sharding_infos",
        # "_embedding_shardings"
        # "_input_dists",
        # "_lookups",
        # "_output_dists",
        # "_optim",
    ]

    for attr in attribute_list:
        assert hasattr(module1, attr) and hasattr(module2, attr)

        val1 = getattr(module1, attr)
        val2 = getattr(module2, attr)

        assert type(val1) is type(val2)
        if type(val1) is torch.Tensor:
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2


def _test_ebc_resharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    module_sharding_plan: EmbeddingModuleShardingPlan,
    new_module_sharding_plan: EmbeddingModuleShardingPlan,
    local_size: Optional[int] = None,
) -> None:
    """
    Distributed call to test resharding for ebc by creating 2 models with identical config and
    states:
        m1 sharded with new_module_sharding_plan
        m2 sharded with module_sharding_plan, then resharded with new_module_sharding_plan

    Expects m1 and resharded m2 to be the same, and predictions outputted from the same KJT
    inputs to be the same.

    TODO: modify to include other modules once dynamic sharding is built out.
    """
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        # Set seed to be 0 to ensure models have the same initialization across ranks
        torch.manual_seed(0)
        m1 = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        m2 = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )
        if initial_state_dict is not None:
            initial_state_dict = {
                fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
            }

            # Load initial State - making sure models are identical
            m1.load_state_dict(initial_state_dict)

            m2.load_state_dict(initial_state_dict)

        else:
            # Note this is the only correct behavior due to setting random seed to 0 above
            # Otherwise the weights generated in EBC initialization will be different on
            # Each rank, resulting in different behavior after resharding
            copy_state_dict(
                loc=m2.state_dict(),
                glob=m1.state_dict(),
            )

        sharder = get_module_to_default_sharders()[type(m1)]

        # pyre-ignore
        env = ShardingEnv.from_process_group(ctx.pg)

        sharded_m1 = sharder.shard(
            module=m1,
            params=new_module_sharding_plan,
            env=env,
            device=ctx.device,
        )

        sharded_m2 = sharder.shard(
            module=m1,
            params=module_sharding_plan,
            env=env,
            device=ctx.device,
        )

        new_module_sharding_plan_delta = output_sharding_plan_delta(
            module_sharding_plan, new_module_sharding_plan
        )

        # pyre-ignore
        resharded_m2 = sharder.reshard(
            sharded_module=sharded_m2,
            changed_shard_to_params=new_module_sharding_plan_delta,
            env=env,
            device=ctx.device,
        )

        are_sharded_ebc_modules_identical(sharded_m1, resharded_m2)

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        # For current test model and inputs, the prediction should be the exact same
        # rtol = 0
        # atol = 0

        for _ in range(world_size):
            # sharded model
            # each rank gets a subbatch
            sharded_m1_pred_kt_no_dict = sharded_m1(kjt_input_per_rank[ctx.rank])
            resharded_m2_pred_kt_no_dict = resharded_m2(kjt_input_per_rank[ctx.rank])

            sharded_m1_pred_kt = sharded_m1_pred_kt_no_dict.to_dict()
            resharded_m2_pred_kt = resharded_m2_pred_kt_no_dict.to_dict()
            sharded_m1_pred = torch.stack(
                [sharded_m1_pred_kt[feature] for feature in feature_keys]
            )

            resharded_m2_pred = torch.stack(
                [resharded_m2_pred_kt[feature] for feature in feature_keys]
            )
            # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
            # in normal author modelling code this won't be an issue because each rank would individually create
            # their model. output from sharded_pred is correctly on the correct device.

            # Compare predictions of sharded vs unsharded models.
            torch.testing.assert_close(sharded_m1_pred.cpu(), resharded_m2_pred.cpu())

            sharded_m1_pred.sum().backward()
            resharded_m2_pred.sum().backward()


@skip_if_asan_class
class MultiRankEBCDynamicShardingTest(MultiProcessTestBase):
    def _run_ebc_resharding_test(
        self,
        per_param_sharding: Dict[str, ParameterSharding],
        new_per_param_sharding: Dict[str, ParameterSharding],
        num_tables: int,
        world_size: int,
        data_type: DataType,
        embedding_dim: int = 16,
        num_embeddings: int = 4,
        use_debug_state_dict: bool = False,  # Turn on to use dummy values for initial state dict
    ) -> None:
        embedding_bag_config = generate_embedding_bag_config(
            data_type, num_tables, embedding_dim, num_embeddings
        )

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            # pyre-ignore
            per_param_sharding=per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        new_module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            # pyre-ignore
            per_param_sharding=new_per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Row-wise not supported on gloo
        if (
            not torch.cuda.is_available()
            and new_module_sharding_plan["table_0"].sharding_type
            == ShardingType.ROW_WISE.value
        ):
            return

        kjt_input_per_rank = [
            ModelInput.create_standard_kjt(
                batch_size=2,
                tables=embedding_bag_config,
            )
            for _ in range(world_size)
        ]

        initial_state_dict = None
        if use_debug_state_dict:
            # initial_state_dict filled with deterministic dummy values
            initial_state_dict = create_test_initial_state_dict(
                ShardedEmbeddingBagCollection,  # pyre-ignore
                num_tables,
                data_type,
                embedding_dim,
                num_embeddings,
            )

        self._run_multi_process_test(
            callable=_test_ebc_resharding,
            world_size=world_size,
            tables=embedding_bag_config,
            initial_state_dict=initial_state_dict,
            kjt_input_per_rank=kjt_input_per_rank,
            backend="nccl" if torch.cuda.is_available() else "gloo",
            module_sharding_plan=module_sharding_plan,
            new_module_sharding_plan=new_module_sharding_plan,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(  # pyre-ignore
        num_tables=st.sampled_from([2, 3, 4]),
        data_type=st.sampled_from([DataType.FP32, DataType.FP16]),
        world_size=st.sampled_from([2, 4]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_dynamic_sharding_ebc_tw(
        self,
        num_tables: int,
        data_type: DataType,
        world_size: int,
    ) -> None:
        # Tests EBC dynamic sharding implementation for TW

        # Table wise can only have 1 rank allocated per table:
        ranks_per_tables = [1 for _ in range(num_tables)]
        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
        old_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        new_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)

        while new_ranks == old_ranks:
            new_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
        per_param_sharding = {}
        new_per_param_sharding = {}

        # Construct parameter shardings
        for i in range(num_tables):
            per_param_sharding[table_name(i)] = table_wise(rank=old_ranks[i][0])
            new_per_param_sharding[table_name(i)] = table_wise(rank=new_ranks[i][0])

        self._run_ebc_resharding_test(
            per_param_sharding,
            new_per_param_sharding,
            num_tables,
            world_size,
            data_type,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(  # pyre-ignore
        num_tables=st.sampled_from([2, 3, 4]),
        data_type=st.sampled_from([DataType.FP32, DataType.FP16]),
        world_size=st.sampled_from([3, 4]),
        embedding_dim=st.sampled_from([16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_dynamic_sharding_ebc_cw(
        self,
        num_tables: int,
        data_type: DataType,
        world_size: int,
        embedding_dim: int,
    ) -> None:
        # Tests EBC dynamic sharding implementation for CW

        # Force the ranks per table to be consistent
        valid_ranks = [i for i in range(1, world_size) if embedding_dim % i == 0]
        ranks_per_tables = [random.choice(valid_ranks) for _ in range(num_tables)]
        new_ranks_per_tables = [random.choice(valid_ranks) for _ in range(num_tables)]

        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size

        old_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        new_ranks = generate_rank_placements(
            world_size, num_tables, new_ranks_per_tables
        )

        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
        while new_ranks == old_ranks:
            old_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
            new_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
        per_param_sharding = {}
        new_per_param_sharding = {}

        # Construct parameter shardings
        for i in range(num_tables):
            per_param_sharding[table_name(i)] = column_wise(ranks=old_ranks[i])
            new_per_param_sharding[table_name(i)] = column_wise(ranks=new_ranks[i])

        self._run_ebc_resharding_test(
            per_param_sharding,
            new_per_param_sharding,
            num_tables,
            world_size,
            data_type,
            embedding_dim,
        )


@skip_if_asan_class
class MultiRankDMPDynamicShardingTest(ModelParallelTestShared):
    @unittest.skipIf(
        torch.cuda.device_count() <= 8,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(  # Pyre-ignore
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
            ]
        ),
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        random_seed=st.integers(0, 1000),
        world_size=st.sampled_from([2, 4, 8]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        data_type: DataType,
        random_seed: int,
        world_size: int,
    ) -> None:
        """
        Tests resharding from DMP module interface with conditional optimizer selection:
        - For Table-Wise sharding: All optimizers including RowWiseAdagrad
        - For Column-Wise sharding: Only Adagrad and SGD (no RowWiseAdagrad)
        """
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        # Fixed to False as variable batch size with CW is more complex
        variable_batch_size = False

        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )

        # Define optimizer options based on sharding type
        if sharding_type == ShardingType.COLUMN_WISE.value:
            # For Column-Wise sharding, exclude RowWiseAdagrad
            # TODO: fix rowise adagrad logic similar to 2D sharding
            optimizer_options = [
                None,
                {"embedding_bags": (optim.Adagrad, {"lr": 0.04})},
                {"embedding_bags": (torch.optim.SGD, {"lr": 0.01})},
            ]
        else:
            # For Table-Wise sharding, include all optimizers
            optimizer_options = [
                None,
                {"embedding_bags": (optim.Adagrad, {"lr": 0.04})},
                {"embedding_bags": (torch.optim.SGD, {"lr": 0.01})},
                {"embedding_bags": (trec_optim.RowWiseAdagrad, {"lr": 0.01})},
            ]

        # Randomly select one optimizer from the appropriate options
        apply_optimizer_in_backward_config = random.choice(optimizer_options)

        sharding_type_e = ShardingType(sharding_type)
        self._test_dynamic_sharding(
            sharders=[  # Pyre-ignore
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=self.device,
                ),
            ],
            backend=self.backend,
            qcomms_config=qcomms_config,
            # Pyre-ignore
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            data_type=data_type,
            sharding_type=sharding_type_e,
            random_seed=random_seed,
        )


class SingleRankDynamicShardingUtilsTest(unittest.TestCase):
    def test_output_sharding_plan_delta(self) -> None:
        """
        Tests output_sharding_plan_delta function
        """
        num_tables = 2
        # NOTE: even though this is a single rank DS test, setting world_size to 2 here to check
        # output_sharding_plan_delta function to see if it works correctly at pruning a sharding plan.
        # Since we don't actually run resharding in this UT, no need to ensure world_size is exactly as ranks used.
        world_size = 2
        data_type = DataType.FP32
        embedding_dim = 16
        num_embeddings = 4

        # Table wise can only have 1 rank allocated per table:
        ranks_per_tables = [1 for _ in range(num_tables)]
        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
        old_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        new_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)

        while new_ranks == old_ranks:
            new_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
        per_param_sharding = {}
        new_per_param_sharding = {}

        # Construct parameter shardings
        for i in range(num_tables):
            per_param_sharding[table_name(i)] = table_wise(rank=old_ranks[i][0])
            new_per_param_sharding[table_name(i)] = table_wise(rank=new_ranks[i][0])

        embedding_bag_config = generate_embedding_bag_config(
            data_type, num_tables, embedding_dim, num_embeddings
        )

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        new_module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=new_per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        new_module_sharding_plan_delta = output_sharding_plan_delta(
            module_sharding_plan, new_module_sharding_plan
        )

        assert len(new_module_sharding_plan_delta) <= len(new_module_sharding_plan)

        # using t_name instead of table_name to avoid clashing with helper method
        for t_name, new_sharding in new_module_sharding_plan.items():
            if new_sharding.ranks != module_sharding_plan[t_name].ranks:
                assert t_name in new_module_sharding_plan_delta
                assert (
                    new_module_sharding_plan_delta[t_name].ranks == new_sharding.ranks
                )
                assert (
                    new_module_sharding_plan_delta[t_name].sharding_type
                    == new_sharding.sharding_type
                )
                assert (
                    new_module_sharding_plan_delta[t_name].compute_kernel
                    == new_sharding.compute_kernel
                )
                # NOTE there are other attributes to test for equivalence in ParameterSharding type
                # but the ones included here are the most important.


def _test_pipeline_resharding(
    rank: int,
    world_size: int,
    backend: str,
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    weighted_tables: List[EmbeddingBagConfig],
    model_config: BaseModelConfig,
    pipeline_config: PipelineConfig,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    module_sharding_plan: EmbeddingModuleShardingPlan,
    new_module_sharding_plan: EmbeddingModuleShardingPlan,
    local_size: Optional[int] = None,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)
    torch.autograd.set_detect_anomaly(True)
    run_option = RunOptions()
    with MultiProcessContext(
        rank, world_size, backend, local_size, use_deterministic_algorithms=False
    ) as ctx:
        unsharded_model = model_config.generate_model(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=ctx.device,
        )

        # Create a topology for sharding
        topology = Topology(
            local_world_size=get_local_size(world_size),
            world_size=world_size,
            compute_device=ctx.device.type,
        )

        batch_sizes = model_config.batch_sizes

        if batch_sizes is None:
            batch_sizes = [model_config.batch_size] * run_option.num_batches
        else:
            assert (
                len(batch_sizes) == run_option.num_batches
            ), "The length of batch_sizes must match the number of batches."

        # Create a planner for sharding based on the specified type
        planner = generate_planner(
            planner_type=run_option.planner_type,
            topology=topology,
            tables=tables,
            weighted_tables=weighted_tables,
            sharding_type=run_option.sharding_type,
            compute_kernel=run_option.compute_kernel,
            batch_sizes=batch_sizes,
            pooling_factors=run_option.pooling_factors,
            num_poolings=run_option.num_poolings,
        )
        bench_inputs = generate_data(
            tables=tables,
            weighted_tables=weighted_tables,
            model_config=model_config,
            batch_sizes=batch_sizes,
        )

        # Prepare fused_params for sparse optimizer
        fused_params = {
            "optimizer": getattr(EmbOptimType, run_option.sparse_optimizer.upper()),
            "learning_rate": run_option.sparse_lr,
        }

        # Add momentum and weight_decay to fused_params if provided
        if run_option.sparse_momentum is not None:
            fused_params["momentum"] = run_option.sparse_momentum

        if run_option.sparse_weight_decay is not None:
            fused_params["weight_decay"] = run_option.sparse_weight_decay

        sharded_model, optimizer = generate_sharded_model_and_optimizer(
            model=unsharded_model,
            sharding_type=run_option.sharding_type.value,
            kernel_type=run_option.compute_kernel.value,
            # pyre-ignore
            pg=ctx.pg,
            device=ctx.device,
            fused_params=fused_params,
            dense_optimizer=run_option.dense_optimizer,
            dense_lr=run_option.dense_lr,
            dense_momentum=run_option.dense_momentum,
            dense_weight_decay=run_option.dense_weight_decay,
            planner=planner,
        )

        pipeline = generate_pipeline(
            pipeline_type=pipeline_config.pipeline,
            emb_lookup_stream=pipeline_config.emb_lookup_stream,
            model=sharded_model,
            opt=optimizer,
            device=ctx.device,
            apply_jit=pipeline_config.apply_jit,
        )
        pipeline.progress(iter(bench_inputs))

        dataloader = iter(bench_inputs)
        i = 0
        while True:
            try:
                if i == 3:
                    # Extract existing sharding plan
                    existing_sharding_plan = pipeline._model.module.sparse.ebc.module_sharding_plan  # pyre-ignore
                    fqn_to_local_shards = "sparse.ebc"
                    # Modify existing sharding plan - Hard code
                    sharding_param = copy.deepcopy(existing_sharding_plan["table_0"])
                    new_device = 1 if sharding_param.ranks[0] == 0 else 0
                    sharding_param.ranks = [new_device]
                    sharding_param.sharding_spec.shards[0].placement = (
                        torch.distributed._remote_device(
                            f"rank:{new_device}/cuda:{new_device}"
                        )
                    )

                    new_sharding_plan = {}
                    new_sharding_plan["table_0"] = sharding_param
                    # Reshard
                    pipeline.progress_with_reshard(  # pyre-ignore
                        dataloader_iter=dataloader,
                        reshard_params=new_sharding_plan,
                        sharded_module_fqn=fqn_to_local_shards,
                    )
                    i += 1
                else:
                    pipeline.progress(dataloader)
                    i += 1
            except StopIteration:
                break


@dataclass
class RunOptions:
    """
    Configuration options for running sparse neural network benchmarks.

    This class defines the parameters that control how the benchmark is executed,
    including distributed training settings, batch configuration, and profiling options.

    Args:
        world_size (int): Number of processes/GPUs to use for distributed training.
            Default is 2.
        num_batches (int): Number of batches to process during the benchmark.
            Default is 10.
        sharding_type (ShardingType): Strategy for sharding embedding tables across devices.
            Default is ShardingType.TABLE_WISE (entire tables are placed on single devices).
        compute_kernel (EmbeddingComputeKernel): Compute kernel to use for embedding tables.
            Default is EmbeddingComputeKernel.FUSED.
        input_type (str): Type of input format to use for the model.
            Default is "kjt" (KeyedJaggedTensor).
        profile (str): Directory to save profiling results. If empty, profiling is disabled.
            Default is "" (disabled).
        planner_type (str): Type of sharding planner to use. Options are:
            - "embedding": EmbeddingShardingPlanner (default)
            - "hetero": HeteroEmbeddingShardingPlanner
        pooling_factors (Optional[List[float]]): Pooling factors for each feature of the table.
            This is the average number of values each sample has for the feature.
        num_poolings (Optional[List[float]]): Number of poolings for each feature of the table.
        dense_optimizer (str): Optimizer to use for dense parameters.
            Default is "SGD".
        dense_lr (float): Learning rate for dense parameters.
            Default is 0.1.
        sparse_optimizer (str): Optimizer to use for sparse parameters.
            Default is "EXACT_ADAGRAD".
        sparse_lr (float): Learning rate for sparse parameters.
            Default is 0.1.
    """

    world_size: int = 2
    num_batches: int = 10
    sharding_type: ShardingType = ShardingType.TABLE_WISE
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.DENSE
    input_type: str = "kjt"
    profile: str = ""
    planner_type: str = "embedding"
    pooling_factors: Optional[List[float]] = None
    num_poolings: Optional[List[float]] = None
    dense_optimizer: str = "SGD"
    dense_lr: float = 0.1
    dense_momentum: Optional[float] = None
    dense_weight_decay: Optional[float] = None
    sparse_optimizer: str = "EXACT_ADAGRAD"
    sparse_lr: float = 0.1
    sparse_momentum: Optional[float] = None
    sparse_weight_decay: Optional[float] = None
    export_stacks: bool = False


@dataclass
class EmbeddingTablesConfig:
    """
    Configuration for embedding tables.

    This class defines the parameters for generating embedding tables with both weighted
    and unweighted features.

    Args:
        num_unweighted_features (int): Number of unweighted features to generate.
            Default is 100.
        num_weighted_features (int): Number of weighted features to generate.
            Default is 100.
        embedding_feature_dim (int): Dimension of the embedding vectors.
            Default is 128.
    """

    num_unweighted_features: int = 100
    num_weighted_features: int = 100
    embedding_feature_dim: int = 128


@dataclass
class PipelineConfig:
    """
    Configuration for training pipelines.

    This class defines the parameters for configuring the training pipeline.

    Args:
        pipeline (str): The type of training pipeline to use. Options include:
            - "base": Basic training pipeline
            - "sparse": Pipeline optimized for sparse operations
            - "fused": Pipeline with fused sparse distribution
            - "semi": Semi-synchronous training pipeline
            - "prefetch": Pipeline with prefetching for sparse distribution
            Default is "base".
        emb_lookup_stream (str): The stream to use for embedding lookups.
            Only used by certain pipeline types (e.g., "fused").
            Default is "data_dist".
        apply_jit (bool): Whether to apply JIT (Just-In-Time) compilation to the model.
            Default is False.
    """

    pipeline: str = "sparse"  # "base",
    emb_lookup_stream: str = "data_dist"
    apply_jit: bool = False


class MultiRankPipelineDynamicShardingTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 4,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(  # pyre-ignore
        # Model selection config parameters
        model_name=st.sampled_from(
            [
                "test_sparse_nn",
                # "test_tower_sparse_nn",
                # "test_tower_collection_sparse_nn",
                # "deepfm",
                # "dlrm",
            ]
        ),
        batch_size=st.integers(10, 9000),
        batch_sizes=st.lists(st.integers(10, 100), min_size=10, max_size=10),
        num_float_features=st.integers(5, 20),
        feature_pooling_avg=st.integers(5, 20),
        use_offsets=st.booleans(),
        dev_str=st.just(""),
        long_kjt_indices=st.booleans(),
        long_kjt_offsets=st.booleans(),
        long_kjt_lengths=st.booleans(),
        pin_memory=st.booleans(),
        zch=st.booleans(),
        hidden_layer_size=st.integers(10, 50),
        deep_fm_dimension=st.integers(3, 10),
        dense_arch_layer_sizes=st.lists(st.integers(10, 200), min_size=2, max_size=2),
        over_arch_layer_sizes=st.lists(st.integers(1, 10), min_size=2, max_size=2),
        # PipelineConfig parameters
        pipeline=st.sampled_from(
            ["sparse"]
        ),  # "base", "sparse", "fused", "semi", "prefetch"s
        emb_lookup_stream=st.sampled_from(["data_dist", "compute"]),
        apply_jit=st.booleans(),
        # EmbeddingTablesConfig parameters
        num_unweighted_features=st.integers(50, 200),
        num_weighted_features=st.integers(50, 200),
        embedding_feature_dim=st.integers(64, 256),
        # RunOptions parameters
        num_batches=st.integers(5, 20),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE,
                ShardingType.COLUMN_WISE,
                ShardingType.ROW_WISE,
                ShardingType.DATA_PARALLEL,
            ]
        ),
        compute_kernel=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE,
                # TODO: Move to CPU
                # EmbeddingComputeKernel.FUSED,
                # EmbeddingComputeKernel.FUSED_UVM,
                # EmbeddingComputeKernel.FUSED_UVM_CACHING,
            ]
        ),
        input_type=st.just("kjt"),
        profile=st.just(""),
        planner_type=st.sampled_from(["embedding", "hetero"]),
        dense_optimizer=st.sampled_from(["SGD", "Adam", "Adagrad"]),
        dense_lr=st.floats(min_value=0.01, max_value=1.0),
        sparse_optimizer=st.sampled_from(
            ["EXACT_ADAGRAD", "EXACT_ROWWISE_ADAGRAD", "EXACT_SGD"]
        ),
        sparse_lr=st.floats(min_value=0.01, max_value=1.0),
        export_stacks=st.booleans(),
        world_size=st.sampled_from([2, 4]),  #  8
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_pipeline_resharding(
        self,
        # ModelSelectionConfig parameters
        model_name: str,
        batch_size: int,
        batch_sizes: List[int],
        num_float_features: int,
        feature_pooling_avg: int,
        use_offsets: bool,
        dev_str: str,
        long_kjt_indices: bool,
        long_kjt_offsets: bool,
        long_kjt_lengths: bool,
        pin_memory: bool,
        zch: bool,
        hidden_layer_size: int,
        deep_fm_dimension: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        # PipelineConfig parameters
        pipeline: str,
        emb_lookup_stream: str,
        apply_jit: bool,
        # EmbeddingTablesConfig parameters
        num_unweighted_features: int,
        num_weighted_features: int,
        embedding_feature_dim: int,
        # RunOptions parameters
        num_batches: int,
        sharding_type: ShardingType,
        compute_kernel: EmbeddingComputeKernel,
        input_type: str,
        profile: str,
        planner_type: str,
        dense_optimizer: str,
        dense_lr: float,
        sparse_optimizer: str,
        sparse_lr: float,
        export_stacks: bool,
        world_size: int,
    ) -> None:
        # Create run options
        run_options = RunOptions(
            world_size=world_size,
            num_batches=num_batches,
            sharding_type=sharding_type,
            compute_kernel=compute_kernel,
            input_type=input_type,
            profile=profile,
            planner_type=planner_type,
            dense_optimizer=dense_optimizer,
            dense_lr=dense_lr,
            sparse_optimizer=sparse_optimizer,
            sparse_lr=sparse_lr,
            export_stacks=export_stacks,
        )

        # Generate tables using embedding tables config parameters
        tables, weighted_tables = generate_tables(
            num_unweighted_features=num_unweighted_features,
            num_weighted_features=num_weighted_features,
            embedding_feature_dim=embedding_feature_dim,
        )

        # Create model config
        model_config = create_model_config(
            model_name=model_name,
            batch_size=batch_size,
            batch_sizes=batch_sizes,
            num_float_features=num_float_features,
            feature_pooling_avg=feature_pooling_avg,
            use_offsets=use_offsets,
            dev_str=dev_str,
            long_kjt_indices=long_kjt_indices,
            long_kjt_offsets=long_kjt_offsets,
            long_kjt_lengths=long_kjt_lengths,
            pin_memory=pin_memory,
            embedding_groups=None,
            feature_processor_modules=None,
            max_feature_lengths=None,
            over_arch_clazz=TestOverArchLarge,
            postproc_module=None,
            zch=zch,
            hidden_layer_size=hidden_layer_size,
            deep_fm_dimension=deep_fm_dimension,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
        )

        # Create pipeline config
        pipeline_config = PipelineConfig(
            pipeline=pipeline,
            emb_lookup_stream=emb_lookup_stream,
            apply_jit=apply_jit,
        )
        self._run_multi_process_test(
            callable=_test_pipeline_resharding,
            world_size=world_size,
            tables=tables,
            weighted_tables=weighted_tables,
            pipeline_config=pipeline_config,
            model_config=model_config,
            initial_state_dict=None,
            kjt_input_per_rank=None,
            backend="nccl",
            module_sharding_plan=None,
            new_module_sharding_plan=None,
        )
