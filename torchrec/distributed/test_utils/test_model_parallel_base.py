#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import Callable, cast, Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.test_model import (
    _get_default_rtol_and_atol,
    ModelInput,
    TestSparseNNBase,
)
from torchrec.distributed.test_utils.test_sharding import (
    copy_state_dict,
    gen_model_and_input,
    ModelInputCallable,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.test_utils import seed_and_log


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
