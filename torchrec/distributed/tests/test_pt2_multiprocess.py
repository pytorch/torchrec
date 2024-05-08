#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import unittest
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import fbgemm_gpu.sparse_ops  # noqa: F401, E402

import torch
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import QCommsConfig

from torchrec.distributed.model_parallel import DistributedModelParallel

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import ShardingPlan
from torchrec.distributed.sharding_plan import EmbeddingBagCollectionSharder

from torchrec.distributed.test_utils.infer_utils import (
    dynamo_skipfiles_allow,
    TestModelInfo,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
    EmbeddingCollection,
)
from torchrec.modules.feature_processor_ import FeatureProcessorsCollection
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.pt2.utils import kjt_for_pt2_tracing
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu")

    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cuda_training"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/codegen:embedding_ops_cpu_training"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:split_table_batched_embeddings"
    )
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_cpu")
except OSError:
    pass


class NoOpFPC(FeatureProcessorsCollection):
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        return features


class _ModelType(Enum):
    EBC = 1
    EC = 2
    FPEBC = 3


class _InputType(Enum):
    SINGLE_BATCH = 1
    VARIABLE_BATCH = 2


class EBCSharderFixedShardingType(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomms_config: Optional[QCommsConfig] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}
        if "learning_rate" not in fused_params:
            fused_params["learning_rate"] = 0.1

        self._sharding_type = sharding_type
        super().__init__(fused_params=fused_params)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


class ECSharderFixedShardingType(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomms_config: Optional[QCommsConfig] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}
        if "learning_rate" not in fused_params:
            fused_params["learning_rate"] = 0.1

        self._sharding_type = sharding_type
        super().__init__(fused_params=fused_params)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


def _gen_model(test_model_type: _ModelType, mi: TestModelInfo) -> torch.nn.Module:
    if test_model_type == _ModelType.EBC:

        class M_ebc(torch.nn.Module):
            def __init__(self, ebc: EmbeddingBagCollection) -> None:
                super().__init__()
                self._ebc = ebc

            def forward(self, x: KeyedJaggedTensor) -> torch.Tensor:
                kt: KeyedTensor = self._ebc(x)
                v = kt.values()
                return torch.sigmoid(torch.mean(v, dim=1))

        return M_ebc(
            EmbeddingBagCollection(
                # pyre-ignore
                tables=mi.tables,
                device=mi.sparse_device,
            )
        )
    if test_model_type == _ModelType.FPEBC:

        class M_fpebc(torch.nn.Module):
            def __init__(self, fpebc: FeatureProcessedEmbeddingBagCollection) -> None:
                super().__init__()
                self._fpebc = fpebc

            def forward(self, x: KeyedJaggedTensor) -> torch.Tensor:
                kt: KeyedTensor = self._fpebc(x)
                return kt.values()

        return M_fpebc(
            FeatureProcessedEmbeddingBagCollection(
                embedding_bag_collection=EmbeddingBagCollection(
                    # pyre-ignore
                    tables=mi.tables,
                    device=mi.sparse_device,
                    is_weighted=True,
                ),
                feature_processors=NoOpFPC(),
            )
        )
    elif test_model_type == _ModelType.EC:

        class M_ec(torch.nn.Module):
            def __init__(self, ec: EmbeddingCollection) -> None:
                super().__init__()
                self._ec = ec

            def forward(self, x: KeyedJaggedTensor) -> List[JaggedTensor]:
                d: Dict[str, JaggedTensor] = self._ec(x)
                return list(d.values())

        return M_ec(
            EmbeddingCollection(
                # pyre-ignore
                tables=mi.tables,
                device=mi.sparse_device,
            )
        )
    else:
        raise RuntimeError(f"Unsupported test_model_type:{test_model_type}")


def _test_compile_rank_fn(
    test_model_type: _ModelType,
    rank: int,
    world_size: int,
    backend: str,
    sharding_type: str,
    kernel_type: str,
    input_type: _InputType,
    convert_to_vb: bool,
    torch_compile_backend: Optional[str] = None,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        num_embeddings = 256
        # emb_dim must be % 4 == 0 for fbgemm
        emb_dim = 12
        batch_size = 10
        num_features: int = 5

        num_float_features: int = 8
        num_weighted_features: int = 1

        device = torch.device("cuda")
        pg = ctx.pg
        assert pg is not None

        topology: Topology = Topology(world_size=world_size, compute_device="cuda")
        mi = TestModelInfo(
            dense_device=device,
            sparse_device=device,
            num_features=num_features,
            num_float_features=num_float_features,
            num_weighted_features=num_weighted_features,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )
        config_type = (
            EmbeddingBagConfig
            if test_model_type == _ModelType.EBC or test_model_type == _ModelType.FPEBC
            else EmbeddingConfig
        )

        # pyre-ignore
        mi.tables = [
            config_type(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(mi.num_features)
        ]

        # pyre-ignore
        mi.weighted_tables = [
            config_type(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(mi.num_weighted_features)
        ]

        mi.model = _gen_model(test_model_type, mi)
        mi.model.training = True

        model = mi.model

        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size, device.type, local_world_size=local_size),
            constraints=None,
        )

        sharders = [
            EBCSharderFixedShardingType(sharding_type),
            ECSharderFixedShardingType(sharding_type),
        ]

        plan: ShardingPlan = planner.collective_plan(
            model,
            # pyre-ignore
            sharders,
            pg,
        )

        dmp = DistributedModelParallel(
            model,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            # pyre-ignore
            sharders=sharders,
            device=device,
        )

        if input_type == _InputType.VARIABLE_BATCH:
            (
                global_model_input,
                local_model_inputs,
            ) = ModelInput.generate_variable_batch_input(
                average_batch_size=batch_size,
                world_size=world_size,
                num_float_features=num_float_features,
                # pyre-ignore
                tables=mi.tables,
            )
        else:
            (
                global_model_input,
                local_model_inputs,
            ) = ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=num_float_features,
                tables=mi.tables,
                weighted_tables=mi.weighted_tables,
                variable_batch_size=False,
            )

        local_model_input = local_model_inputs[0].to(device)

        kjt = local_model_input.idlist_features
        kjt_ft = kjt_for_pt2_tracing(kjt, convert_to_vb=convert_to_vb)

        dmp.train(True)
        eager_out = dmp(kjt_ft)

        if torch_compile_backend is None:
            return

        ##### COMPILE #####
        with dynamo_skipfiles_allow("torchrec"):
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            opt_fn = torch.compile(
                dmp, backend=torch_compile_backend, fullgraph=True, dynamic=True
            )
            compile_out = opt_fn(kjt_for_pt2_tracing(kjt, convert_to_vb=convert_to_vb))
            torch.testing.assert_close(eager_out, compile_out)
        ##### COMPILE END #####

        ##### NUMERIC CHECK #####
        with dynamo_skipfiles_allow("torchrec"):
            n = len(local_model_inputs)
            for i in range(n - 1):
                local_model_input = local_model_inputs[1 + i].to(device)
                kjt = local_model_input.idlist_features
                kjt_ft = kjt_for_pt2_tracing(kjt, convert_to_vb=convert_to_vb)
                eager_out_i = dmp(kjt_ft)
                compile_out_i = opt_fn(kjt_ft)
                torch.testing.assert_close(eager_out_i, compile_out_i)
        ##### NUMERIC CHECK END #####


class TestPt2Train(MultiProcessTestBase):
    def disable_cuda_tf32(self) -> bool:
        return True

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore
    @given(
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
            ],
        ),
        model_type_sharding_type_input_type_tovb_backend=st.sampled_from(
            [
                (
                    _ModelType.EBC,
                    ShardingType.TABLE_WISE.value,
                    _InputType.SINGLE_BATCH,
                    True,
                    "eager",
                ),
                (
                    _ModelType.EBC,
                    ShardingType.COLUMN_WISE.value,
                    _InputType.SINGLE_BATCH,
                    True,
                    "eager",
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_compile_multiprocess(
        self,
        kernel_type: str,
        model_type_sharding_type_input_type_tovb_backend: Tuple[
            _ModelType,
            str,
            _InputType,
            bool,
            str,
        ],
    ) -> None:
        model_type, sharding_type, input_type, tovb, compile_backend = (
            model_type_sharding_type_input_type_tovb_backend
        )
        self._run_multi_process_test(
            callable=_test_compile_rank_fn,
            test_model_type=model_type,
            world_size=2,
            backend="nccl",
            sharding_type=sharding_type,
            kernel_type=kernel_type,
            input_type=input_type,
            convert_to_vb=tovb,
            torch_compile_backend=compile_backend,
        )
