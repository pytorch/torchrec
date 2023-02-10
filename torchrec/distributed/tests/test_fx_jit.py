#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import unittest
from enum import Enum

from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
from torch import quantization as quant
from torchrec import EmbeddingCollection, EmbeddingConfig
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    QuantEmbeddingBagCollectionSharder,
)
from torchrec.distributed.shard import shard_modules
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.types import Awaitable, ShardingEnv
from torchrec.distributed.utils import CopyableMixin
from torchrec.fx.tracer import Tracer as TorchrecFxTracer
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class FxJitTestType(Enum):
    CREATE_ONLY = 0
    FX = 1
    FX_JIT = 2


# Wrapper for module that accepts ModelInput to avoid jit scripting of ModelInput (dataclass) and be fully torch types bound.
class TorchTypesModelInputWrapper(CopyableMixin):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(
        self,
        float_features: torch.Tensor,
        idlist_features_keys: List[str],
        idlist_features_values: torch.Tensor,
        idscore_features_keys: List[str],
        idscore_features_values: torch.Tensor,
        idscore_features_weights: torch.Tensor,
        label: torch.Tensor,
        idlist_features_lengths: Optional[torch.Tensor] = None,
        idlist_features_offsets: Optional[torch.Tensor] = None,
        idscore_features_lengths: Optional[torch.Tensor] = None,
        idscore_features_offsets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        idlist_kjt = KeyedJaggedTensor(
            keys=idlist_features_keys,
            values=idlist_features_values,
            lengths=idlist_features_lengths,
            offsets=idlist_features_offsets,
        )
        idscore_kjt = KeyedJaggedTensor(
            keys=idscore_features_keys,
            values=idscore_features_values,
            weights=idscore_features_weights,
            lengths=idscore_features_lengths,
            offsets=idscore_features_offsets,
        )
        mi = ModelInput(
            float_features=float_features,
            idlist_features=idlist_kjt,
            idscore_features=idscore_kjt,
            label=label,
        )
        return self._module(mi)


class KJTInputWrapper(torch.nn.Module):
    def __init__(
        self,
        module_kjt_input: torch.nn.Module,
    ) -> None:
        super().__init__()
        self._module_kjt_input = module_kjt_input
        self.add_module("_module_kjt_input", self._module_kjt_input)

    # pyre-ignore
    def forward(
        self,
        keys: List[str],
        values: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ):
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
            offsets=offsets,
        )
        return self._module_kjt_input(kjt)


class TestQuantEBCSharder(QuantEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        shardable_params: Optional[List[str]] = None,
    ) -> None:
        super().__init__(fused_params=fused_params, shardable_params=shardable_params)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


class TestQuantECSharder(QuantEmbeddingCollectionSharder):
    def __init__(self, sharding_type: str, kernel_type: str) -> None:
        super().__init__()
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


def quantize(
    module: torch.nn.Module,
    inplace: bool,
    output_type: torch.dtype = torch.float,
) -> torch.nn.Module:
    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_type),
        weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
    )
    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            EmbeddingBagCollection: qconfig,
        },
        mapping={
            EmbeddingBagCollection: QuantEmbeddingBagCollection,
        },
        inplace=inplace,
    )


# We want to be torch types bound, args for TorchTypesModelInputWrapper
def model_input_to_forward_args(
    mi: ModelInput,
) -> Tuple[
    torch.Tensor,
    List[str],
    torch.Tensor,
    List[str],
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    idlist_kjt = mi.idlist_features
    idscore_kjt = mi.idscore_features
    assert idscore_kjt is not None
    return (
        mi.float_features,
        idlist_kjt._keys,
        idlist_kjt._values,
        idscore_kjt._keys,
        idscore_kjt._values,
        idscore_kjt._weights,
        mi.label,
        idlist_kjt._lengths,
        idlist_kjt._offsets,
        idscore_kjt._lengths,
        idscore_kjt._offsets,
    )


class ModelTraceScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")
        self.num_features = 2
        self.num_float_features = 10
        self.num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(self.num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(self.num_weighted_features)
        ]
        self.model = TorchTypesModelInputWrapper(
            TestSparseNN(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                num_float_features=self.num_float_features,
                dense_device=self.device,
                sparse_device=self.device,
            )
        )

        self.model.training = False
        self.quant_model = quantize(self.model, inplace=True)

        self.sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[table.name for table in self.tables],
                ),
            ),
            cast(ModuleSharder[torch.nn.Module], EmbeddingCollectionSharder()),
        ]

    def _prep_inputs(self, world_size: int) -> List[Tuple[ModelInput]]:
        inputs = []
        for _ in range(5):
            inputs.append(
                (
                    ModelInput.generate(
                        batch_size=16,
                        world_size=world_size,
                        num_float_features=self.num_float_features,
                        tables=self.tables,
                        weighted_tables=self.weighted_tables,
                    )[1][0].to(self.device),
                )
            )
        return inputs

    def shard_modules_QEBC(
        self,
        world_size: int
        # pyre-ignore
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        sharded_model = shard_modules(
            module=self.quant_model,
            sharders=self.sharders,
            device=self.device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
        )

        inputs = self._prep_inputs(world_size)

        return (
            self.quant_model,
            sharded_model,
            [model_input_to_forward_args(*inp) for inp in inputs],
        )

    def DMP_QEBC(
        self,
        world_size: int,
        unwrap_dmp: bool
        # pyre-ignore
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        topology = Topology(world_size=world_size, compute_device="cuda")
        plan = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=10,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=1,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        ).plan(self.quant_model, self.sharders)

        dmp = DistributedModelParallel(
            self.quant_model,
            plan=plan,
            device=self.device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
            init_data_parallel=False,
        )

        dmp = dmp.copy(self.device)

        inputs = self._prep_inputs(world_size)

        m = dmp.module if unwrap_dmp else dmp
        return (
            self.quant_model,
            m,
            [model_input_to_forward_args(*inp) for inp in inputs],
        )

    def DMP_QEC(
        self,
        world_size: int,
        sharding_enabled: bool,
        # pyre-ignore
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        device = torch.device("cuda:0")
        num_features = 4
        num_float_features = 10

        tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(tables=tables, device=device)
            )
        )

        quant_model = quantize(model, inplace=True)
        m = quant_model

        if sharding_enabled:
            sharders = [
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantECSharder(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ]
            topology = Topology(world_size=world_size, compute_device="cuda")
            plan = EmbeddingShardingPlanner(
                topology=topology,
                batch_size=10,
                enumerator=EmbeddingEnumerator(
                    topology=topology,
                    batch_size=1,
                    estimator=[
                        EmbeddingPerfEstimator(topology=topology, is_inference=True),
                        EmbeddingStorageEstimator(topology=topology),
                    ],
                ),
            ).plan(quant_model, sharders)
            m = DistributedModelParallel(
                quant_model,
                plan=plan,
                device=device,
                env=ShardingEnv.from_local(world_size=world_size, rank=0),
                init_data_parallel=False,
            )

        inputs = []
        for _ in range(5):
            inputs.append(
                (
                    ModelInput.generate(
                        batch_size=16,
                        world_size=world_size,
                        num_float_features=num_float_features,
                        tables=tables,
                        weighted_tables=[],
                    )[1][0].to(device),
                )
            )

        def model_input_to_forward_args(
            mi: ModelInput,
        ) -> Tuple[
            List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
        ]:
            kjt = mi.idlist_features
            return (
                kjt._keys,
                kjt._values,
                kjt._lengths,
                kjt._offsets,
            )

        return (
            quant_model,
            m,
            [model_input_to_forward_args(*inp) for inp in inputs],
        )

    def _models_with_inputs(
        self,
        # pyre-ignore
        *args,
        # pyre-ignore
        **kwargs,
        # pyre-ignore
    ) -> List[Tuple[torch.nn.Module, torch.nn.Module, List[Tuple], FxJitTestType]]:
        return [
            (*fn(*args, **kwargs), test_type)
            for fn, test_type in [
                (
                    lambda world_size: self.DMP_QEBC(
                        world_size=world_size,
                        unwrap_dmp=True,  # preferred usage is to provide fx trace with unwrapped dmp
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.DMP_QEBC(
                        world_size=world_size, unwrap_dmp=False
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.DMP_QEC(
                        world_size=world_size, sharding_enabled=True
                    ),
                    FxJitTestType.CREATE_ONLY,  # waiting for torch.Await support
                ),
                (
                    lambda world_size: self.DMP_QEC(
                        world_size=world_size, sharding_enabled=False
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (self.shard_modules_QEBC, FxJitTestType.FX_JIT),
            ]
        ]

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_fxtrace_jitscript(self) -> None:
        for non_sharded_model, model, inputs, test_type in self._models_with_inputs(
            world_size=2
        ):
            if test_type == FxJitTestType.CREATE_ONLY:
                continue

            # We need more than one input to verify correctness of tracing and scripting using input different from what was used for tracing
            assert len(inputs) > 1

            # Run model first time to go through lazy initialized blocks before tracing
            # Targeting only inference for this time
            non_sharded_model.train(False)
            model.train(False)

            non_sharded_model(*inputs[0])
            eager_output = model(*inputs[0])
            tracer = TorchrecFxTracer()
            graph = tracer.trace(model)
            print(f"This is model type: {type(model)}")

            # pyre-ignore
            gm = torch.fx.GraphModule(tracer.root, graph)

            if test_type == FxJitTestType.FX_JIT:
                gm_script = torch.jit.script(gm)
                gm_script_output = gm_script(*inputs[0])

                # pyre-ignore
                # TODO: Add JaggedTensor check to assert_close
                def assert_close(expected, got) -> None:
                    if isinstance(expected, dict):
                        for feature, jt_e in expected.items():
                            jt_got = got[feature]
                            torch.testing.assert_close(jt_e.lengths(), jt_got.lengths())
                            torch.testing.assert_close(jt_e.values(), jt_got.values())
                            torch.testing.assert_close(jt_e.offsets(), jt_got.offsets())
                    else:
                        torch.testing.assert_close(expected, got)

                if isinstance(eager_output, Awaitable):
                    eager_output = eager_output.wait()

                assert_close(eager_output, gm_script_output)

                for inp in inputs[1:]:
                    eager_output = model(*inp)
                    script_output = gm_script(*inp)
                    assert_close(eager_output, script_output)
