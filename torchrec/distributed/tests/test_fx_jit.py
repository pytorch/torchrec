#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import unittest
from dataclasses import dataclass, field
from enum import Enum

from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
from torch import quantization as quant
from torchrec import EmbeddingCollection, EmbeddingConfig
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    KJTList,
    ListOfKJTList,
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
from torchrec.fx.utils import fake_range
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


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
            EmbeddingCollection: qconfig,
        },
        mapping={
            EmbeddingBagCollection: QuantEmbeddingBagCollection,
            EmbeddingCollection: QuantEmbeddingCollection,
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


def model_input_to_forward_args_kjt(
    mi: ModelInput,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    kjt = mi.idlist_features
    return (
        kjt._keys,
        kjt._values,
        kjt._lengths,
        kjt._offsets,
    )


@dataclass
class TestModelInfo:
    device: torch.device
    num_features: int
    num_float_features: int
    num_weighted_features: int
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]] = field(
        default_factory=list
    )
    weighted_tables: List[EmbeddingBagConfig] = field(default_factory=list)
    model: torch.nn.Module = torch.nn.Module()
    quant_model: torch.nn.Module = torch.nn.Module()
    sharders: List[ModuleSharder] = field(default_factory=list)


class ModelTraceScriptTest(unittest.TestCase):
    def _set_up_qebc(self) -> TestModelInfo:
        model_info = TestModelInfo(
            device=torch.device("cuda:0"),
            num_features=2,
            num_float_features=10,
            num_weighted_features=2,
        )

        model_info.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(model_info.num_features)
        ]
        model_info.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(model_info.num_weighted_features)
        ]
        model_info.model = TorchTypesModelInputWrapper(
            TestSparseNN(
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
                num_float_features=model_info.num_float_features,
                dense_device=model_info.device,
                sparse_device=model_info.device,
            )
        )

        model_info.model.training = False
        model_info.quant_model = quantize(model_info.model, inplace=True)

        model_info.sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[table.name for table in model_info.tables],
                ),
            ),
            cast(ModuleSharder[torch.nn.Module], EmbeddingCollectionSharder()),
        ]

        return model_info

    def _set_up_qec(self) -> TestModelInfo:
        model_info = TestModelInfo(
            device=torch.device("cuda:0"),
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
        )
        model_info.tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(model_info.num_features)
        ]

        model_info.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=model_info.tables,
                    device=model_info.device,
                )
            )
        )

        model_info.model.training = False
        model_info.quant_model = quantize(model_info.model, inplace=True)

        model_info.sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantECSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                ),
            )
        ]

        return model_info

    def _prep_inputs(
        self, model_info: TestModelInfo, world_size: int
    ) -> List[Tuple[ModelInput]]:
        inputs = []
        for _ in range(5):
            inputs.append(
                (
                    ModelInput.generate(
                        batch_size=16,
                        world_size=world_size,
                        num_float_features=model_info.num_float_features,
                        tables=model_info.tables,
                        weighted_tables=model_info.weighted_tables,
                    )[1][0].to(model_info.device),
                )
            )
        return inputs

    def shard_modules_QEBC(
        self,
        world_size: int
        # pyre-ignore
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qebc()
        sharded_model = shard_modules(
            module=model_info.quant_model,
            sharders=model_info.sharders,
            device=model_info.device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
        )

        inputs = self._prep_inputs(model_info, world_size)

        return (
            model_info.quant_model,
            sharded_model,
            [model_input_to_forward_args(*inp) for inp in inputs],
        )

    def shard_modules_QEC(
        self,
        world_size: int
        # pyre-ignore
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qec()
        sharded_model = shard_modules(
            module=model_info.quant_model,
            sharders=model_info.sharders,
            device=model_info.device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
        )

        inputs = self._prep_inputs(model_info, world_size)

        return (
            model_info.quant_model,
            sharded_model,
            [model_input_to_forward_args_kjt(*inp) for inp in inputs],
        )

    def DMP_QEBC(
        self,
        world_size: int,
        unwrap_dmp: bool
        # pyre-ignore
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qebc()
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
        ).plan(model_info.quant_model, model_info.sharders)

        dmp = DistributedModelParallel(
            model_info.quant_model,
            plan=plan,
            device=model_info.device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
            init_data_parallel=False,
        )

        dmp = dmp.copy(model_info.device)

        inputs = self._prep_inputs(model_info, world_size)

        m = dmp.module if unwrap_dmp else dmp
        return (
            model_info.quant_model,
            m,
            [model_input_to_forward_args(*inp) for inp in inputs],
        )

    def DMP_QEC(
        self,
        world_size: int,
        sharding_enabled: bool,
        # pyre-ignore
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qec()

        if sharding_enabled:
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
            ).plan(model_info.quant_model, model_info.sharders)
            m = DistributedModelParallel(
                model_info.quant_model,
                plan=plan,
                device=model_info.device,
                env=ShardingEnv.from_local(world_size=world_size, rank=0),
                init_data_parallel=False,
            )
            model_info.model = m.module

        inputs = self._prep_inputs(model_info, world_size)

        return (
            model_info.quant_model,
            model_info.model,
            [model_input_to_forward_args_kjt(*inp) for inp in inputs],
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
                (self.shard_modules_QEC, FxJitTestType.FX_JIT),
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
            print(
                f"test_fxtrace_jitscript: non_sharded_model:{non_sharded_model}\n model:{model}"
            )

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

    def test_jitscript(self) -> None:
        # Check main types to be torch jit scriptable
        for clz in [
            JaggedTensor,
            KeyedJaggedTensor,
            KeyedTensor,
            KJTList,
            ListOfKJTList,
        ]:
            # Using torch.jit._script._recursive_compile_class instead of torch.jit.script
            # As classes later is more restrictive, checking no inheritance
            # (e.g. Multistreamable which we so far do not need in jit script) etc.
            # We need those classes not as it is, but as composable blocks in model.
            # _recursive_compile_class for that is enough
            torch.jit._script._recursive_compile_class(clz, fake_range())
