#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import unittest
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
from torch import quantization as quant
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
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    QuantEmbeddingBagCollectionSharder,
)
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.types import ShardingEnv
from torchrec.distributed.utils import CopyableMixin
from torchrec.fx.tracer import Tracer as TorchrecFxTracer
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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


class ModelTraceScriptTest(unittest.TestCase):
    # pyre-ignore
    def DMP_QEBC(self, world_size: int) -> Tuple[torch.nn.Module, List[Tuple]]:
        device = torch.device("cuda:0")
        num_features = 2
        num_float_features = 10
        num_weighted_features = 2

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]
        model = TorchTypesModelInputWrapper(
            TestSparseNN(
                tables=tables,
                weighted_tables=weighted_tables,
                num_float_features=num_float_features,
                dense_device=device,
                sparse_device=device,
            )
        )

        model.training = False

        def _quantize(
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

        quant_model = _quantize(model, inplace=True)

        class TestQuantEBCSharder(QuantEmbeddingBagCollectionSharder):
            def __init__(
                self,
                sharding_type: str,
                kernel_type: str,
                fused_params: Optional[Dict[str, Any]] = None,
                shardable_params: Optional[List[str]] = None,
            ) -> None:
                super().__init__(
                    fused_params=fused_params, shardable_params=shardable_params
                )
                self._sharding_type = sharding_type
                self._kernel_type = kernel_type

            def sharding_types(self, compute_device_type: str) -> List[str]:
                return [self._sharding_type]

            def compute_kernels(
                self, sharding_type: str, compute_device_type: str
            ) -> List[str]:
                return [self._kernel_type]

        sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[table.name for table in tables],
                ),
            ),
            cast(ModuleSharder[torch.nn.Module], EmbeddingCollectionSharder()),
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
        dmp = DistributedModelParallel(
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
                        weighted_tables=weighted_tables,
                    )[1][0].to(device),
                )
            )
        dmp = dmp.copy(device)

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

        return (dmp, [model_input_to_forward_args(*inp) for inp in inputs])

    def _models_with_inputs(
        self,
        # pyre-ignore
        *args,
        # pyre-ignore
        **kwargs
    ) -> List[Tuple[torch.nn.Module, List[Tuple]]]:  # pyre-ignore
        return [fn(*args, **kwargs) for fn in [self.DMP_QEBC]]

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_fxtrace_jitscript(self) -> None:
        for model, inputs in self._models_with_inputs(world_size=2):
            # We need more than one input to verify correctness of tracing and scripting using input different from what was used for tracing
            assert len(inputs) > 1

            # Run model first time to go through lazy initialized blocks before tracing
            # Targeting only inference for this time
            model.train(False)

            eager_output = model(*inputs[0])

            tracer = TorchrecFxTracer()
            g = tracer.trace(model)
            # pyre-ignore
            gm = torch.fx.GraphModule(tracer.root, g)
            gm_script = torch.jit.script(gm)
            gm_script_output = gm_script(*inputs[0])
            torch.testing.assert_close(gm_script_output, eager_output)

            for inp in inputs[1:]:
                torch.testing.assert_close(gm_script(*inp), model(*inp))
