#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torchrec
from torch import quantization as quant
from torch.distributed._shard.sharding_spec import ShardingSpec
from torch.utils import _pytree as pytree
from torchrec import (
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
    KeyedTensor,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.quant_embedding import (
    QuantEmbeddingCollectionSharder,
    ShardedQuantEmbeddingCollection,
)
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    QuantEmbeddingBagCollectionSharder,
    ShardedQuantEmbeddingBagCollection,
)
from torchrec.distributed.quant_state import WeightSpec
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.types import (
    ModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.distributed.utils import CopyableMixin
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    MODULE_ATTR_REGISTER_TBES_BOOL,
    quant_prep_enable_quant_state_dict_split_scale_bias_for_types,
    quant_prep_enable_register_tbes,
)


@dataclass
class TestModelInfo:
    sparse_device: torch.device
    dense_device: torch.device
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
    topology: Optional[Topology] = None
    planner: Optional[EmbeddingShardingPlanner] = None


class KJTInputExportWrapper(torch.nn.Module):
    def __init__(
        self,
        module_kjt_input: torch.nn.Module,
        kjt_keys: List[str],
    ) -> None:
        super().__init__()
        self._module_kjt_input = module_kjt_input
        self._kjt_keys = kjt_keys

    # pyre-ignore
    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        # pyre-ignore
        *args,
        # pyre-ignore
        **kwargs,
    ):
        kjt = KeyedJaggedTensor(
            keys=self._kjt_keys,
            values=values,
            lengths=lengths,
        )
        output = self._module_kjt_input(kjt, *args, **kwargs)
        # TODO(ivankobzarev): Support of None leaves in dynamo/export (e.g. KJT offsets)
        return [leaf for leaf in pytree.tree_leaves(output) if leaf is not None]


def prep_inputs(
    model_info: TestModelInfo,
    world_size: int,
    batch_size: int = 1,
    count: int = 5,
    long_indices: bool = True,
) -> List[ModelInput]:
    inputs = []
    for _ in range(count):
        inputs.append(
            ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=model_info.num_float_features,
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
                long_indices=long_indices,
            )[1][0],
        )
    return inputs


def prep_inputs_multiprocess(
    model_info: TestModelInfo, world_size: int, batch_size: int = 1, count: int = 5
) -> List[Tuple[ModelInput, List[ModelInput]]]:
    inputs = []
    for _ in range(count):
        inputs.append(
            ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=model_info.num_float_features,
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
            )
        )
    return inputs


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


def create_cw_min_partition_constraints(
    table_min_partition_pairs: List[Tuple[str, int]]
) -> Dict[str, ParameterConstraints]:
    return {
        name: ParameterConstraints(
            sharding_types=[ShardingType.COLUMN_WISE.value],
            min_partition=min_partition,
        )
        for name, min_partition in table_min_partition_pairs
    }


def quantize(
    module: torch.nn.Module,
    inplace: bool,
    output_type: torch.dtype = torch.float,
    register_tbes: bool = False,
    quant_state_dict_split_scale_bias: bool = False,
    weight_dtype: torch.dtype = torch.qint8,
) -> torch.nn.Module:
    module_types: List[Type[torch.nn.Module]] = [
        torchrec.modules.embedding_modules.EmbeddingBagCollection,
        torchrec.modules.embedding_modules.EmbeddingCollection,
    ]
    if register_tbes:
        quant_prep_enable_register_tbes(module, module_types)
    if quant_state_dict_split_scale_bias:
        quant_prep_enable_quant_state_dict_split_scale_bias_for_types(
            module, module_types
        )

    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_type),
        weight=quant.PlaceholderObserver.with_args(dtype=weight_dtype),
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

    def shard(
        self,
        module: QuantEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
            module, MODULE_ATTR_REGISTER_TBES_BOOL, False
        )
        fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
            module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
        )
        return ShardedQuantEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=fused_params,
            device=device,
        )


class TestQuantECSharder(QuantEmbeddingCollectionSharder):
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

    def shard(
        self,
        module: QuantEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedQuantEmbeddingCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
            module, MODULE_ATTR_REGISTER_TBES_BOOL, False
        )
        fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
            module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
        )
        return ShardedQuantEmbeddingCollection(
            module, params, env, fused_params, device
        )


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


def create_test_model(
    num_embeddings: int,
    emb_dim: int,
    world_size: int,
    batch_size: int,
    dense_device: torch.device,
    sparse_device: torch.device,
    quant_state_dict_split_scale_bias: bool = False,
    num_features: int = 1,
    num_float_features: int = 8,
    num_weight_features: int = 1,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    weight_dtype: torch.dtype = torch.qint8,
) -> TestModelInfo:
    topology: Topology = Topology(world_size=world_size, compute_device="cuda")
    mi = TestModelInfo(
        dense_device=dense_device,
        sparse_device=sparse_device,
        num_features=num_features,
        num_float_features=num_float_features,
        num_weighted_features=num_weight_features,
        topology=topology,
    )

    mi.planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            estimator=[
                EmbeddingPerfEstimator(topology=topology, is_inference=True),
                EmbeddingStorageEstimator(topology=topology),
            ],
            constraints=constraints,
        ),
    )

    mi.tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(mi.num_features)
    ]

    mi.weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(mi.num_weighted_features)
    ]

    mi.model = TorchTypesModelInputWrapper(
        TestSparseNN(
            tables=mi.tables,
            weighted_tables=mi.weighted_tables,
            num_float_features=mi.num_float_features,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
    )
    mi.model.training = False
    mi.quant_model = quantize(
        module=mi.model,
        inplace=False,
        quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        weight_dtype=weight_dtype,
    )
    return mi


def shard_qebc(
    mi: TestModelInfo,
    sharding_type: ShardingType,
    device: torch.device,
    expected_shards: List[List[Tuple[Tuple[int, int, int, int], str]]],
    plan: Optional[ShardingPlan] = None,
) -> torch.nn.Module:
    sharder = TestQuantEBCSharder(
        sharding_type=sharding_type.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in mi.tables],
    )

    if not plan:
        # pyre-ignore
        plan = mi.planner.plan(
            mi.quant_model,
            [sharder],
        )

    msp = plan.plan["_module.sparse.ebc"]
    for i in range(mi.num_features):
        ps: ParameterSharding = msp[f"table_{i}"]
        assert ps.sharding_type == sharding_type.value
        assert ps.sharding_spec is not None
        sharding_spec: ShardingSpec = ps.sharding_spec
        # pyre-ignore
        assert len(sharding_spec.shards) == len(expected_shards[i])
        for shard, ((offset_r, offset_c, size_r, size_c), placement) in zip(
            sharding_spec.shards, expected_shards[i]
        ):
            assert shard.shard_offsets == [offset_r, offset_c]
            assert shard.shard_sizes == [size_r, size_c]
            assert str(shard.placement) == placement

    # We want to leave quant_model unchanged to compare the results with it
    quant_model_copy = copy.deepcopy(mi.quant_model)
    sharded_model = _shard_modules(
        module=quant_model_copy,
        # pyre-ignore
        sharders=[sharder],
        device=device,
        plan=plan,
        # pyre-ignore
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )
    return sharded_model


def shard_qec(
    mi: TestModelInfo,
    sharding_type: ShardingType,
    device: torch.device,
    expected_shards: List[List[Tuple[Tuple[int, int, int, int], str]]],
) -> torch.nn.Module:
    sharder = TestQuantECSharder(
        sharding_type=sharding_type.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
    )
    # pyre-ignore
    plan = mi.planner.plan(
        mi.quant_model,
        [sharder],
    )
    msp: ModuleShardingPlan = plan.plan["_module_kjt_input.0"]  # TODO: hardcoded
    for i in range(mi.num_features):
        # pyre-ignore
        ps: ParameterSharding = msp[f"table_{i}"]
        assert ps.sharding_type == sharding_type.value
        assert ps.sharding_spec is not None
        sharding_spec: ShardingSpec = ps.sharding_spec
        # pyre-ignore
        assert len(sharding_spec.shards) == len(expected_shards[i])
        for shard, ((offset_r, offset_c, size_r, size_c), placement) in zip(
            sharding_spec.shards, expected_shards[i]
        ):
            assert shard.shard_offsets == [offset_r, offset_c]
            assert shard.shard_sizes == [size_r, size_c]
            assert str(shard.placement) == placement

    # We want to leave quant_model unchanged to compare the results with it
    quant_model_copy = copy.deepcopy(mi.quant_model)
    sharded_model = _shard_modules(
        module=quant_model_copy,
        # pyre-ignore
        sharders=[sharder],
        device=device,
        plan=plan,
        # pyre-ignore
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )
    return sharded_model


# pyre-ignore
def assert_close(expected, actual) -> None:
    if isinstance(expected, KeyedTensor):
        assert isinstance(actual, KeyedTensor)
        assert len(expected.keys()) == len(actual.keys())
        torch.testing.assert_close(expected.values(), actual.values())
        torch.testing.assert_close(expected.length_per_key(), actual.length_per_key())
    if isinstance(expected, dict):
        assert list(expected.keys()) == list(actual.keys())
        for feature, jt_e in expected.items():
            jt_got = actual[feature]
            assert_close(jt_e.lengths(), jt_got.lengths())
            assert_close(jt_e.values(), jt_got.values())
            assert_close(jt_e.offsets(), jt_got.offsets())
    else:
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            if actual.device != expected.device:
                actual = actual.to(expected.device)

        torch.testing.assert_close(expected, actual)


def assert_weight_spec(
    weights_spec: Dict[str, WeightSpec],
    all_expected_shards: List[List[Tuple[Tuple[int, int, int, int], str]]],
    ebc_fqn: str,
    weights_prefix: str,
    all_table_names: List[str],
    sharding_type: str,
) -> None:
    tbe_table_idxs = [0, 0]
    for table_name, expected_shards in zip(all_table_names, all_expected_shards):
        unsharded_weight_fqn = f"{ebc_fqn}.{weights_prefix}.{table_name}.weight"
        for (offset_r, offset_c, size_r, size_c), placement in expected_shards:
            tbe_idx: int = 0
            if "rank:1/cuda:1" == placement:
                tbe_idx = 1
            sharded_weight_fqn: str = f"{ebc_fqn}.tbes.{tbe_idx}.{tbe_table_idxs[tbe_idx]}.{table_name}.weight"
            tbe_table_idxs[tbe_idx] += 1
            assert sharded_weight_fqn in weights_spec
            wspec = weights_spec[sharded_weight_fqn]
            assert wspec.fqn == unsharded_weight_fqn
            assert wspec.shard_sizes == [size_r, size_c]
            assert wspec.shard_offsets == [offset_r, offset_c]
            assert wspec.sharding_type == sharding_type

            for qcomp in ["qscale", "qbias"]:
                sharded_weight_qcomp_fqn: str = f"{sharded_weight_fqn}_{qcomp}"
                assert sharded_weight_qcomp_fqn in weights_spec
                wqcomp_spec = weights_spec[sharded_weight_qcomp_fqn]
                assert wqcomp_spec.fqn == f"{unsharded_weight_fqn}_{qcomp}"
                assert wqcomp_spec.shard_sizes == [size_r, 2]
                assert wqcomp_spec.shard_offsets == [offset_r, 0]
                assert wqcomp_spec.sharding_type == sharding_type
