#!/usr/bin/env python3

import copy
from collections import OrderedDict
from typing import (
    List,
    Dict,
    Optional,
    Type,
    Any,
    TypeVar,
    Mapping,
    Union,
    Tuple,
    Iterator,
    Set,
)

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.cw_sharding import CwEmbeddingSharding
from torchrec.distributed.dp_sharding import DpEmbeddingSharding
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    SparseFeaturesListAwaitable,
)
from torchrec.distributed.embedding_types import (
    SparseFeatures,
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    BaseEmbeddingLookup,
)
from torchrec.distributed.rw_sharding import RwEmbeddingSharding
from torchrec.distributed.tw_sharding import TwEmbeddingSharding
from torchrec.distributed.twrw_sharding import TwRwEmbeddingSharding
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    ShardedModule,
    ShardingType,
    ShardedModuleContext,
    ShardedTensor,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import EmbeddingTableConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import KeyedOptimizer, CombinedOptimizer
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def create_embedding_sharding(
    sharding_type: str,
    embedding_configs: List[Tuple[EmbeddingTableConfig, ParameterSharding]],
    pg: dist.ProcessGroup,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding:
    if sharding_type == ShardingType.TABLE_WISE.value:
        return TwEmbeddingSharding(embedding_configs, pg, device)
    elif sharding_type == ShardingType.ROW_WISE.value:
        return RwEmbeddingSharding(embedding_configs, pg, device)
    elif sharding_type == ShardingType.DATA_PARALLEL.value:
        return DpEmbeddingSharding(embedding_configs, pg, device)
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return TwRwEmbeddingSharding(embedding_configs, pg, device)
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return CwEmbeddingSharding(embedding_configs, pg, device)
    else:
        raise ValueError(f"Sharding not supported {sharding_type}")


def filter_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]", name: str
) -> "OrderedDict[str, torch.Tensor]":
    rtn_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(name):
            # + 1 to length is to remove the '.' after the key
            rtn_dict[key[len(name) + 1 :]] = value
    return rtn_dict


def _create_embedding_configs_by_sharding(
    module: EmbeddingBagCollection,
    pg: dist.ProcessGroup,
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
) -> Dict[str, List[Tuple[EmbeddingTableConfig, ParameterSharding]]]:
    shared_feature: Dict[str, bool] = {}
    for embedding_config in module.embedding_bag_configs:
        if not embedding_config.feature_names:
            embedding_config.feature_names = [embedding_config.name]
        for feature_name in embedding_config.feature_names:
            if feature_name not in shared_feature:
                shared_feature[feature_name] = False
            else:
                shared_feature[feature_name] = True

    sharding_type_to_embedding_configs: Dict[
        str, List[Tuple[EmbeddingTableConfig, ParameterSharding]]
    ] = {}
    for config in module.embedding_bag_configs:
        table_name = config.name
        assert table_name in table_name_to_parameter_sharding
        parameter_sharding = table_name_to_parameter_sharding[table_name]
        if parameter_sharding.compute_kernel not in [
            kernel.value for kernel in EmbeddingComputeKernel
        ]:
            raise ValueError(
                f"Compute kernel not supported {parameter_sharding.compute_kernel}"
            )
        embedding_names: List[str] = []
        for feature_name in config.feature_names:
            if shared_feature[feature_name]:
                embedding_names.append(feature_name + "@" + config.name)
            else:
                embedding_names.append(feature_name)

        if parameter_sharding.sharding_type not in sharding_type_to_embedding_configs:
            sharding_type_to_embedding_configs[parameter_sharding.sharding_type] = []
        sharding_type_to_embedding_configs[parameter_sharding.sharding_type].append(
            (
                EmbeddingTableConfig(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=config.embedding_dim,
                    name=config.name,
                    data_type=config.data_type,
                    feature_names=copy.deepcopy(config.feature_names),
                    pooling=config.pooling,
                    is_weighted=module.is_weighted,
                    embedding_names=embedding_names,
                ),
                parameter_sharding,
            )
        )
    return sharding_type_to_embedding_configs


class EmbeddingCollectionAwaitable(LazyAwaitable[KeyedTensor]):
    def __init__(
        self,
        awaitables: List[Awaitable[torch.Tensor]],
        embedding_dims: List[int],
        embedding_names: List[str],
    ) -> None:
        super().__init__()
        self._awaitables = awaitables
        self._embedding_dims = embedding_dims
        self._embedding_names = embedding_names

    def wait(self) -> KeyedTensor:
        embeddings = [w.wait() for w in self._awaitables]
        if len(embeddings) == 1:
            embeddings = embeddings[0]
        else:
            embeddings = torch.cat(embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            length_per_key=self._embedding_dims,
            values=embeddings,
            key_dim=1,
        )


class ShardedEmbeddingBagCollection(
    ShardedModule[
        List[SparseFeatures],
        List[torch.Tensor],
        KeyedTensor,
    ],
    FusedOptimizerModule,
):
    """
    Sharded implementation of EmbeddingBagCollection.
    This is part of public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        pg: dist.ProcessGroup,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        sharding_type_to_embedding_configs = _create_embedding_configs_by_sharding(
            module, pg, table_name_to_parameter_sharding
        )
        self._sharding_type_to_sharding: Dict[str, EmbeddingSharding] = {
            sharding_type: create_embedding_sharding(
                sharding_type, embedding_confings, pg, device
            )
            for sharding_type, embedding_confings in sharding_type_to_embedding_configs.items()
        }

        self._is_weighted: bool = module.is_weighted
        self._device = device
        self._create_lookups(fused_params)
        self._create_output_dist()
        self._input_dists: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_features_permute: bool = True

        # Get all fused optimizers and combine them.
        optims = []
        for lookup in self._lookups:
            for _, module in lookup.named_modules():
                if isinstance(module, FusedOptimizerModule):
                    # modify param keys to match EmbeddingBagCollection
                    params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in module.fused_optimizer.params.items():
                        params["embedding_bags." + param_key] = weight
                    module.fused_optimizer.params = params
                    optims.append(("", module.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:

        feature_names: List[str] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(
                sharding.id_score_list_feature_names()
                if self._is_weighted
                else sharding.id_list_feature_names()
            )
            self._feature_splits.append(
                len(
                    sharding.id_score_list_feature_names()
                    if self._is_weighted
                    else sharding.id_list_feature_names()
                )
            )

        if feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(
                    self._features_order, device=self._device, dtype=torch.int32
                ),
            )

    def _create_lookups(
        self,
        fused_params: Optional[Dict[str, Any]],
    ) -> None:
        self._lookups: nn.ModuleList[BaseEmbeddingLookup] = nn.ModuleList()
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup(fused_params))

    def _create_output_dist(self) -> None:
        self._output_dists: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_pooled_output_dist())
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())

    # pyre-ignore [14]
    def input_dist(
        self, ctx: ShardedModuleContext, features: KeyedJaggedTensor
    ) -> Awaitable[List[SparseFeatures]]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(features.keys())
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    # pyre-ignore [6]
                    self._features_order_tensor,
                )
            features_by_shards = features.split(
                self._feature_splits,
            )
            awaitables = [
                module(
                    SparseFeatures(
                        id_list_features=None
                        if self._is_weighted
                        else features_by_shard,
                        id_score_list_features=features_by_shard
                        if self._is_weighted
                        else None,
                    )
                )
                for module, features_by_shard in zip(
                    self._input_dists, features_by_shards
                )
            ]
            return SparseFeaturesListAwaitable(awaitables)

    def compute(
        self, ctx: ShardedModuleContext, dist_input: List[SparseFeatures]
    ) -> List[torch.Tensor]:
        return [lookup(features) for lookup, features in zip(self._lookups, dist_input)]

    def output_dist(
        self, ctx: ShardedModuleContext, output: List[torch.Tensor]
    ) -> LazyAwaitable[KeyedTensor]:
        return EmbeddingCollectionAwaitable(
            awaitables=[
                dist(embeddings) for dist, embeddings in zip(self._output_dists, output)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def compute_and_output_dist(
        self, ctx: ShardedModuleContext, input: List[SparseFeatures]
    ) -> LazyAwaitable[KeyedTensor]:
        return EmbeddingCollectionAwaitable(
            awaitables=[
                dist(lookup(features))
                for lookup, dist, features in zip(
                    self._lookups, self._output_dists, input
                )
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        for lookup in self._lookups:
            lookup.state_dict(destination, prefix + "embedding_bags.", keep_vars)
        return destination

    def named_modules(
        self,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, nn.Module]]:
        yield from [(prefix, self)]

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for lookup in self._lookups:
            yield from lookup.named_parameters(
                append_prefix(prefix, "embedding_bags"), recurse
            )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for lookup in self._lookups:
            yield from lookup.named_buffers(
                append_prefix(prefix, "embedding_bags"), recurse
            )

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        for lookup in self._lookups:
            missing, unexpected = lookup.load_state_dict(
                filter_state_dict(state_dict, "embedding_bags"),
                strict,
            )
            missing_keys.extend(missing)
            unexpected_keys.extend(unexpected)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def sparse_grad_parameter_names(
        self,
        destination: Optional[List[str]] = None,
        prefix: str = "",
    ) -> List[str]:
        destination = [] if destination is None else destination
        for lookup in self._lookups:
            lookup.sparse_grad_parameter_names(
                destination, append_prefix(prefix, "embedding_bags")
            )
        return destination

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim


M = TypeVar("M", bound=nn.Module)


class EmbeddingBagCollectionSharder(BaseEmbeddingSharder[M]):
    """
    This implementation uses non-fused EmbeddingBagCollection
    """

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingBagCollection:
        return ShardedEmbeddingBagCollection(
            module, params, pg, self.fused_params, device
        )

    def shardable_parameters(
        self, module: EmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }

    @property
    def module_type(self) -> Type[EmbeddingBagCollection]:
        return EmbeddingBagCollection
