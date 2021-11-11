#!/usr/bin/env python3

import abc
import copy
import itertools
from collections import OrderedDict
from typing import List, Optional, Dict, Any, Union, Tuple, cast, Iterator

import torch
import torch.distributed as dist
import torch.distributed._sharded_tensor as sharded_tensor
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    EmbeddingLocation,
    ComputeDevice,
    PoolingMode,
    DenseTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    BaseEmbeddingLookup,
    SparseFeatures,
    EmbeddingComputeKernel,
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.grouped_position_weighted import (
    GroupedPositionWeightedModule,
)
from torchrec.distributed.types import (
    Shard,
    ShardedTensorMetadata,
    ShardMetadata,
    ShardedTensor,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    PoolingType,
    DataType,
    DATA_TYPE_NUM_BITS,
)
from torchrec.optim.fused import FusedOptimizerModule, FusedOptimizer
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)


def _load_state_dict(
    emb_modules: "nn.ModuleList[nn.Module]",
    state_dict: "OrderedDict[str, torch.Tensor]",
) -> Tuple[List[str], List[str]]:
    missing_keys = []
    unexpected_keys = list(state_dict.keys())
    for emb_module in emb_modules:
        for key, param in emb_module.state_dict().items():
            if key in state_dict:
                if isinstance(param, ShardedTensor):
                    assert len(param.local_shards()) == 1
                    dst_tensor = param.local_shards()[0].tensor
                else:
                    dst_tensor = param
                if isinstance(state_dict[key], ShardedTensor):
                    # pyre-fixme[16]
                    assert len(state_dict[key].local_shards()) == 1
                    src_tensor = state_dict[key].local_shards()[0].tensor
                else:
                    src_tensor = state_dict[key]
                dst_tensor.detach().copy_(src_tensor)
                unexpected_keys.remove(key)
            else:
                missing_keys.append(cast(str, key))
    return missing_keys, unexpected_keys


class BaseEmbedding(abc.ABC, nn.Module):
    """
    abstract base class for grouped nn.Embedding
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        pass

    """
    return sparse gradient parameter names
    """

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination


class GroupedEmbedding(BaseEmbedding):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        sparse: bool,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        self._pg = pg
        self._emb_modules: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._sparse = sparse
        for embedding_config in self._config.embedding_tables:
            self._emb_modules.append(
                nn.Embedding(
                    num_embeddings=embedding_config.local_rows,
                    embedding_dim=embedding_config.local_cols,
                    device=device,
                    sparse=self._sparse,
                    _weight=torch.empty(
                        embedding_config.local_rows,
                        embedding_config.local_cols,
                        device=device,
                    ).uniform_(
                        embedding_config.get_weight_init_min(),
                        embedding_config.get_weight_init_max(),
                    ),
                )
            )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        indices_dict: Dict[str, torch.Tensor] = {}
        indices_list = torch.split(features.values(), features.length_per_key())
        for key, indices in zip(features.keys(), indices_list):
            indices_dict[key] = indices
        unpooled_embeddings: List[torch.Tensor] = []
        for embedding_config, emb_module in zip(
            self._config.embedding_tables, self._emb_modules
        ):
            for feature_name in embedding_config.feature_names:
                unpooled_embeddings.append(emb_module(input=indices_dict[feature_name]))
        return torch.cat(unpooled_embeddings, dim=0)

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

        for config, emb_module in zip(
            self._config.embedding_tables,
            self._emb_modules,
        ):
            key = prefix + f"{config.name}.weight"
            param = emb_module.weight if keep_vars else emb_module.weight.data
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            if config.global_metadata is not None:
                # set additional field of sharded tensor based on local tensor properties
                config.global_metadata.tensor_properties.dtype = param.dtype
                config.global_metadata.tensor_properties.requires_grad = (
                    param.requires_grad
                )
                destination[key] = sharded_tensor.init_from_local_shards(
                    local_shards=[Shard(param, config.local_metadata)],
                    sharded_tensor_metadata=config.global_metadata,
                    process_group=self._pg,
                )
            else:
                destination[key] = param

        return destination

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for config, emb_module in zip(
            self._config.embedding_tables,
            self._emb_modules,
        ):
            param = emb_module.weight
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            yield append_prefix(prefix, f"{config.name}.weight"), param

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        if self._sparse:
            for config in self._config.embedding_tables:
                destination.append(append_prefix(prefix, f"{config.name}.weight"))
        return destination

    def config(self) -> GroupedEmbeddingConfig:
        return self._config


class GroupedEmbeddingsLookup(BaseEmbeddingLookup):
    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
        ) -> BaseEmbedding:
            if config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return GroupedEmbedding(
                    config=config,
                    sparse=False,
                    pg=pg,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.SPARSE:
                return GroupedEmbedding(
                    config=config,
                    sparse=True,
                    pg=pg,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        super().__init__()
        self._emb_modules: nn.ModuleList[BaseEmbedding] = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config))

        self._id_list_feature_splits: List[int] = []
        for config in grouped_configs:
            self._id_list_feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> torch.Tensor:
        assert sparse_features.id_list_features is not None
        embeddings: List[torch.Tensor] = []
        id_list_features_by_group = sparse_features.id_list_features.split(
            self._id_list_feature_splits,
        )
        for emb_op, features in zip(self._emb_modules, id_list_features_by_group):
            embeddings.append(emb_op(features).view(-1))

        if len(embeddings) == 0:
            # a hack for empty ranks
            return self._dummy_embs_tensor
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings)

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

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        # pyre-ignore [6]
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        for emb_module in self._emb_modules:
            emb_module.sparse_grad_parameter_names(destination, prefix)
        return destination


class BaseEmbeddingBag(nn.Module):
    """
    abstract base class for grouped nn.EmbeddingBag
    """

    """
    return sparse gradient parameter names
    """

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination

    @property
    @abc.abstractmethod
    def config(self) -> GroupedEmbeddingConfig:
        pass


class GroupedEmbeddingBag(BaseEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        sparse: bool,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        def _to_mode(pooling: PoolingType) -> str:
            if pooling == PoolingType.SUM:
                return "sum"
            elif pooling == PoolingType.MEAN:
                return "mean"
            else:
                raise ValueError(f"Unsupported pooling {pooling}")

        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        self._pg = pg
        self._emb_modules: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._sparse = sparse
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []

        shared_feature: Dict[str, bool] = {}
        for embedding_config in self._config.embedding_tables:
            self._emb_modules.append(
                nn.EmbeddingBag(
                    num_embeddings=embedding_config.local_rows,
                    embedding_dim=embedding_config.local_cols,
                    mode=_to_mode(embedding_config.pooling),
                    device=device,
                    include_last_offset=True,
                    sparse=self._sparse,
                    _weight=torch.empty(
                        embedding_config.local_rows,
                        embedding_config.local_cols,
                        device=device,
                    ).uniform_(
                        embedding_config.get_weight_init_min(),
                        embedding_config.get_weight_init_max(),
                    ),
                )
            )
            for feature_name in embedding_config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True
                self._lengths_per_emb.append(embedding_config.embedding_dim)

        for embedding_config in self._config.embedding_tables:
            for feature_name in embedding_config.feature_names:
                if shared_feature[feature_name]:
                    self._emb_names.append(feature_name + "@" + embedding_config.name)
                else:
                    self._emb_names.append(feature_name)

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        pooled_embeddings: List[torch.Tensor] = []
        for embedding_config, emb_module in zip(
            self._config.embedding_tables, self._emb_modules
        ):
            for feature_name in embedding_config.feature_names:
                values = features[feature_name].values()
                offsets = features[feature_name].offsets()
                weights = features[feature_name].weights_or_none()
                if weights is not None and not torch.is_floating_point(weights):
                    weights = None
                pooled_embeddings.append(
                    emb_module(
                        input=values,
                        offsets=offsets,
                        per_sample_weights=weights,
                    )
                )
        return KeyedTensor(
            keys=self._emb_names,
            values=torch.cat(pooled_embeddings, dim=1),
            length_per_key=self._lengths_per_emb,
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

        for config, emb_module in zip(
            self._config.embedding_tables,
            self._emb_modules,
        ):
            key = prefix + f"{config.name}.weight"
            param = emb_module.weight if keep_vars else emb_module.weight.data
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            if config.global_metadata is not None:
                # set additional field of sharded tensor based on local tensor properties
                config.global_metadata.tensor_properties.dtype = param.dtype
                config.global_metadata.tensor_properties.requires_grad = (
                    param.requires_grad
                )
                destination[key] = sharded_tensor.init_from_local_shards(
                    local_shards=[Shard(param, config.local_metadata)],
                    sharded_tensor_metadata=config.global_metadata,
                    process_group=self._pg,
                )
            else:
                destination[key] = param

        return destination

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for config, emb_module in zip(
            self._config.embedding_tables,
            self._emb_modules,
        ):
            param = emb_module.weight
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            yield append_prefix(prefix, f"{config.name}.weight"), param

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        if self._sparse:
            for config in self._config.embedding_tables:
                destination.append(append_prefix(prefix, f"{config.name}.weight"))
        return destination

    def config(self) -> GroupedEmbeddingConfig:
        return self._config


class BaseBatchedEmbeddingBag(BaseEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        self._pg = pg

        def to_pooling_mode(pooling_type: PoolingType) -> PoolingMode:
            if pooling_type == PoolingType.SUM:
                return PoolingMode.SUM
            else:
                assert pooling_type == PoolingType.MEAN
                return PoolingMode.MEAN

        self._pooling: PoolingMode = to_pooling_mode(config.pooling)

        self._local_rows: List[int] = []
        self._weight_init_mins: List[float] = []
        self._weight_init_maxs: List[float] = []
        self._num_embeddings: List[int] = []
        self._local_cols: List[int] = []
        self._feature_table_map: List[int] = []
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []

        shared_feature: Dict[str, bool] = {}
        for idx, config in enumerate(self._config.embedding_tables):
            self._local_rows.append(config.local_rows)
            self._weight_init_mins.append(config.get_weight_init_min())
            self._weight_init_maxs.append(config.get_weight_init_max())
            self._num_embeddings.append(config.num_embeddings)
            self._local_cols.append(config.local_cols)
            self._feature_table_map.extend([idx] * config.num_features())
            for feature_name in config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True
                self._lengths_per_emb.append(config.embedding_dim)

        for embedding_config in self._config.embedding_tables:
            for feature_name in embedding_config.feature_names:
                if shared_feature[feature_name]:
                    self._emb_names.append(feature_name + "@" + embedding_config.name)
                else:
                    self._emb_names.append(feature_name)

    def init_parameters(self) -> None:
        # initialize embedding weights
        assert len(self._num_embeddings) == len(self.split_embedding_weights())
        for (rows, emb_dim, weight_init_min, weight_init_max, param) in zip(
            self._local_rows,
            self._local_cols,
            self._weight_init_mins,
            self._weight_init_maxs,
            self.split_embedding_weights(),
        ):
            assert param.shape == (rows, emb_dim)
            param.data.uniform_(
                weight_init_min,
                weight_init_max,
            )

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        weights = features.weights_or_none()
        if weights is not None and not torch.is_floating_point(weights):
            weights = None
        values = self.emb_module(
            indices=features.values().long(),
            offsets=features.offsets().long(),
            per_sample_weights=weights,
        )
        return KeyedTensor(
            keys=self._emb_names,
            values=values,
            length_per_key=self._lengths_per_emb,
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
        for config, param in zip(
            self._config.embedding_tables,
            self.split_embedding_weights(),
        ):
            key = prefix + f"{config.name}.weight"
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            if config.global_metadata is not None:
                # set additional field of sharded tensor based on local tensor properties
                config.global_metadata.tensor_properties.dtype = param.dtype
                config.global_metadata.tensor_properties.requires_grad = (
                    param.requires_grad
                )
                destination[key] = sharded_tensor.init_from_local_shards(
                    local_shards=[Shard(param, config.local_metadata)],
                    sharded_tensor_metadata=config.global_metadata,
                    process_group=self._pg,
                )
            else:
                destination[key] = param
        return destination

    def split_embedding_weights(self) -> List[torch.Tensor]:
        return self.emb_module.split_embedding_weights()

    @property
    @abc.abstractmethod
    def emb_module(
        self,
    ) -> Union[
        DenseTableBatchedEmbeddingBagsCodegen,
        SplitTableBatchedEmbeddingBagsCodegen,
        IntNBitTableBatchedEmbeddingBagsCodegen,
    ]:
        ...

    def config(self) -> GroupedEmbeddingConfig:
        return self._config


class EmbeddingBagFusedOptimizer(FusedOptimizer):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        emb_module: SplitTableBatchedEmbeddingBagsCodegen,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = emb_module
        self._pg = pg

        def to_rowwise_sharded_metadata(
            local_metadata: ShardMetadata,
            global_metadata: ShardedTensorMetadata,
            sharding_dim: int,
        ) -> Tuple[ShardMetadata, ShardedTensorMetadata]:
            rw_shards: List[ShardMetadata] = []
            rw_local_shard: ShardMetadata = local_metadata
            shards_metadata = global_metadata.shards_metadata
            # column-wise sharding
            # sort the metadata based on column offset and
            # we construct the momentum tensor in row-wise sharded way
            if sharding_dim == 1:
                shards_metadata = sorted(
                    shards_metadata, key=lambda shard: shard.shard_offsets[1]
                )

            for idx, shard in enumerate(shards_metadata):
                offset = shard.shard_offsets[0]
                # for column-wise sharding, we still create row-wise sharded metadata for optimizer
                # manually create a row-wise offset
                if sharding_dim == 1:
                    offset = idx * shard.shard_lengths[0]
                rw_shard = ShardMetadata(
                    shard_lengths=[shard.shard_lengths[0]],
                    shard_offsets=[offset],
                    placement=shard.placement,
                )

                if local_metadata == shard:
                    rw_local_shard = rw_shard

                rw_shards.append(rw_shard)

            tensor_properties = sharded_tensor.TensorProperties(
                dtype=global_metadata.tensor_properties.dtype,
                layout=global_metadata.tensor_properties.layout,
                requires_grad=global_metadata.tensor_properties.requires_grad,
                memory_format=global_metadata.tensor_properties.memory_format,
                pin_memory=global_metadata.tensor_properties.pin_memory,
            )
            len_rw_shards = len(shards_metadata) if sharding_dim == 1 else 1
            rw_metadata = ShardedTensorMetadata(
                shards_metadata=rw_shards,
                size=torch.Size([global_metadata.size[0] * len_rw_shards]),
                tensor_properties=tensor_properties,
            )
            return rw_local_shard, rw_metadata

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.optimizer_args.learning_rate,
        }
        params: Dict[str, torch.Tensor] = {}

        # Fused optimizers use buffers (they don't use autograd) and we want to make sure
        # that state_dict look identical to non-fused version.
        split_embedding_weights = emb_module.split_embedding_weights()
        for table_config, weight in zip(
            config.embedding_tables,
            split_embedding_weights,
        ):
            param_group["params"].append(weight)
            param_key = table_config.name + ".weight"
            params[param_key] = weight

        for table_config, optimizer_states, weight in zip(
            config.embedding_tables,
            emb_module.split_optimizer_states(),
            split_embedding_weights,
        ):
            state[weight] = {}
            # momentum1
            assert table_config.local_rows == optimizer_states[0].size(0)
            sharding_dim = (
                1 if table_config.local_cols != table_config.embedding_dim else 0
            )
            momentum1_key = f"{table_config.name}.momentum1"
            if optimizer_states[0].dim() == 1:
                (local_metadata, sharded_tensor_metadata) = to_rowwise_sharded_metadata(
                    table_config.local_metadata,
                    table_config.global_metadata,
                    sharding_dim,
                )
            else:
                (local_metadata, sharded_tensor_metadata) = (
                    table_config.local_metadata,
                    table_config.global_metadata,
                )

            momentum1 = sharded_tensor.init_from_local_shards(
                local_shards=[Shard(optimizer_states[0], local_metadata)],
                sharded_tensor_metadata=sharded_tensor_metadata,
                process_group=self._pg,
            )
            state[weight][momentum1_key] = momentum1
            # momentum2
            if len(optimizer_states) == 2:
                assert table_config.local_rows == optimizer_states[1].size(0)
                momentum2_key = f"{table_config.name}.momentum2"

                if optimizer_states[1].dim() == 1:
                    (
                        local_metadata,
                        sharded_tensor_metadata,
                    ) = to_rowwise_sharded_metadata(
                        table_config.local_metadata,
                        table_config.global_metadata,
                        sharding_dim,
                    )
                else:
                    (local_metadata, sharded_tensor_metadata) = (
                        table_config.local_metadata,
                        table_config.global_metadata,
                    )
                momentum2 = sharded_tensor.init_from_local_shards(
                    local_shards=[Shard(optimizer_states[1], local_metadata)],
                    sharded_tensor_metadata=sharded_tensor_metadata,
                    process_group=self._pg,
                )
                state[weight][momentum2_key] = momentum2

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])


class BatchedFusedEmbeddingBag(BaseBatchedEmbeddingBag, FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, pg, device)

        def to_embedding_location(
            compute_kernel: EmbeddingComputeKernel,
        ) -> EmbeddingLocation:
            if compute_kernel == EmbeddingComputeKernel.BATCHED_FUSED:
                return EmbeddingLocation.DEVICE
            elif compute_kernel == EmbeddingComputeKernel.BATCHED_FUSED_UVM:
                return EmbeddingLocation.MANAGED
            elif compute_kernel == EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING:
                return EmbeddingLocation.MANAGED_CACHING
            else:
                raise ValueError(f"Invalid EmbeddingComputeKernel {compute_kernel}")

        managed: List[EmbeddingLocation] = []
        compute_devices: List[ComputeDevice] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(to_embedding_location(table.compute_kernel))
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)
        if fused_params is None:
            fused_params = {}
        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=list(
                    zip(self._local_rows, self._local_cols, managed, compute_devices)
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                weights_precision=BatchedFusedEmbeddingBag.to_sparse_type(
                    config.data_type
                ),
                device=device,
                **fused_params,
            )
        )
        self._optim: EmbeddingBagFusedOptimizer = EmbeddingBagFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )

        self.init_parameters()

    @staticmethod
    def to_sparse_type(data_type: DataType) -> SparseType:
        if data_type == DataType.FP32:
            return SparseType.FP32
        elif data_type == DataType.FP16:
            return SparseType.FP16
        elif data_type == DataType.INT8:
            return SparseType.INT8
        else:
            raise ValueError(f"Invalid DataType {data_type}")

    @property
    def emb_module(
        self,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        yield from ()

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for config, param in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, param


class BatchedDenseEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._local_cols)),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                use_cpu=device is None or device.type == "cpu",
            )
        )

        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> DenseTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        combined_key = "/".join(
            [config.name for config in self._config.embedding_tables]
        )
        yield append_prefix(prefix, f"{combined_key}.weight"), cast(
            nn.Parameter, self._emb_module.weights
        )


class QuantBatchedEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = (
            IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        "",
                        local_rows,
                        table.embedding_dim,
                        QuantBatchedEmbeddingBag.to_sparse_type(config.data_type),
                        EmbeddingLocation.DEVICE
                        if (device is not None and device.type == "cuda")
                        else EmbeddingLocation.HOST,
                    )
                    for local_rows, table in zip(
                        self._local_rows, config.embedding_tables
                    )
                ],
                pooling_mode=self._pooling,
            )
        )
        if device is not None and device.type != "meta":
            self._emb_module.initialize_weights()

    @staticmethod
    def to_sparse_type(data_type: DataType) -> SparseType:
        if data_type == DataType.FP16:
            return SparseType.FP16
        elif data_type == DataType.INT8:
            return SparseType.INT8
        elif data_type == DataType.INT4:
            return SparseType.INT4
        elif data_type == DataType.INT2:
            return SparseType.INT2
        else:
            raise ValueError(f"Invalid DataType {data_type}")

    def init_parameters(self) -> None:
        pass

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        values = self.emb_module(
            indices=features.values().int(),
            offsets=features.offsets().int(),
            per_sample_weights=features.weights_or_none(),
        )
        return KeyedTensor(
            keys=self._emb_names,
            values=values,
            length_per_key=self._lengths_per_emb,
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for config, weight in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            yield append_prefix(prefix, f"{config.name}.weight"), weight[0]

    def split_embedding_weights(self) -> List[torch.Tensor]:
        return [
            weight
            for weight, _ in self.emb_module.split_embedding_weights(
                split_scale_shifts=False
            )
        ]

    @classmethod
    def from_float(cls, module: BaseEmbeddingBag) -> "QuantBatchedEmbeddingBag":
        assert hasattr(
            module, "qconfig"
        ), "EmbeddingBagCollectionInterface input float module must have qconfig defined"

        def _to_data_type(dtype: torch.dtype) -> DataType:
            if dtype == torch.quint8 or dtype == torch.qint8:
                return DataType.INT8
            elif dtype == torch.quint4 or dtype == torch.qint4:
                return DataType.INT4
            elif dtype == torch.quint2 or dtype == torch.qint2:
                return DataType.INT2
            else:
                raise Exception(f"Invalid data type {dtype}")

        # pyre-ignore [16]
        data_type = _to_data_type(module.qconfig.weight().dtype)
        sparse_type = QuantBatchedEmbeddingBag.to_sparse_type(data_type)

        state_dict = dict(
            itertools.chain(module.named_buffers(), module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        # Adjust config to quantized version.
        # This obviously doesn't work for column-wise sharding.
        # pyre-ignore [29]
        config = copy.deepcopy(module.config())
        config.data_type = data_type
        for table in config.embedding_tables:
            table.local_cols = rounded_row_size_in_bytes(table.local_cols, sparse_type)
            if table.local_metadata is not None:
                table.local_metadata.shard_lengths = [
                    table.local_rows,
                    table.local_cols,
                ]
            if table.global_metadata is not None:
                for shard_meta in table.global_metadata.shards_metadata:
                    if shard_meta != table.local_metadata:
                        shard_meta.shard_lengths = [
                            shard_meta.shard_lengths[0],
                            rounded_row_size_in_bytes(
                                shard_meta.shard_lengths[1], sparse_type
                            ),
                        ]
                table.global_metadata.size = torch.Size(
                    [
                        table.global_metadata.size[0],
                        sum(
                            shard_meta.shard_lengths[1]
                            for shard_meta in table.global_metadata.shards_metadata
                        ),
                    ]
                )

        ret = QuantBatchedEmbeddingBag(config=config, device=device)

        # Quantize weights.
        quant_weight_list = []
        for _, weight in state_dict.items():
            quantized_weights = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                weight, DATA_TYPE_NUM_BITS[data_type]
            )
            # weight and 4 byte scale shift (2xfp16)
            quant_weight = quantized_weights[:, :-4]
            scale_shift = quantized_weights[:, -4:]

            quant_weight_list.append((quant_weight, scale_shift))
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret


class GroupedPooledEmbeddingsLookup(BaseEmbeddingLookup):
    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        grouped_score_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        pg: Optional[dist.ProcessGroup] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
        ) -> BaseEmbeddingBag:
            if config.compute_kernel == EmbeddingComputeKernel.BATCHED_DENSE:
                return BatchedDenseEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.BATCHED_FUSED:
                return BatchedFusedEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                    fused_params=fused_params,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return GroupedEmbeddingBag(
                    config=config,
                    sparse=False,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.SPARSE:
                return GroupedEmbeddingBag(
                    config=config,
                    sparse=True,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.BATCHED_QUANT:
                return QuantBatchedEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        super().__init__()
        self._emb_modules: nn.ModuleList[BaseEmbeddingBag] = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config))

        self._score_emb_modules: nn.ModuleList[BaseEmbeddingBag] = nn.ModuleList()
        for config in grouped_score_configs:
            self._score_emb_modules.append(_create_lookup(config))

        self._id_list_feature_splits: List[int] = []
        for config in grouped_configs:
            self._id_list_feature_splits.append(config.num_features())
        self._id_score_list_feature_splits: List[int] = []
        for config in grouped_score_configs:
            self._id_score_list_feature_splits.append(config.num_features())

        # return a dummy empty tensor
        # when grouped_configs and grouped_score_configs are empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs
        self.grouped_score_configs = grouped_score_configs
        self._feature_processor = feature_processor

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> torch.Tensor:
        assert (
            sparse_features.id_list_features is not None
            or sparse_features.id_score_list_features is not None
        )
        embeddings: List[torch.Tensor] = []
        if len(self._emb_modules) > 0:
            assert sparse_features.id_list_features is not None
            id_list_features_by_group = sparse_features.id_list_features.split(
                self._id_list_feature_splits,
            )
            for config, emb_op, features in zip(
                self.grouped_configs, self._emb_modules, id_list_features_by_group
            ):
                if (
                    config.has_feature_processor
                    and self._feature_processor is not None
                    and isinstance(
                        self._feature_processor, GroupedPositionWeightedModule
                    )
                ):
                    features = self._feature_processor(features)
                embeddings.append(emb_op(features).values())
        if len(self._score_emb_modules) > 0:
            assert sparse_features.id_score_list_features is not None
            id_score_list_features_by_group = (
                sparse_features.id_score_list_features.split(
                    self._id_score_list_feature_splits,
                )
            )
            for emb_op, features in zip(
                self._score_emb_modules, id_score_list_features_by_group
            ):
                embeddings.append(emb_op(features).values())

        if len(embeddings) == 0:
            # a hack for empty ranks
            batch_size: int = (
                sparse_features.id_list_features.stride()
                if sparse_features.id_list_features is not None
                # pyre-fixme[16]: `Optional` has no attribute `stride`.
                else sparse_features.id_score_list_features.stride()
            )
            return self._dummy_embs_tensor.view(batch_size, 0)
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings, dim=1)

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

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)
        for emb_module in self._score_emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        # pyre-ignore [6]
        m1, u1 = _load_state_dict(self._emb_modules, state_dict)
        # pyre-ignore [6]
        m2, u2 = _load_state_dict(self._score_emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m1 + m2, unexpected_keys=u1 + u2)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)
        for emb_module in self._score_emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)
        for emb_module in self._score_emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        for emb_module in self._emb_modules:
            emb_module.sparse_grad_parameter_names(destination, prefix)
        for emb_module in self._score_emb_modules:
            emb_module.sparse_grad_parameter_names(destination, prefix)
        return destination
