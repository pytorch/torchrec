#!/usr/bin/env python3

import abc
from collections import OrderedDict
from math import sqrt
from typing import List, Optional, Dict, Any, Union

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    EmbeddingLocation,
    ComputeDevice,
    PoolingMode,
    DenseTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    BaseEmbeddingLookup,
    SparseFeatures,
    EmbeddingComputeKernel,
)
from torchrec.distributed.types import (
    ShardedTensor,
    ShardMetadata,
    ShardedTensorMetadata,
)
from torchrec.modules.embedding_configs import (
    PoolingType,
    DataType,
)
from torchrec.optim.fused import FusedOptimizerModule, FusedOptimizer
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)


def wrap_weight_to_parameter(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    for i, v in enumerate(weights):
        if not isinstance(v, torch.nn.Parameter):
            weights[i] = torch.nn.Parameter(v)
    return weights


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
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
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
                        -sqrt(1 / embedding_config.num_embeddings),
                        sqrt(1 / embedding_config.num_embeddings),
                    ),
                )
            )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        unpooled_embeddings: List[torch.Tensor] = []
        for embedding_config, emb_module in zip(
            self._config.embedding_tables, self._emb_modules
        ):
            for feature_name in embedding_config.feature_names:
                values = features[feature_name].values()
                unpooled_embeddings.append(emb_module(input=values))
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
            param = emb_module.weight if keep_vars else emb_module.weight.data
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            destination[prefix + f"{config.name}.weight"] = param
        return destination

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        if self._sparse:
            for config in self._config.embedding_tables:
                destination.append(prefix + f"{config.name}.weight")
        return destination


class GroupedEmbeddingsLookup(BaseEmbeddingLookup):
    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
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
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.SPARSE:
                return GroupedEmbedding(
                    config=config,
                    sparse=True,
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
            self._id_list_feature_splits
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

        def update_destination(
            # pyre-ignore [24]
            emb_modules: nn.ModuleList,
            grouped_configs: List[GroupedEmbeddingConfig],
        ) -> None:
            for emb_module, config in zip(emb_modules, grouped_configs):
                for (key, param), table_config in zip(
                    emb_module.state_dict(None, prefix, keep_vars).items(),
                    config.embedding_tables,
                ):
                    # pyre-ignore [16]
                    destination[key] = (
                        ShardedTensor(
                            local_shard=param, sharding_metadata=table_config.metadata
                        )
                        if table_config.sharded_tensor
                        else param
                    )

        update_destination(self._emb_modules, self.grouped_configs)

        return destination

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        for emb_module in self._emb_modules:
            emb_module.sparse_grad_parameter_names(destination, prefix)
        return destination


class BaseEmbeddingBag(abc.ABC, nn.Module):
    """
    abstract base class for grouped nn.EmbeddingBag
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        pass

    """
    return sparse gradient parameter names
    """

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination


class GroupedEmbeddingBag(BaseEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        sparse: bool,
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
                        -sqrt(1 / embedding_config.num_embeddings),
                        sqrt(1 / embedding_config.num_embeddings),
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
            param = emb_module.weight if keep_vars else emb_module.weight.data
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            destination[prefix + f"{config.name}.weight"] = param
        return destination

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        if self._sparse:
            for config in self._config.embedding_tables:
                destination.append(prefix + f"{config.name}.weight")
        return destination


class BaseBatchedEmbeddingBag(BaseEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config

        def to_pooling_mode(pooling_type: PoolingType) -> PoolingMode:
            if pooling_type == PoolingType.SUM:
                return PoolingMode.SUM
            else:
                assert pooling_type == PoolingType.MEAN
                return PoolingMode.MEAN

        self._pooling: PoolingMode = to_pooling_mode(config.pooling)

        self._local_rows: List[int] = []
        self._num_embeddings: List[int] = []
        self._embedding_dims: List[int] = []
        self._feature_table_map: List[int] = []
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []

        shared_feature: Dict[str, bool] = {}
        for idx, config in enumerate(self._config.embedding_tables):
            self._local_rows.append(config.local_rows)
            self._num_embeddings.append(config.num_embeddings)
            self._embedding_dims.append(config.local_cols)
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
        assert len(self._num_embeddings) == len(
            self.emb_module.split_embedding_weights()
        )
        for (rows, num_emb, emb_dim, param) in zip(
            self._local_rows,
            self._num_embeddings,
            self._embedding_dims,
            wrap_weight_to_parameter(self.emb_module.split_embedding_weights()),
        ):
            assert param.shape == (rows, emb_dim)
            param.data.uniform_(-sqrt(1 / num_emb), sqrt(1 / num_emb))

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        values = self.emb_module(
            indices=features.values().long(),
            offsets=features.offsets().long(),
            per_sample_weights=features.weights_or_none(),
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
            self.emb_module.split_embedding_weights(),
        ):
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)

            # When optimizer is not fused
            # and we are asked to not detach,
            # we need to return full weight
            # as otherwise autograd won't work.
            if keep_vars and isinstance(
                self.emb_module,
                DenseTableBatchedEmbeddingBagsCodegen,
            ):
                param = self.emb_module.weights

            destination[prefix + f"{config.name}.weight"] = param
        return destination

    @property
    @abc.abstractmethod
    def emb_module(
        self,
    ) -> Union[
        DenseTableBatchedEmbeddingBagsCodegen,
        SplitTableBatchedEmbeddingBagsCodegen,
    ]:
        ...


class EmbeddingBagFusedOptimizer(FusedOptimizer):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        emb_module: SplitTableBatchedEmbeddingBagsCodegen,
    ) -> None:
        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = emb_module

        def to_sharded_metadata(
            metadata: ShardedTensorMetadata,
        ) -> ShardedTensorMetadata:
            rw_shards: List[ShardMetadata] = []
            for shard in metadata.shards:
                rw_shards.append(
                    ShardMetadata(
                        dims=[shard.dims[0]],
                        offsets=[shard.offsets[0]],
                    )
                )
            return ShardedTensorMetadata(
                shards=rw_shards,
            )

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.optimizer_args.learning_rate,
        }
        params: Dict[str, torch.Tensor] = {}

        # Fused optimizers use buffers (they don't use autograd) and we want to make sure
        # that state_dict look identical to non-fused version.
        split_embedding_weights = wrap_weight_to_parameter(
            emb_module.split_embedding_weights()
        )
        for table_config, weight in zip(
            config.embedding_tables,
            split_embedding_weights,
        ):
            param_group["params"].append(weight)
            param_key = "embedding_bags." + table_config.name + ".weight"
            params[param_key] = weight

        for table_config, optimizer_states, weight in zip(
            config.embedding_tables,
            emb_module.split_optimizer_states(),
            split_embedding_weights,
        ):
            state[weight] = {}
            # momentum1
            assert table_config.local_rows == optimizer_states[0].size(0)
            momentum1_key = f"{table_config.name}.momentum1"
            """
            TODO T99255928 Uncomment this code once moved to torch.dist.ShardedTensor,
            currently it's pickled in OptimizerAgent
            and we can't pickle types loaded from torch.package.
            momentum1 = ShardedTensor(
                local_shard=optimizer_states[0],
                sharding_metadata=to_sharded_metadata(table_config.metadata)
                if optimizer_states[0].dim() == 1
                else table_config.metadata,
            )"""
            momentum1 = optimizer_states[0]

            state[weight][momentum1_key] = momentum1
            # momentum2
            if len(optimizer_states) == 2:
                assert table_config.local_rows == optimizer_states[1].size(0)
                momentum2_key = f"{table_config.name}.momentum2"
                """
                TODO T99255928 Uncomment this code once moved to torch.dist.ShardedTensor,
                currently it's pickled in OptimizerAgent
                and we can't pickle types loaded from torch.package.
                momentum2 = ShardedTensor(
                    local_shard=optimizer_states[1],
                    sharding_metadata=to_sharded_metadata(table_config.metadata)
                    if optimizer_states[1].dim() == 1
                    else table_config.metadata,
                )
                """
                momentum2 = optimizer_states[1]
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
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, device)

        def to_sparse_type(data_type: DataType) -> SparseType:
            if data_type == DataType.FP32:
                return SparseType.FP32
            elif data_type == DataType.FP16:
                return SparseType.FP16
            elif data_type == DataType.INT8:
                return SparseType.INT8
            else:
                raise ValueError(f"Invalid DataType {data_type}")

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
                    zip(
                        self._local_rows, self._embedding_dims, managed, compute_devices
                    )
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                weights_precision=to_sparse_type(config.data_type),
                **fused_params,
            )
        )
        self._optim: EmbeddingBagFusedOptimizer = EmbeddingBagFusedOptimizer(
            config, self._emb_module
        )

        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim


class BatchedDenseEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, device)

        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._embedding_dims)),
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


class GroupedPooledEmbeddingsLookup(BaseEmbeddingLookup):
    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        grouped_score_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
        ) -> BaseEmbeddingBag:
            if config.compute_kernel == EmbeddingComputeKernel.BATCHED_DENSE:
                return BatchedDenseEmbeddingBag(
                    config=config,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.BATCHED_FUSED:
                return BatchedFusedEmbeddingBag(
                    config=config,
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
                self._id_list_feature_splits
            )
            for emb_op, features in zip(self._emb_modules, id_list_features_by_group):
                embeddings.append(emb_op(features).values())
        if len(self._score_emb_modules) > 0:
            assert sparse_features.id_score_list_features is not None
            id_score_list_features_by_group = (
                sparse_features.id_score_list_features.split(
                    self._id_score_list_feature_splits
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

        def update_destination(
            # pyre-ignore [24]
            emb_modules: nn.ModuleList,
            grouped_configs: List[GroupedEmbeddingConfig],
        ) -> None:
            for emb_module, config in zip(emb_modules, grouped_configs):
                for (key, param), table_config in zip(
                    emb_module.state_dict(None, prefix, keep_vars).items(),
                    config.embedding_tables,
                ):
                    # pyre-ignore [16]
                    destination[key] = (
                        ShardedTensor(
                            local_shard=param, sharding_metadata=table_config.metadata
                        )
                        if table_config.sharded_tensor
                        else param
                    )

        update_destination(self._emb_modules, self.grouped_configs)
        update_destination(self._score_emb_modules, self.grouped_score_configs)

        return destination

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        for emb_module in self._emb_modules:
            emb_module.sparse_grad_parameter_names(destination, prefix)
        for emb_module in self._score_emb_modules:
            emb_module.sparse_grad_parameter_names(destination, prefix)
        return destination
