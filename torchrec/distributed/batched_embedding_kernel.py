#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import copy
import itertools
from dataclasses import dataclass
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    PoolingMode,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_kernel import BaseEmbedding, get_state_dict
from torchrec.distributed.embedding_types import (
    compute_kernel_to_embedding_location,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.types import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
    TensorProperties,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    pooling_type_to_pooling_mode,
)
from torchrec.optim.fused import (
    EmptyFusedOptimizer,
    FusedOptimizer,
    FusedOptimizerModule,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class EmbeddingFusedOptimizer(FusedOptimizer):
    def __init__(  # noqa C901
        self,
        config: GroupedEmbeddingConfig,
        emb_module: SplitTableBatchedEmbeddingBagsCodegen,
        pg: Optional[dist.ProcessGroup] = None,
        create_for_table: Optional[str] = None,
        param_weight_for_table: Optional[nn.Parameter] = None,
    ) -> None:
        """
        Implementation of a FusedOptimizer. Designed as a base class Embedding kernels

        create_for_table is an optional flag, which if passed in only creates the optimizer for a single table.
        This optimizer shares data with the broader optimizer (one per embedding kernel)
        and is used to share step and LR changes
        """
        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = emb_module
        self._pg = pg

        @dataclass
        class ShardParams:
            optimizer_states: List[Optional[Tuple[torch.Tensor]]]
            local_metadata: List[ShardMetadata]
            embedding_weights: List[torch.Tensor]

        def get_optimizer_rowwise_shard_metadata_and_global_metadata(
            table_global_metadata: ShardedTensorMetadata,
            optimizer_state: torch.Tensor,
            sharding_dim: int,
        ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:

            table_global_shards_metadata: List[
                ShardMetadata
            ] = table_global_metadata.shards_metadata

            # column-wise sharding
            # sort the metadata based on column offset and
            # we construct the momentum tensor in row-wise sharded way
            if sharding_dim == 1:
                table_global_shards_metadata = sorted(
                    table_global_shards_metadata,
                    key=lambda shard: shard.shard_offsets[1],
                )

            table_shard_metadata_to_optimizer_shard_metadata = {}

            for idx, table_shard_metadata in enumerate(table_global_shards_metadata):
                offset = table_shard_metadata.shard_offsets[0]
                # for column-wise sharding, we still create row-wise sharded metadata for optimizer
                # manually create a row-wise offset

                if sharding_dim == 1:
                    offset = idx * table_shard_metadata.shard_sizes[0]

                table_shard_metadata_to_optimizer_shard_metadata[
                    table_shard_metadata
                ] = ShardMetadata(
                    shard_sizes=[table_shard_metadata.shard_sizes[0]],
                    shard_offsets=[offset],
                    placement=table_shard_metadata.placement,
                )

            tensor_properties = TensorProperties(
                dtype=optimizer_state.dtype,
                layout=optimizer_state.layout,
                requires_grad=False,
            )
            len_rw_shards = (
                len(table_shard_metadata_to_optimizer_shard_metadata)
                if sharding_dim == 1
                else 1
            )
            rowwise_optimizer_st_metadata = ShardedTensorMetadata(
                shards_metadata=list(
                    table_shard_metadata_to_optimizer_shard_metadata.values()
                ),
                size=torch.Size([table_global_metadata.size[0] * len_rw_shards]),
                tensor_properties=tensor_properties,
            )

            return (
                table_shard_metadata_to_optimizer_shard_metadata,
                rowwise_optimizer_st_metadata,
            )

        def get_optimizer_pointwise_shard_metadata_and_global_metadata(
            table_global_metadata: ShardedTensorMetadata,
            optimizer_state: torch.Tensor,
        ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
            table_global_shards_metadata: List[
                ShardMetadata
            ] = table_global_metadata.shards_metadata

            table_shard_metadata_to_optimizer_shard_metadata = {}

            for table_shard_metadata in table_global_shards_metadata:
                table_shard_metadata_to_optimizer_shard_metadata[
                    table_shard_metadata
                ] = ShardMetadata(
                    shard_sizes=table_shard_metadata.shard_sizes,
                    shard_offsets=table_shard_metadata.shard_offsets,
                    placement=table_shard_metadata.placement,
                )
            tensor_properties = TensorProperties(
                dtype=optimizer_state.dtype,
                layout=optimizer_state.layout,
                requires_grad=False,
            )
            pointwise_optimizer_st_metadata = ShardedTensorMetadata(
                shards_metadata=list(
                    table_shard_metadata_to_optimizer_shard_metadata.values()
                ),
                size=table_global_metadata.size,
                tensor_properties=tensor_properties,
            )

            return (
                table_shard_metadata_to_optimizer_shard_metadata,
                pointwise_optimizer_st_metadata,
            )

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.optimizer_args.learning_rate,
        }

        params: Dict[str, Union[torch.Tensor, ShardedTensor]] = {}

        # Fused optimizers use buffers (they don't use autograd) and we want to make sure
        # that state_dict look identical to no-fused version.
        table_to_shard_params: Dict[str, ShardParams] = {}

        split_embedding_weights = emb_module.split_embedding_weights()
        split_optimizer_states = emb_module.split_optimizer_states()

        for table_config, optimizer_states, weight in itertools.zip_longest(
            config.embedding_tables,
            split_optimizer_states,
            split_embedding_weights,
        ):
            # When EmbeddingFusedOptimizer is created for composability, only create state
            if create_for_table is not None and create_for_table != table_config.name:
                continue
            if table_config.name not in table_to_shard_params:
                table_to_shard_params[table_config.name] = ShardParams(
                    optimizer_states=[], local_metadata=[], embedding_weights=[]
                )

            if optimizer_states:
                for optimizer_state in optimizer_states:
                    assert table_config.local_rows == optimizer_state.size(0)

            local_metadata = table_config.local_metadata

            table_to_shard_params[table_config.name].optimizer_states.append(
                optimizer_states
            )
            table_to_shard_params[table_config.name].local_metadata.append(
                local_metadata
            )
            table_to_shard_params[table_config.name].embedding_weights.append(weight)

        seen_tables = set()
        for table_config in config.embedding_tables:
            if create_for_table is not None and create_for_table != table_config.name:
                continue
            if table_config.name in seen_tables:
                continue
            seen_tables.add(table_config.name)
            table_config_global_metadata: Optional[
                ShardedTensorMetadata
            ] = copy.deepcopy(table_config.global_metadata)

            shard_params: ShardParams = table_to_shard_params[table_config.name]

            assert table_config_global_metadata is not None
            if create_for_table is None:
                local_weight_shards = []
                for local_weight, local_metadata in zip(
                    shard_params.embedding_weights, shard_params.local_metadata
                ):
                    local_weight_shards.append(Shard(local_weight, local_metadata))
                    table_config_global_metadata.tensor_properties.dtype = (
                        local_weight.dtype
                    )
                    table_config_global_metadata.tensor_properties.requires_grad = (
                        local_weight.requires_grad
                    )
                # TODO share this logic to create the same TableBatchedEmbeddingSlice in FusedModules below
                weight = ShardedTensor._init_from_local_shards_and_global_metadata(
                    local_shards=local_weight_shards,
                    sharded_tensor_metadata=table_config_global_metadata,
                    process_group=self._pg,
                )
                param_key = table_config.name + ".weight"
            else:
                assert (
                    param_weight_for_table is not None
                ), "param_weight_for_table cannot be None when using create_for_table"
                weight = param_weight_for_table
                param_key = ""

            state[weight] = {}
            param_group["params"].append(weight)
            params[param_key] = weight

            # Setting optimizer states
            sharding_dim: int = (
                1 if table_config.local_cols != table_config.embedding_dim else 0
            )

            if all(
                [opt_state is not None for opt_state in shard_params.optimizer_states]
            ):
                # pyre-ignore
                def get_momentum(momentum_idx: int) -> ShardedTensor:
                    assert momentum_idx > 0
                    momentum_local_shards: List[Shard] = []
                    optimizer_sharded_tensor_metadata: ShardedTensorMetadata

                    is_rowwise_optimizer_state: bool = (
                        # pyre-ignore
                        shard_params.optimizer_states[0][momentum_idx - 1].dim()
                        == 1
                    )

                    if is_rowwise_optimizer_state:
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_rowwise_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            shard_params.optimizer_states[0][momentum_idx - 1],
                            sharding_dim,
                        )
                    else:
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_pointwise_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            shard_params.optimizer_states[0][momentum_idx - 1],
                        )

                    for (optimizer_state, table_shard_local_metadata) in zip(
                        shard_params.optimizer_states, shard_params.local_metadata
                    ):
                        local_optimizer_shard_metadata = (
                            table_shard_metadata_to_optimizer_shard_metadata[
                                table_shard_local_metadata
                            ]
                        )
                        momentum_local_shards.append(
                            Shard(
                                optimizer_state[momentum_idx - 1],
                                local_optimizer_shard_metadata,
                            )
                        )

                    # TODO we should be creating this in SPMD fashion (e.g. init_from_local_shards), and let it derive global metadata.
                    return ShardedTensor._init_from_local_shards_and_global_metadata(
                        local_shards=momentum_local_shards,
                        sharded_tensor_metadata=optimizer_sharded_tensor_metadata,
                        process_group=self._pg,
                    )

                if all(
                    # pyre-ignore
                    [len(opt_state) >= 1 for opt_state in shard_params.optimizer_states]
                ):
                    state[weight][f"{table_config.name}.momentum1"] = get_momentum(1)
                if all(
                    # pyre-ignore
                    [len(opt_state) >= 2 for opt_state in shard_params.optimizer_states]
                ):
                    state[weight][f"{table_config.name}.momentum2"] = get_momentum(2)

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])


def _gen_named_parameters_by_table_fused(
    emb_module: SplitTableBatchedEmbeddingBagsCodegen,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
    # TODO: move logic to FBGEMM to avoid accessing fbgemm internals
    for t_idx, (rows, dim, location, _) in enumerate(emb_module.embedding_specs):
        table_name = config.embedding_tables[t_idx].name
        if table_name not in table_name_to_count:
            continue
        table_count = table_name_to_count.pop(table_name)
        if emb_module.weights_precision == SparseType.INT8:
            dim += emb_module.int8_emb_row_dim_offset
        # pyre-ignore[29]
        offset = emb_module.weights_physical_offsets[t_idx]
        weights: torch.Tensor
        if location == EmbeddingLocation.DEVICE.value:
            # pyre-ignore
            weights = emb_module.weights_dev
        elif location == EmbeddingLocation.HOST.value:
            # pyre-ignore
            weights = emb_module.weights_host
        else:
            # pyre-ignore
            weights = emb_module.weights_uvm
        weight = TableBatchedEmbeddingSlice(
            data=weights,
            start_offset=offset,
            end_offset=offset + table_count * rows * dim,
            num_embeddings=-1,
            embedding_dim=dim,
        )
        # this reuses logic in EmbeddingFusedOptimizer but is per table
        # pyre-ignore
        weight._in_backward_optimizers = [
            EmbeddingFusedOptimizer(
                config,
                emb_module,
                create_for_table=table_name,
                param_weight_for_table=weight,
            )
        ]
        yield (table_name, weight)


def _gen_named_parameters_by_table_dense(
    emb_module: DenseTableBatchedEmbeddingBagsCodegen,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
    # TODO: move logic to FBGEMM to avoid accessing fbgemm internals
    for t_idx, (rows, dim) in enumerate(emb_module.embedding_specs):
        table_name = config.embedding_tables[t_idx].name
        if table_name not in table_name_to_count:
            continue
        table_count = table_name_to_count.pop(table_name)
        offset = emb_module.weights_physical_offsets[t_idx]
        weight = TableBatchedEmbeddingSlice(
            data=emb_module.weights,
            start_offset=offset,
            end_offset=offset + table_count * rows * dim,
            num_embeddings=-1,
            embedding_dim=dim,
        )
        yield (table_name, weight)


class BaseBatchedEmbedding(BaseEmbedding):
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

        self._local_rows: List[int] = []
        self._weight_init_mins: List[float] = []
        self._weight_init_maxs: List[float] = []
        self._num_embeddings: List[int] = []
        self._local_cols: List[int] = []
        self._feature_table_map: List[int] = []
        self.table_name_to_count: Dict[str, int] = {}
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = {}

        for idx, config in enumerate(self._config.embedding_tables):
            self._local_rows.append(config.local_rows)
            self._weight_init_mins.append(config.get_weight_init_min())
            self._weight_init_maxs.append(config.get_weight_init_max())
            self._num_embeddings.append(config.num_embeddings)
            self._local_cols.append(config.local_cols)
            self._feature_table_map.extend([idx] * config.num_features())
            if config.name not in self.table_name_to_count:
                self.table_name_to_count[config.name] = 0
            self.table_name_to_count[config.name] += 1

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

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        return self.emb_module(
            indices=features.values().long(),
            offsets=features.offsets().long(),
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        self.flush()
        return get_state_dict(
            self._config.embedding_tables,
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

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

    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config

    def flush(self) -> None:
        pass

    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, param in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, param

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for name, param in self._param_per_table.items():
            yield name, param


class BatchedFusedEmbedding(BaseBatchedEmbedding, FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        compute_devices: List[ComputeDevice] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)

        weights_precision = data_type_to_sparse_type(config.data_type)

        fused_params = config.fused_params or {}
        if "cache_precision" not in fused_params:
            fused_params["cache_precision"] = weights_precision

        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=list(
                    zip(self._local_rows, self._local_cols, managed, compute_devices)
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=PoolingMode.NONE,
                weights_precision=weights_precision,
                device=device,
                **fused_params,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_fused(
                self._emb_module,
                self.table_name_to_count.copy(),
                self._config,
            )
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

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        # TODO can delete this override once SEA is removed
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after SEA deprecation
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()


class BatchedDenseEmbedding(BaseBatchedEmbedding):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        weights_precision = data_type_to_sparse_type(config.data_type)
        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._local_cols)),
                feature_table_map=self._feature_table_map,
                pooling_mode=PoolingMode.NONE,
                use_cpu=device is None
                or device.type == "cpu"
                or not torch.cuda.is_available(),
                weights_precision=weights_precision,
            )
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_dense(
                self._emb_module, self.table_name_to_count.copy(), self._config
            )
        )
        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> DenseTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        combined_key = "/".join(
            [config.name for config in self._config.embedding_tables]
        )
        yield append_prefix(prefix, f"{combined_key}.weight"), cast(
            nn.Parameter, self._emb_module.weights
        )


class BaseBatchedEmbeddingBag(BaseEmbedding):
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

        self._pooling: PoolingMode = pooling_type_to_pooling_mode(config.pooling)

        self._local_rows: List[int] = []
        self._weight_init_mins: List[float] = []
        self._weight_init_maxs: List[float] = []
        self._num_embeddings: List[int] = []
        self._local_cols: List[int] = []
        self._feature_table_map: List[int] = []
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []
        self.table_name_to_count: Dict[str, int] = {}
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = {}

        for idx, config in enumerate(self._config.embedding_tables):
            self._local_rows.append(config.local_rows)
            self._weight_init_mins.append(config.get_weight_init_min())
            self._weight_init_maxs.append(config.get_weight_init_max())
            self._num_embeddings.append(config.num_embeddings)
            self._local_cols.append(config.local_cols)
            self._feature_table_map.extend([idx] * config.num_features())
            if config.name not in self.table_name_to_count:
                self.table_name_to_count[config.name] = 0
            self.table_name_to_count[config.name] += 1

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

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        weights = features.weights_or_none()
        if weights is not None and not torch.is_floating_point(weights):
            weights = None
        return self.emb_module(
            indices=features.values().long(),
            offsets=features.offsets().long(),
            per_sample_weights=weights,
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        self.flush()
        return get_state_dict(
            self._config.embedding_tables,
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

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

    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config

    def flush(self) -> None:
        pass

    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for name, param in self._param_per_table.items():
            yield name, param


class BatchedFusedEmbeddingBag(BaseBatchedEmbeddingBag, FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        compute_devices: List[ComputeDevice] = []
        for table in config.embedding_tables:
            assert table.local_cols % 4 == 0, (
                f"table {table.name} has local_cols={table.local_cols} "
                "not divisible by 4. "
            )
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        if "cache_precision" not in fused_params:
            fused_params["cache_precision"] = weights_precision

        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=list(
                    zip(self._local_rows, self._local_cols, managed, compute_devices)
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                weights_precision=weights_precision,
                device=device,
                **fused_params,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_fused(
                self._emb_module,
                self.table_name_to_count.copy(),
                self._config,
            )
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

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        # TODO can delete this override once SEA is removed
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()


class BatchedDenseEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        weights_precision = data_type_to_sparse_type(config.data_type)
        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._local_cols)),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                use_cpu=device is None
                or device.type == "cpu"
                or not torch.cuda.is_available(),
                weights_precision=weights_precision,
            )
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_dense(
                self._emb_module, self.table_name_to_count.copy(), self._config
            )
        )
        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> DenseTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        combined_key = "/".join(
            [config.name for config in self._config.embedding_tables]
        )
        yield append_prefix(prefix, f"{combined_key}.weight"), cast(
            nn.Parameter, self._emb_module.weights
        )
