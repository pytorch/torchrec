#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import copy
import inspect
import itertools
import logging
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
    PoolingMode,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.ssd import ASSOC, SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.ssd.utils.partially_materialized_tensor import (
    PartiallyMaterializedTensor,
)
from torch import nn
from torch.distributed._tensor import DTensor, Replicate, Shard as DTensorShard
from torchrec.distributed.comm import get_local_rank, get_node_group_size
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_kernel import BaseEmbedding, get_state_dict
from torchrec.distributed.embedding_types import (
    compute_kernel_to_embedding_location,
    DTensorMetadata,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.types import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardingType,
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

logger: logging.Logger = logging.getLogger(__name__)


def _populate_ssd_tbe_params(config: GroupedEmbeddingConfig) -> Dict[str, Any]:
    """
    Construct SSD TBE params dict from config and fused params dict.
    """
    fused_params = config.fused_params or {}

    ssd_tbe_params: Dict[str, Any] = {}

    # drop the non-ssd tbe fused params
    ssd_tbe_signature = inspect.signature(
        SSDTableBatchedEmbeddingBags.__init__
    ).parameters.keys()
    invalid_keys: List[str] = []

    for key, value in fused_params.items():
        if key not in ssd_tbe_signature:
            invalid_keys.append(key)
        else:
            ssd_tbe_params[key] = value
    if len(invalid_keys) > 0:
        logger.warning(
            f"Dropping {invalid_keys} since they are not valid SSD TBE params."
        )

    # populate number cache sets, aka number of rows of the cache space
    if "cache_sets" not in ssd_tbe_params:
        cache_load_factor = fused_params.get("cache_load_factor")
        if cache_load_factor:
            cache_load_factor = fused_params.get("cache_load_factor")
            logger.info(
                f"Using cache load factor from fused params dict: {cache_load_factor}"
            )
        else:
            cache_load_factor = 0.2

        local_rows_sum: int = sum(table.local_rows for table in config.embedding_tables)
        ssd_tbe_params["cache_sets"] = max(
            int(cache_load_factor * local_rows_sum / ASSOC), 1
        )

    # populate init min and max
    if (
        "ssd_uniform_init_lower" not in ssd_tbe_params
        or "ssd_uniform_init_upper" not in ssd_tbe_params
    ):
        # Right now we do not support a per table init max and min. To use
        # per table init max and min, either we allow it in SSD TBE, or we
        # create one SSD TBE per table.
        # TODO: Solve the init problem
        mins = [table.get_weight_init_min() for table in config.embedding_tables]
        maxs = [table.get_weight_init_max() for table in config.embedding_tables]
        ssd_tbe_params["ssd_uniform_init_lower"] = sum(mins) / len(
            config.embedding_tables
        )
        ssd_tbe_params["ssd_uniform_init_upper"] = sum(maxs) / len(
            config.embedding_tables
        )

    if "ssd_storage_directory" not in ssd_tbe_params:
        ssd_tbe_params["ssd_storage_directory"] = tempfile.mkdtemp()
    else:
        if "@local_rank" in ssd_tbe_params["ssd_storage_directory"]:
            # assume we have initialized a process group already
            ssd_tbe_params["ssd_storage_directory"] = ssd_tbe_params[
                "ssd_storage_directory"
            ].replace("@local_rank", str(get_local_rank()))

    if "weights_precision" not in ssd_tbe_params:
        weights_precision = data_type_to_sparse_type(config.data_type)
        ssd_tbe_params["weights_precision"] = weights_precision

    if "max_l1_cache_size" in fused_params:
        l1_cache_size = fused_params.get("max_l1_cache_size") * 1024 * 1024
        max_dim: int = max(table.local_cols for table in config.embedding_tables)
        weight_precision_bytes = ssd_tbe_params["weights_precision"].bit_rate() / 8
        max_cache_sets = (
            l1_cache_size / ASSOC / weight_precision_bytes / max_dim
        )  # 100MB

        if ssd_tbe_params["cache_sets"] > int(max_cache_sets):
            logger.warning(
                f"cache_sets {ssd_tbe_params['cache_sets']} is larger than max_cache_sets {max_cache_sets} calculated "
                "by max_l1_cache_size, cap at max_cache_sets instead"
            )
            ssd_tbe_params["cache_sets"] = int(max_cache_sets)

    return ssd_tbe_params


class KeyValueEmbeddingFusedOptimizer(FusedOptimizer):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        emb_module: SSDTableBatchedEmbeddingBags,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Fused optimizer for SSD TBE. Right now it only supports tuning learning
        rate.
        """
        self._emb_module: SSDTableBatchedEmbeddingBags = emb_module
        self._pg = pg

        # TODO: support optimizer states checkpointing once FBGEMM support
        # split_optimizer_states API

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.optimizer_args.learning_rate,
        }

        params: Dict[str, Union[torch.Tensor, ShardedTensor]] = {}

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])


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
            dtensor_metadata: List[DTensorMetadata]

        def get_optimizer_single_value_shard_metadata_and_global_metadata(
            table_global_metadata: ShardedTensorMetadata,
            optimizer_state: torch.Tensor,
        ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
            table_global_shards_metadata: List[ShardMetadata] = (
                table_global_metadata.shards_metadata
            )

            table_shard_metadata_to_optimizer_shard_metadata = {}
            for offset, table_shard_metadata in enumerate(table_global_shards_metadata):
                table_shard_metadata_to_optimizer_shard_metadata[
                    table_shard_metadata
                ] = ShardMetadata(
                    shard_sizes=[1],  # single value optimizer state
                    shard_offsets=[offset],  # offset increases by 1 for each shard
                    placement=table_shard_metadata.placement,
                )

            tensor_properties = TensorProperties(
                dtype=optimizer_state.dtype,
                layout=optimizer_state.layout,
                requires_grad=False,
            )
            single_value_optimizer_st_metadata = ShardedTensorMetadata(
                shards_metadata=list(
                    table_shard_metadata_to_optimizer_shard_metadata.values()
                ),
                size=torch.Size([len(table_global_shards_metadata)]),
                tensor_properties=tensor_properties,
            )

            return (
                table_shard_metadata_to_optimizer_shard_metadata,
                single_value_optimizer_st_metadata,
            )

        def get_optimizer_rowwise_shard_metadata_and_global_metadata(
            table_global_metadata: ShardedTensorMetadata,
            optimizer_state: torch.Tensor,
            sharding_dim: int,
            is_grid_sharded: bool = False,
        ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
            table_global_shards_metadata: List[ShardMetadata] = (
                table_global_metadata.shards_metadata
            )

            if sharding_dim == 1:
                # column-wise sharding
                # sort the metadata based on column offset and
                # we construct the momentum tensor in row-wise sharded way
                table_global_shards_metadata = sorted(
                    table_global_shards_metadata,
                    key=lambda shard: shard.shard_offsets[1],
                )

            table_shard_metadata_to_optimizer_shard_metadata = {}
            rolling_offset = 0
            for idx, table_shard_metadata in enumerate(table_global_shards_metadata):
                offset = table_shard_metadata.shard_offsets[0]

                if is_grid_sharded:
                    # we use a rolling offset to calculate the current offset for shard to account for uneven row wise case for our shards
                    offset = rolling_offset
                    rolling_offset += table_shard_metadata.shard_sizes[0]
                elif sharding_dim == 1:
                    # for column-wise sharding, we still create row-wise sharded metadata for optimizer
                    # manually create a row-wise offset
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
                if sharding_dim == 1 and not is_grid_sharded
                else 1
            )
            # for grid sharding, the row dimension is replicated CW shard times
            grid_shard_nodes = (
                len(table_global_shards_metadata) // get_node_group_size()
                if is_grid_sharded
                else 1
            )
            rowwise_optimizer_st_metadata = ShardedTensorMetadata(
                shards_metadata=list(
                    table_shard_metadata_to_optimizer_shard_metadata.values()
                ),
                size=torch.Size(
                    [table_global_metadata.size[0] * len_rw_shards * grid_shard_nodes]
                ),
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
            table_global_shards_metadata: List[ShardMetadata] = (
                table_global_metadata.shards_metadata
            )

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

        embedding_weights_by_table = emb_module.split_embedding_weights()

        all_optimizer_states = emb_module.get_optimizer_state()
        optimizer_states_keys_by_table: Dict[str, List[torch.Tensor]] = {}
        for (
            table_config,
            optimizer_states,
            weight,
        ) in itertools.zip_longest(
            config.embedding_tables,
            all_optimizer_states,
            embedding_weights_by_table,
        ):
            # When EmbeddingFusedOptimizer is created for composability, only create state
            if create_for_table is not None and create_for_table != table_config.name:
                continue
            if table_config.name not in table_to_shard_params:
                table_to_shard_params[table_config.name] = ShardParams(
                    optimizer_states=[],
                    local_metadata=[],
                    embedding_weights=[],
                    dtensor_metadata=[],
                )
            optimizer_state_values = None
            if optimizer_states:
                optimizer_state_values = tuple(optimizer_states.values())
                for optimizer_state_value in optimizer_state_values:
                    assert (
                        table_config.local_rows == optimizer_state_value.size(0)
                        or optimizer_state_value.nelement() == 1  # single value state
                    )
                optimizer_states_keys_by_table[table_config.name] = list(
                    optimizer_states.keys()
                )
            local_metadata = table_config.local_metadata

            table_to_shard_params[table_config.name].optimizer_states.append(
                optimizer_state_values
            )
            table_to_shard_params[table_config.name].local_metadata.append(
                local_metadata
            )
            table_to_shard_params[table_config.name].dtensor_metadata.append(
                table_config.dtensor_metadata
            )
            table_to_shard_params[table_config.name].embedding_weights.append(weight)

        seen_tables = set()
        for table_config in config.embedding_tables:
            if create_for_table is not None and create_for_table != table_config.name:
                continue
            if table_config.name in seen_tables:
                continue
            seen_tables.add(table_config.name)
            table_config_global_metadata: Optional[ShardedTensorMetadata] = (
                copy.deepcopy(table_config.global_metadata)
            )

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

            is_grid_sharded: bool = (
                True
                if table_config.local_cols != table_config.embedding_dim
                and table_config.local_rows != table_config.num_embeddings
                else False
            )

            if all(
                opt_state is not None for opt_state in shard_params.optimizer_states
            ):
                # pyre-ignore
                def get_sharded_optim_state(
                    momentum_idx: int, state_key: str
                ) -> Union[ShardedTensor, DTensor]:
                    assert momentum_idx > 0
                    momentum_local_shards: List[Shard] = []
                    optimizer_sharded_tensor_metadata: ShardedTensorMetadata

                    # pyre-ignore [16]
                    optim_state = shard_params.optimizer_states[0][momentum_idx - 1]
                    if (
                        optim_state.nelement() == 1 and state_key != "momentum1"
                    ):  # special handling for backward compatibility, momentum1 is rowwise state for rowwise_adagrad
                        # single value state: one value per table
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_single_value_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            optim_state,
                        )
                    elif optim_state.dim() == 1:
                        # rowwise state: param.shape[0] == state.shape[0], state.shape[1] == 1
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_rowwise_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            optim_state,
                            sharding_dim,
                            is_grid_sharded,
                        )
                    else:
                        # pointwise state: param.shape == state.shape
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_pointwise_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            optim_state,
                        )

                    for optimizer_state, table_shard_local_metadata in zip(
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

                    # Convert optimizer state to DTensor if enabled
                    if table_config.dtensor_metadata:
                        # if rowwise state we do Shard(0), regardless of how the table is sharded
                        if optim_state.dim() == 1:
                            stride = (1,)
                            placements = (
                                (Replicate(), DTensorShard(0))
                                if table_config.dtensor_metadata.mesh.ndim == 2
                                else (DTensorShard(0),)
                            )
                        else:
                            stride = table_config.dtensor_metadata.stride
                            placements = table_config.dtensor_metadata.placements

                        return DTensor.from_local(
                            local_tensor=LocalShardsWrapper(
                                local_shards=[x.tensor for x in momentum_local_shards],
                                local_offsets=[  # pyre-ignore[6]
                                    x.metadata.shard_offsets
                                    for x in momentum_local_shards
                                ],
                            ),
                            device_mesh=table_config.dtensor_metadata.mesh,
                            placements=placements,
                            shape=optimizer_sharded_tensor_metadata.size,
                            stride=stride,
                            run_check=False,
                        )
                    else:
                        # TODO we should be creating this in SPMD fashion (e.g. init_from_local_shards), and let it derive global metadata.
                        return ShardedTensor._init_from_local_shards_and_global_metadata(
                            local_shards=momentum_local_shards,
                            sharded_tensor_metadata=optimizer_sharded_tensor_metadata,
                            process_group=self._pg,
                        )

                num_states: int = min(
                    # pyre-ignore
                    [len(opt_state) for opt_state in shard_params.optimizer_states]
                )
                optimizer_state_keys = []
                if num_states > 0:
                    optimizer_state_keys = optimizer_states_keys_by_table[
                        table_config.name
                    ]
                for cur_state_idx in range(0, num_states):
                    if cur_state_idx == 0:
                        # for backward compatibility
                        cur_state_key = "momentum1"
                    else:
                        cur_state_key = optimizer_state_keys[cur_state_idx]

                    state[weight][f"{table_config.name}.{cur_state_key}"] = (
                        get_sharded_optim_state(cur_state_idx + 1, cur_state_key)
                    )

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    def set_optimizer_step(self, step: int) -> None:
        self._emb_module.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        self._emb_module.update_hyper_parameters(params_dict)


def _gen_named_parameters_by_table_ssd(
    emb_module: SSDTableBatchedEmbeddingBags,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
    pg: Optional[dist.ProcessGroup] = None,
) -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Return an empty tensor to indicate that the table is on remote device.
    """
    for table in config.embedding_tables:
        table_name = table.name
        # placeholder
        weight: nn.Parameter = nn.Parameter(torch.empty(0))
        # pyre-ignore
        weight._in_backward_optimizers = [EmptyFusedOptimizer()]
        yield (table_name, weight)


def _gen_named_parameters_by_table_ssd_pmt(
    emb_module: SSDTableBatchedEmbeddingBags,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
    pg: Optional[dist.ProcessGroup] = None,
) -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Return an iterator over module parameters that are embedding tables, yielding both the table
    name as well as the parameter itself. The embedding table is in the form of
    PartiallyMaterializedTensor to support windowed access.
    """
    pmts = emb_module.split_embedding_weights()
    for table_config, pmt in zip(config.embedding_tables, pmts):
        table_name = table_config.name
        emb_table = pmt
        weight: nn.Parameter = nn.Parameter(emb_table)
        # pyre-ignore
        weight._in_backward_optimizers = [EmptyFusedOptimizer()]
        yield (table_name, weight)


def _gen_named_parameters_by_table_fused(
    emb_module: SplitTableBatchedEmbeddingBagsCodegen,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
    pg: Optional[dist.ProcessGroup] = None,
) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
    # TODO: move logic to FBGEMM to avoid accessing fbgemm internals
    for t_idx, (rows, dim, location, _) in enumerate(emb_module.embedding_specs):
        table_name = config.embedding_tables[t_idx].name
        if table_name not in table_name_to_count:
            continue
        table_count = table_name_to_count.pop(table_name)
        if emb_module.weights_precision == SparseType.INT8:
            dim += emb_module.int8_emb_row_dim_offset
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        offset = emb_module.weights_physical_offsets[t_idx]
        weights: torch.Tensor
        if location == EmbeddingLocation.DEVICE.value:
            # pyre-fixme[9]: weights has type `Tensor`; used as `Union[Module, Tensor]`.
            weights = emb_module.weights_dev
        elif location == EmbeddingLocation.HOST.value:
            # pyre-fixme[9]: weights has type `Tensor`; used as `Union[Module, Tensor]`.
            weights = emb_module.weights_host
        else:
            # pyre-fixme[9]: weights has type `Tensor`; used as `Union[Module, Tensor]`.
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
                config=config,
                emb_module=emb_module,
                pg=pg,
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


SplitWeightType = TypeVar("SplitWeightType")


class BaseBatchedEmbedding(BaseEmbedding, Generic[SplitWeightType]):
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

        # pyre-fixme[9]: config has type `GroupedEmbeddingConfig`; used as
        #  `ShardedEmbeddingTable`.
        for idx, config in enumerate(self._config.embedding_tables):
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute `local_rows`.
            self._local_rows.append(config.local_rows)
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute
            #  `get_weight_init_min`.
            self._weight_init_mins.append(config.get_weight_init_min())
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute
            #  `get_weight_init_max`.
            self._weight_init_maxs.append(config.get_weight_init_max())
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute
            #  `num_embeddings`.
            self._num_embeddings.append(config.num_embeddings)
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute `local_cols`.
            self._local_cols.append(config.local_cols)
            self._feature_table_map.extend([idx] * config.num_features())
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute `name`.
            if config.name not in self.table_name_to_count:
                self.table_name_to_count[config.name] = 0
            self.table_name_to_count[config.name] += 1

    def init_parameters(self) -> None:
        # initialize embedding weights
        assert len(self._num_embeddings) == len(self.split_embedding_weights())
        for rows, emb_dim, weight_init_min, weight_init_max, param in zip(
            self._local_rows,
            self._local_cols,
            self._weight_init_mins,
            self._weight_init_maxs,
            self.split_embedding_weights(),
        ):
            assert param.shape == (rows, emb_dim)  # pyre-ignore[16]
            param.data.uniform_(  # pyre-ignore[16]
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
            # pyre-ignore
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

    def split_embedding_weights(self) -> List[SplitWeightType]:
        return self.emb_module.split_embedding_weights()

    @property
    @abc.abstractmethod
    def emb_module(
        self,
    ) -> Union[
        DenseTableBatchedEmbeddingBagsCodegen,
        SplitTableBatchedEmbeddingBagsCodegen,
        IntNBitTableBatchedEmbeddingBagsCodegen,
    ]: ...

    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config

    def flush(self) -> None:
        pass

    def purge(self) -> None:
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


class KeyValueEmbedding(BaseBatchedEmbedding[torch.Tensor], FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        assert (
            len(config.embedding_tables) > 0
        ), "Expected to see at least one table in SSD TBE, but found 0."
        assert (
            len({table.embedding_dim for table in config.embedding_tables}) == 1
        ), "Currently we expect all tables in SSD TBE to have the same embedding dimension."

        ssd_tbe_params = _populate_ssd_tbe_params(config)
        compute_kernel = config.embedding_tables[0].compute_kernel
        embedding_location = compute_kernel_to_embedding_location(compute_kernel)

        self._emb_module: SSDTableBatchedEmbeddingBags = SSDTableBatchedEmbeddingBags(
            embedding_specs=list(zip(self._local_rows, self._local_cols)),
            feature_table_map=self._feature_table_map,
            ssd_cache_location=embedding_location,
            pooling_mode=PoolingMode.NONE,
            **ssd_tbe_params,
        ).to(device)

        self._optim: KeyValueEmbeddingFusedOptimizer = KeyValueEmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, nn.Parameter] = dict(
            _gen_named_parameters_by_table_ssd_pmt(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        An advantage of SSD TBE is that we don't need to init weights. Hence skipping.
        """
        pass

    @property
    def emb_module(
        self,
    ) -> SSDTableBatchedEmbeddingBags:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        """
        SSD Embedding fuses backward with backward.
        """
        return self._optim

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            no_snapshot (bool): the tensors in the returned dict are
                PartiallyMaterializedTensors. this argument controls wether the
                PartiallyMaterializedTensor owns a RocksDB snapshot handle. True means the
                PartiallyMaterializedTensor doesn't have a RocksDB snapshot handle.  False means the
                PartiallyMaterializedTensor has a RocksDB snapshot handle
        """
        # in the case no_snapshot=False, a flush is required. we rely on the flush operation in
        # ShardedEmbeddingBagCollection._pre_state_dict_hook()

        emb_tables = self.split_embedding_weights(no_snapshot=no_snapshot)
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")
        ret = get_state_dict(
            emb_table_config_copy,
            emb_tables,
            self._pg,
            destination,
            prefix,
        )
        return ret

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Only allowed ways to get state_dict.
        """
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            # pyre-ignore [6]
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    # pyre-ignore [15]
    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, PartiallyMaterializedTensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights(),
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    def get_named_split_embedding_weights_snapshot(
        self, prefix: str = ""
    ) -> Iterator[Tuple[str, PartiallyMaterializedTensor]]:
        """
        Return an iterator over embedding tables, yielding both the table name as well as the embedding
        table itself. The embedding table is in the form of PartiallyMaterializedTensor with a valid
        RocksDB snapshot to support windowed access.
        """
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights(no_snapshot=False),
        ):
            key = append_prefix(prefix, f"{config.name}")
            yield key, tensor

    def flush(self) -> None:
        """
        Flush the embeddings in cache back to SSD. Should be pretty expensive.
        """
        self.emb_module.flush()

    def purge(self) -> None:
        """
        Reset the cache space. This is needed when we load state dict.
        """
        # TODO: move the following to SSD TBE.
        self.emb_module.lxu_cache_weights.zero_()
        self.emb_module.lxu_cache_state.fill_(-1)

    # pyre-ignore [15]
    def split_embedding_weights(
        self, no_snapshot: bool = True
    ) -> List[PartiallyMaterializedTensor]:
        return self.emb_module.split_embedding_weights(no_snapshot)


class BatchedFusedEmbedding(BaseBatchedEmbedding[torch.Tensor], FusedOptimizerModule):
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
            elif device is not None and device.type == "mtia":
                compute_devices.append(ComputeDevice.MTIA)
                # Set EmbeddingLocation.HOST to make embedding op in FBGEMM choose CPU path.
                # But the tensor will still be created on MTIA with device type "mtia".
                managed.append(EmbeddingLocation.HOST)
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
                table_names=[t.name for t in config.embedding_tables],
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
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
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

    def purge(self) -> None:
        self._emb_module.reset_cache_states()


class BatchedDenseEmbedding(BaseBatchedEmbedding[torch.Tensor]):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        output_dtype = fused_params.get("output_dtype", SparseType.FP32)
        use_cpu: bool = (
            device is None
            or device.type == "cpu"
            or (not (torch.cuda.is_available() or torch.mtia.is_available()))
        )
        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._local_cols)),
                feature_table_map=self._feature_table_map,
                pooling_mode=PoolingMode.NONE,
                use_cpu=use_cpu,
                weights_precision=weights_precision,
                output_dtype=output_dtype,
                use_mtia=device is not None and device.type == "mtia",
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


class BaseBatchedEmbeddingBag(BaseEmbedding, Generic[SplitWeightType]):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        self._pg = pg

        self._pooling: PoolingMode = pooling_type_to_pooling_mode(
            config.pooling, sharding_type  # pyre-ignore[6]
        )

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

        # pyre-fixme[9]: config has type `GroupedEmbeddingConfig`; used as
        #  `ShardedEmbeddingTable`.
        for idx, config in enumerate(self._config.embedding_tables):
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute `local_rows`.
            self._local_rows.append(config.local_rows)
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute
            #  `get_weight_init_min`.
            self._weight_init_mins.append(config.get_weight_init_min())
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute
            #  `get_weight_init_max`.
            self._weight_init_maxs.append(config.get_weight_init_max())
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute
            #  `num_embeddings`.
            self._num_embeddings.append(config.num_embeddings)
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute `local_cols`.
            self._local_cols.append(config.local_cols)
            self._feature_table_map.extend([idx] * config.num_features())
            # pyre-fixme[16]: `GroupedEmbeddingConfig` has no attribute `name`.
            if config.name not in self.table_name_to_count:
                self.table_name_to_count[config.name] = 0
            self.table_name_to_count[config.name] += 1

    def init_parameters(self) -> None:
        # initialize embedding weights
        assert len(self._num_embeddings) == len(self.split_embedding_weights())
        for rows, emb_dim, weight_init_min, weight_init_max, param in zip(
            self._local_rows,
            self._local_cols,
            self._weight_init_mins,
            self._weight_init_maxs,
            self.split_embedding_weights(),
        ):
            assert param.shape == (rows, emb_dim)  # pyre-ignore[16]
            param.data.uniform_(  # pyre-ignore[16]
                weight_init_min,
                weight_init_max,
            )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        weights = features.weights_or_none()
        if weights is not None and not torch.is_floating_point(weights):
            weights = None
        if features.variable_stride_per_key() and isinstance(
            self.emb_module,
            (
                SplitTableBatchedEmbeddingBagsCodegen,
                DenseTableBatchedEmbeddingBagsCodegen,
                SSDTableBatchedEmbeddingBags,
            ),
        ):
            return self.emb_module(
                indices=features.values().long(),
                offsets=features.offsets().long(),
                per_sample_weights=weights,
                batch_size_per_feature_per_rank=features.stride_per_key_per_rank(),
            )
        else:
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
            # pyre-ignore
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

    def split_embedding_weights(self) -> List[SplitWeightType]:
        return self.emb_module.split_embedding_weights()

    @property
    @abc.abstractmethod
    def emb_module(
        self,
    ) -> Union[
        DenseTableBatchedEmbeddingBagsCodegen,
        SplitTableBatchedEmbeddingBagsCodegen,
        IntNBitTableBatchedEmbeddingBagsCodegen,
    ]: ...

    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config

    def flush(self) -> None:
        pass

    def purge(self) -> None:
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


class KeyValueEmbeddingBag(BaseBatchedEmbeddingBag[torch.Tensor], FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

        assert (
            len(config.embedding_tables) > 0
        ), "Expected to see at least one table in SSD TBE, but found 0."
        assert (
            len({table.embedding_dim for table in config.embedding_tables}) == 1
        ), "Currently we expect all tables in SSD TBE to have the same embedding dimension."

        ssd_tbe_params = _populate_ssd_tbe_params(config)
        compute_kernel = config.embedding_tables[0].compute_kernel
        embedding_location = compute_kernel_to_embedding_location(compute_kernel)

        self._emb_module: SSDTableBatchedEmbeddingBags = SSDTableBatchedEmbeddingBags(
            embedding_specs=list(zip(self._local_rows, self._local_cols)),
            feature_table_map=self._feature_table_map,
            ssd_cache_location=embedding_location,
            pooling_mode=self._pooling,
            **ssd_tbe_params,
        ).to(device)

        logger.info(
            f"tbe_unique_id:{self._emb_module.tbe_unique_id} => table name to count dict:{self.table_name_to_count}"
        )

        self._optim: KeyValueEmbeddingFusedOptimizer = KeyValueEmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, nn.Parameter] = dict(
            _gen_named_parameters_by_table_ssd_pmt(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        An advantage of SSD TBE is that we don't need to init weights. Hence
        skipping.
        """
        pass

    @property
    def emb_module(
        self,
    ) -> SSDTableBatchedEmbeddingBags:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        """
        SSD Embedding fuses backward with backward.
        """
        return self._optim

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            no_snapshot (bool): the tensors in the returned dict are
                PartiallyMaterializedTensors. this argument controls wether the
                PartiallyMaterializedTensor owns a RocksDB snapshot handle. True means the
                PartiallyMaterializedTensor doesn't have a RocksDB snapshot handle.  False means the
                PartiallyMaterializedTensor has a RocksDB snapshot handle
        """
        # in the case no_snapshot=False, a flush is required. we rely on the flush operation in
        # ShardedEmbeddingBagCollection._pre_state_dict_hook()

        emb_tables = self.split_embedding_weights(no_snapshot=no_snapshot)
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")
        ret = get_state_dict(
            emb_table_config_copy,
            emb_tables,
            self._pg,
            destination,
            prefix,
        )
        return ret

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Only allowed ways to get state_dict.
        """
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            # pyre-ignore [6]
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    # pyre-ignore [15]
    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, PartiallyMaterializedTensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights(),
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    def get_named_split_embedding_weights_snapshot(
        self, prefix: str = ""
    ) -> Iterator[Tuple[str, PartiallyMaterializedTensor]]:
        """
        Return an iterator over embedding tables, yielding both the table name as well as the embedding
        table itself. The embedding table is in the form of PartiallyMaterializedTensor with a valid
        RocksDB snapshot to support windowed access.
        """
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights(no_snapshot=False),
        ):
            key = append_prefix(prefix, f"{config.name}")
            yield key, tensor

    def flush(self) -> None:
        """
        Flush the embeddings in cache back to SSD. Should be pretty expensive.
        """
        self.emb_module.flush()

    def purge(self) -> None:
        """
        Reset the cache space. This is needed when we load state dict.
        """
        # TODO: move the following to SSD TBE.
        self.emb_module.lxu_cache_weights.zero_()
        self.emb_module.lxu_cache_state.fill_(-1)

    # pyre-ignore [15]
    def split_embedding_weights(
        self, no_snapshot: bool = True
    ) -> List[PartiallyMaterializedTensor]:
        return self.emb_module.split_embedding_weights(no_snapshot)


class BatchedFusedEmbeddingBag(
    BaseBatchedEmbeddingBag[torch.Tensor], FusedOptimizerModule
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

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
            elif device is not None and device.type == "mtia":
                compute_devices.append(ComputeDevice.MTIA)
                # Set EmbeddingLocation.HOST to make embedding op in FBGEMM choose CPU path.
                # But the tensor will still be created on MTIA with device type "mtia".
                managed.append(EmbeddingLocation.HOST)
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
                table_names=[t.name for t in config.embedding_tables],
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
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
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

    def purge(self) -> None:
        self._emb_module.reset_cache_states()


class BatchedDenseEmbeddingBag(BaseBatchedEmbeddingBag[torch.Tensor]):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        output_dtype = fused_params.get("output_dtype", SparseType.FP32)
        use_cpu: bool = (
            device is None
            or device.type == "cpu"
            or (not (torch.cuda.is_available() or torch.mtia.is_available()))
        )
        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._local_cols)),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                use_cpu=use_cpu,
                weights_precision=weights_precision,
                output_dtype=output_dtype,
                use_mtia=device is not None and device.type == "mtia",
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
