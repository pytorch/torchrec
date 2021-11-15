#!/usr/bin/env python3

# pyre-ignore-all-errors[56]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import logging
from dataclasses import dataclass
from itertools import accumulate
from math import log2
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import fbgemm_gpu.split_embedding_codegen_lookup_invokers as invokers
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_embedding_configs import SparseType
from torch import Tensor, nn

ASSOC = 32
# Maximum number of times prefetch() can be called without
# a corresponding forward() call
MAX_PREFETCH_DEPTH = 100
INT8_EMB_ROW_DIM_OFFSET = 8


class DoesNotHavePrefix(Exception):
    pass


class EmbeddingLocation(enum.IntEnum):
    DEVICE = 0
    MANAGED = 1
    MANAGED_CACHING = 2
    HOST = 3


class ComputeDevice(enum.IntEnum):
    CPU = 0
    CUDA = 1


class CacheAlgorithm(enum.Enum):
    LRU = 0
    LFU = 1


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2


class BoundsCheckMode(enum.IntEnum):
    # Raise an exception (CPU) or device-side assert (CUDA)
    FATAL = 0
    # Log the first out-of-bounds instance per kernel, and set to zero.
    WARNING = 1
    # Set to zero.
    IGNORE = 2
    # No bounds checks.
    NONE = 3


RecordCacheMetrics: NamedTuple = NamedTuple(
    "RecordCacheMetrics",
    [("record_cache_miss_counter", bool), ("record_tablewise_cache_miss", bool)],
)


@dataclass
class SplitState:
    dev_size: int
    host_size: int
    uvm_size: int
    placements: List[EmbeddingLocation]
    offsets: List[int]


def construct_split_state(
    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]],
    rowwise: bool,
    cacheable: bool,
    precision: SparseType = SparseType.FP32,
    int8_emb_row_dim_offset: int = INT8_EMB_ROW_DIM_OFFSET,
) -> SplitState:
    placements = []
    offsets = []
    dev_size = 0
    host_size = 0
    uvm_size = 0
    for (num_embeddings, embedding_dim, location, _) in embedding_specs:
        assert embedding_dim % 4 == 0, f"{embedding_dim}"
        if precision == SparseType.INT8:
            embedding_dim += int8_emb_row_dim_offset
        state_size = num_embeddings * embedding_dim if not rowwise else num_embeddings
        if location == EmbeddingLocation.HOST:
            placements.append(EmbeddingLocation.HOST)
            offsets.append(host_size)
            host_size += state_size
        # If table is on device, then opimtizer is on device.
        # If table is managed, then if optimizer state is rowwise, optimizer is on device, otherwise optimizer is managed.
        elif location == EmbeddingLocation.DEVICE or rowwise:
            placements.append(EmbeddingLocation.DEVICE)
            offsets.append(dev_size)
            dev_size += state_size
        else:
            if cacheable and location == EmbeddingLocation.MANAGED_CACHING:
                placements.append(EmbeddingLocation.MANAGED_CACHING)
            else:
                placements.append(EmbeddingLocation.MANAGED)
            offsets.append(uvm_size)
            uvm_size += state_size
    assert len(placements) == len(offsets)
    return SplitState(
        dev_size=dev_size,
        host_size=host_size,
        uvm_size=uvm_size,
        placements=placements,
        offsets=offsets,
    )


@dataclass
class CacheState:
    # T + 1 elements and cache_hash_size_cumsum[-1] == total_cache_hash_size
    cache_hash_size_cumsum: List[int]
    cache_index_table_map: List[int]
    total_cache_hash_size: int


def construct_cache_state(
    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]],
    feature_table_map: List[int],
) -> CacheState:
    _cache_hash_size_cumsum = [0]
    total_cache_hash_size = 0
    for (num_embeddings, _, location, _) in embedding_specs:
        if location == EmbeddingLocation.MANAGED_CACHING:
            total_cache_hash_size += num_embeddings
        _cache_hash_size_cumsum.append(total_cache_hash_size)
    # [T], -1: non-cached table
    cache_hash_size_cumsum = []
    # [total_cache_hash_size], linear cache index -> table index
    cache_index_table_map = [-1] * total_cache_hash_size
    for t, t_ in enumerate(feature_table_map):
        for i in range(_cache_hash_size_cumsum[t_], _cache_hash_size_cumsum[t_ + 1]):
            cache_index_table_map[i] = t
        (_, _, location, _) = embedding_specs[t_]
        if location == EmbeddingLocation.MANAGED_CACHING:
            cache_hash_size_cumsum.append(_cache_hash_size_cumsum[t_])
        else:
            cache_hash_size_cumsum.append(-1)
    cache_hash_size_cumsum.append(total_cache_hash_size)
    s = CacheState(
        cache_hash_size_cumsum=cache_hash_size_cumsum,
        cache_index_table_map=cache_index_table_map,
        total_cache_hash_size=total_cache_hash_size,
    )
    return s


class SplitTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Multiple sparse features can share one embedding table.
    'feature_table_map' specifies the feature-table mapping.
    T:  number of logical tables
    T_: number of physical tables
    T >= T_
    """

    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]]
    optimizer_args: invokers.lookup_args.OptimizerArgs
    lxu_cache_locations_list: List[Tensor]
    lxu_cache_locations_empty: Tensor
    timesteps_prefetched: List[int]
    record_cache_metrics: RecordCacheMetrics

    def __init__(  # noqa C901
        self,
        embedding_specs: List[
            Tuple[int, int, EmbeddingLocation, ComputeDevice]
        ],  # tuple of (rows, dims, placements, compute_devices)
        feature_table_map: Optional[List[int]] = None,  # [T]
        cache_algorithm: CacheAlgorithm = CacheAlgorithm.LRU,
        cache_load_factor: float = 0.2,
        cache_sets: int = 0,
        cache_reserved_memory: float = 0.0,
        cache_precision: SparseType = SparseType.FP32,
        weights_precision: SparseType = SparseType.FP32,
        pooled_output_precision: SparseType = SparseType.FP32,
        enforce_hbm: bool = False,  # place all weights/momentums in HBM when using cache
        optimizer: OptimType = OptimType.EXACT_SGD,
        record_cache_metrics: Optional[RecordCacheMetrics] = None,
        # General Optimizer args
        stochastic_rounding: bool = False,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        learning_rate: float = 0.01,
        eps: float = 1.0e-8,  # used by Adagrad, LAMB, and Adam
        momentum: float = 0.9,  # used by LARS-SGD
        weight_decay: float = 0.0,  # used by LARS-SGD, LAMB, and ADAM
        eta: float = 0.001,  # used by LARS-SGD,
        beta1: float = 0.9,  # used by LAMB and ADAM
        beta2: float = 0.999,  # used by LAMB and ADAM
        pooling_mode: PoolingMode = PoolingMode.SUM,
        device: Optional[torch.device] = None,
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
    ) -> None:
        super(SplitTableBatchedEmbeddingBagsCodegen, self).__init__()

        self.pooling_mode = pooling_mode
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self.weights_precision = weights_precision
        self.pooled_output_precision: int = pooled_output_precision.as_int()

        if record_cache_metrics is not None:
            self.record_cache_metrics = record_cache_metrics
        else:
            self.record_cache_metrics = RecordCacheMetrics(False, False)
        # NOTE: a placeholder to avoid multi-construction and make TorchScript work!
        self.dummy_tensor: Tensor = torch.zeros(0, device=device)

        self.embedding_specs = embedding_specs
        (rows, dims, locations, compute_devices) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        self.dims: List[int] = dims
        assert T_ > 0

        assert all(
            cd == compute_devices[0] for cd in compute_devices
        ), "Heterogenous compute_devices are NOT supported!"
        self.use_cpu: bool = all(cd == ComputeDevice.CPU for cd in compute_devices)
        assert not self.use_cpu or all(
            loc == EmbeddingLocation.HOST for loc in locations
        ), "ComputeDevice.CPU is only for EmbeddingLocation.HOST!"
        if self.use_cpu:
            assert (
                pooled_output_precision == SparseType.FP32
            ), "Fused pooled embedding quantization only supported for cuda."

        if device is not None:
            self.current_device: torch.device = device
        else:
            self.current_device: torch.device = (
                torch.device("cpu") if self.use_cpu else torch.cuda.current_device()
            )

        # add placeholder require_grad param tensor to enable autograd with int8 weights
        self.placeholder_autograd_tensor = nn.Parameter(
            torch.zeros(0, device=self.current_device, dtype=torch.float)
        )

        self.int8_emb_row_dim_offset: int = INT8_EMB_ROW_DIM_OFFSET

        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        D_offsets = [dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(D_offsets))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(dims)
        cached_dims = [
            embedding_spec[1]
            for embedding_spec in embedding_specs
            if embedding_spec[2] == EmbeddingLocation.MANAGED_CACHING
        ]
        self.max_D_cache: int = max(cached_dims) if len(cached_dims) > 0 else 0

        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )

        hash_size_cumsum = [0] + list(accumulate(rows))
        self.total_hash_size_bits = int(log2(float(hash_size_cumsum[-1])) + 1)
        # The last element is to easily access # of rows of each table by
        # hash_size_cumsum[t + 1] - hash_size_cumsum[t]
        hash_size_cumsum = [hash_size_cumsum[t] for t in self.feature_table_map] + [
            hash_size_cumsum[-1]
        ]
        self.register_buffer(
            "hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )

        self.register_buffer(
            "rows_per_table",
            torch.tensor(
                [rows[t] for t in self.feature_table_map],
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "bounds_check_warning",
            torch.tensor([0], device=self.current_device, dtype=torch.int64),
        )

        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=weights_precision,
        )
        table_embedding_dtype = torch.float32
        if weights_precision == SparseType.FP16:
            table_embedding_dtype = torch.float16
        elif weights_precision == SparseType.INT8:
            table_embedding_dtype = torch.uint8

        self._apply_split(
            weight_split,
            prefix="weights",
            # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param but
            #  got `Type[typing.Union[torch.float16, torch.float32, torch.uint8]]`.
            dtype=table_embedding_dtype,
            enforce_hbm=enforce_hbm,
        )

        if self.use_cpu:
            # Construct optimizer states
            assert optimizer in (
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.ROWWISE_ADAGRAD,
                OptimType.SGD,
            ), f"Optimizer {optimizer} is not supported in cpu mode."
        else:
            assert optimizer in (
                OptimType.ADAM,
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.LAMB,
                OptimType.LARS_SGD,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.PARTIAL_ROWWISE_LAMB,
                OptimType.SGD,
            ), f"Optimizer {optimizer} is not supported."

        self.stochastic_rounding = stochastic_rounding
        self.optimizer = optimizer

        self.optimizer_args = invokers.lookup_args.OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            learning_rate=learning_rate,
            eps=eps,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eta=eta,
            momentum=momentum,
        )

        if optimizer in (
            OptimType.SGD,
            OptimType.EXACT_SGD,
        ):
            # NOTE: make TorchScript work!
            self.register_buffer(
                "momentum1_dev", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            self.register_buffer(
                "momentum1_host", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            self.register_buffer(
                "momentum1_uvm", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            self.register_buffer(
                "momentum1_placements",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "momentum1_offsets",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
        else:
            self._apply_split(
                construct_split_state(
                    embedding_specs,
                    rowwise=optimizer
                    in [OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.ROWWISE_ADAGRAD],
                    cacheable=False,
                ),
                prefix="momentum1",
                # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                #  but got `Type[torch.float32]`.
                dtype=torch.float32,
                enforce_hbm=enforce_hbm,
            )
        if optimizer in (
            OptimType.ADAM,
            OptimType.PARTIAL_ROWWISE_ADAM,
            OptimType.LAMB,
            OptimType.PARTIAL_ROWWISE_LAMB,
        ):
            self._apply_split(
                construct_split_state(
                    embedding_specs,
                    rowwise=optimizer
                    in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.PARTIAL_ROWWISE_LAMB),
                    cacheable=False,
                ),
                prefix="momentum2",
                # pyre-fixme[6]: Expected `Type[Type[torch._dtype]]` for 3rd param
                #  but got `Type[torch.float32]`.
                dtype=torch.float32,
            )
            self.register_buffer(
                "iter", torch.zeros(1, dtype=torch.int64, device=self.current_device)
            )
        else:
            # NOTE: make TorchScript work!
            self.register_buffer(
                "momentum2_dev",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "momentum2_host",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "momentum2_uvm",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "momentum2_placements",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "momentum2_offsets",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "iter",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )

        cache_state = construct_cache_state(embedding_specs, self.feature_table_map)

        # Add table-wise cache miss counter
        if self.record_cache_metrics.record_tablewise_cache_miss:
            num_tables = len(cache_state.cache_hash_size_cumsum) - 1
            self.register_buffer(
                "table_wise_cache_miss",
                torch.zeros(
                    num_tables,
                    device=self.current_device,
                    dtype=torch.int64,
                ),
            )
        # NOTE: make TorchScript work!
        else:
            self.register_buffer(
                "table_wise_cache_miss",
                torch.zeros(
                    0,
                    device=self.current_device,
                    dtype=torch.int64,
                ),
            )

        if cache_precision == SparseType.FP32:
            cache_embedding_dtype = torch.float32
        elif cache_precision == SparseType.FP16:
            cache_embedding_dtype = torch.float16
        else:
            raise AssertionError(f"cache_precision {cache_precision} not supported!")

        self._apply_cache_state(
            cache_state,
            cache_algorithm,
            cache_load_factor,
            cache_sets,
            cache_reserved_memory,
            dtype=cache_embedding_dtype,
        )

        logging.debug(
            f"Using fused {optimizer} with optimizer_args={self.optimizer_args}"
        )

        self.step = 0

    def get_states(self, prefix: str) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not hasattr(self, f"{prefix}_physical_placements"):
            raise DoesNotHavePrefix()
        dev_param = getattr(self, f"{prefix}_dev")
        host_param = getattr(self, f"{prefix}_host")
        uvm_param = getattr(self, f"{prefix}_uvm")
        placements = getattr(self, f"{prefix}_physical_placements")
        offsets = getattr(self, f"{prefix}_physical_offsets")
        return (
            dev_param,
            host_param,
            uvm_param,
            torch.tensor(placements, dtype=torch.int32),
            torch.tensor(offsets, dtype=torch.int64),
        )

    def get_all_states(self) -> List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        all_states = []
        for prefix in ["weights", "momentum1", "momentum2"]:
            try:
                all_states.append(self.get_states(prefix))
            except DoesNotHavePrefix:
                pass
        return all_states

    @torch.jit.export
    def get_cache_miss_counter(self) -> Tensor:
        # cache_miss_counter contains two items:
        # The first one is cache_miss_forward_count which records the total number of forwards which has at least one cache miss
        # The second one is the unique_cache_miss_count which records to total number of unique (dedup) cache misses

        # pyre-fixme[7]: Expected `Tensor` but got `typing.Union[Tensor,
        # nn.Module]`.
        return self.cache_miss_counter

    @torch.jit.export
    def get_table_wise_cache_miss(self) -> Tensor:
        # table_wise_cache_miss contains all the cache miss count for each table in this embedding table object:

        return self.table_wise_cache_miss

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        (indices, offsets) = indices.long(), offsets.long()
        if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
            torch.ops.fb.bounds_check_indices(
                self.rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode_int,
                self.bounds_check_warning,
            )
        self.step += 1
        if len(self.timesteps_prefetched) == 0:
            self.prefetch(indices, offsets)

        self.timesteps_prefetched.pop(0)
        lxu_cache_locations = (
            self.lxu_cache_locations_empty
            if len(self.lxu_cache_locations_list) == 0
            else self.lxu_cache_locations_list.pop(0)
        )
        common_args = invokers.lookup_args.CommonArgs(
            placeholder_autograd_tensor=self.placeholder_autograd_tensor,
            # pyre-fixme[6]: Expected `Tensor` for 2nd param but got `Union[Tensor,
            #  nn.Module]`.
            dev_weights=self.weights_dev,
            # pyre-fixme[6]: Expected `Tensor` for 3rd param but got `Union[Tensor,
            #  nn.Module]`.
            host_weights=self.weights_host,
            # pyre-fixme[6]: Expected `Tensor` for 4th param but got `Union[Tensor,
            #  nn.Module]`.
            uvm_weights=self.weights_uvm,
            # pyre-fixme[6]: Expected `Tensor` for 5th param but got `Union[Tensor,
            #  nn.Module]`.
            lxu_cache_weights=self.lxu_cache_weights,
            # pyre-fixme[6]: Expected `Tensor` for 6th param but got `Union[Tensor,
            #  nn.Module]`.
            weights_placements=self.weights_placements,
            # pyre-fixme[6]: Expected `Tensor` for 7th param but got `Union[Tensor,
            #  nn.Module]`.
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_D=self.max_D,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=indices,
            offsets=offsets,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
            feature_requires_grad=feature_requires_grad,
            lxu_cache_locations=lxu_cache_locations,
            output_dtype=self.pooled_output_precision,
        )

        if self.optimizer == OptimType.EXACT_SGD:
            return invokers.lookup_sgd.invoke(common_args, self.optimizer_args)
        elif self.optimizer == OptimType.SGD:
            assert self.use_cpu, "Approx SGD is only supported in CPU mode"
            return invokers.lookup_approx_sgd.invoke(common_args, self.optimizer_args)

        momentum1 = invokers.lookup_args.Momentum(
            dev=self.momentum1_dev,
            host=self.momentum1_host,
            uvm=self.momentum1_uvm,
            offsets=self.momentum1_offsets,
            placements=self.momentum1_placements,
        )

        if self.optimizer == OptimType.LARS_SGD:
            return invokers.lookup_lars_sgd.invoke(
                common_args, self.optimizer_args, momentum1
            )
        if self.optimizer == OptimType.EXACT_ADAGRAD:
            return invokers.lookup_adagrad.invoke(
                common_args, self.optimizer_args, momentum1
            )
        if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            return invokers.lookup_rowwise_adagrad.invoke(
                common_args, self.optimizer_args, momentum1
            )
        if self.optimizer == OptimType.ROWWISE_ADAGRAD:
            assert self.use_cpu, "Approx rowwise AdaGrad is only supported in CPU mode"
            return invokers.lookup_approx_rowwise_adagrad.invoke(
                common_args, self.optimizer_args, momentum1
            )

        momentum2 = invokers.lookup_args.Momentum(
            dev=self.momentum2_dev,
            host=self.momentum2_host,
            uvm=self.momentum2_uvm,
            offsets=self.momentum2_offsets,
            placements=self.momentum2_placements,
        )
        # Ensure iter is always on CPU so the increment doesn't synchronize.
        if self.iter.is_cuda:
            self.iter = self.iter.cpu()
        self.iter[0] += 1

        if self.optimizer == OptimType.ADAM:
            return invokers.lookup_adam.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[6]: Expected `int` for 5th param but got `Union[float,
                #  int]`.
                self.iter.item(),
            )
        if self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
            return invokers.lookup_partial_rowwise_adam.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[6]: Expected `int` for 5th param but got `Union[float,
                #  int]`.
                self.iter.item(),
            )
        if self.optimizer == OptimType.LAMB:
            return invokers.lookup_lamb.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[6]: Expected `int` for 5th param but got `Union[float,
                #  int]`.
                self.iter.item(),
            )
        if self.optimizer == OptimType.PARTIAL_ROWWISE_LAMB:
            return invokers.lookup_partial_rowwise_lamb.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[6]: Expected `int` for 5th param but got `Union[float,
                #  int]`.
                self.iter.item(),
            )

        raise ValueError(f"Invalid OptimType: {self.optimizer}")

    def prefetch(self, indices: Tensor, offsets: Tensor) -> None:
        self.timestep += 1
        self.timesteps_prefetched.append(self.timestep)
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.numel)[[Named(self, Tensor)],
        #  int], Tensor], Tensor, nn.Module]` is not a function.
        if not self.lxu_cache_weights.numel():
            return

        (indices, offsets) = indices.long(), offsets.long()
        linear_cache_indices = torch.ops.fb.linearize_cache_indices(
            self.cache_hash_size_cumsum,
            indices,
            offsets,
        )

        if (
            self.record_cache_metrics.record_cache_miss_counter
            or self.record_cache_metrics.record_tablewise_cache_miss
        ):
            lxu_cache_locations = torch.ops.fb.lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
            )
            if self.record_cache_metrics.record_cache_miss_counter:
                self._update_cache_miss_counter(
                    lxu_cache_locations, linear_cache_indices
                )
            if self.record_cache_metrics.record_tablewise_cache_miss:
                self._update_tablewise_cache_miss(
                    lxu_cache_locations, linear_cache_indices, offsets
                )

        if self.cache_algorithm == CacheAlgorithm.LRU:
            torch.ops.fb.lru_cache_populate(
                self.weights_uvm,
                self.cache_hash_size_cumsum,
                self.total_cache_hash_size,
                self.cache_index_table_map,
                self.weights_offsets,
                self.D_offsets,
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.timestep,
                self.lxu_state,
                self.stochastic_rounding,
            )
        elif self.cache_algorithm == CacheAlgorithm.LFU:
            torch.ops.fb.lfu_cache_populate(
                self.weights_uvm,
                self.cache_hash_size_cumsum,
                self.total_cache_hash_size,
                self.cache_index_table_map,
                self.weights_offsets,
                self.D_offsets,
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.lxu_state,
                self.stochastic_rounding,
            )

        assert (
            len(self.lxu_cache_locations_list) < self.max_prefetch_depth
        ), f"self.lxu_cache_locations_list has grown to size: {len(self.lxu_cache_locations_list)}, this exceeds the maximum: {self.max_prefetch_depth}. This probably indicates an error in logic where prefetch() is being called more frequently than forward()"
        self.lxu_cache_locations_list.append(
            torch.ops.fb.lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
            )
        )

    def _update_cache_miss_counter(
        self,
        lxu_cache_locations: Tensor,
        linear_cache_indices: Tensor,
    ) -> None:
        CACHE_MISS = -1
        CACHE_HIT = -2

        cache_missed_locations = torch.where(
            lxu_cache_locations == CACHE_MISS, linear_cache_indices, CACHE_HIT
        )
        unique_ids_list = torch.unique(cache_missed_locations)
        unique_ids_count_list = torch.where(unique_ids_list == CACHE_HIT, 0, 1)

        miss_count = torch.sum(unique_ids_count_list)

        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.__getitem__)[[Named(self,
        #  Tensor), Named(item, typing.Any)], typing.Any], Tensor], Tensor,
        #  nn.Module]` is not a function.
        self.cache_miss_counter[0] += (miss_count > 0).to(torch.int64)

        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.__getitem__)[[Named(self,
        #  Tensor), Named(item, typing.Any)], typing.Any], Tensor], Tensor,
        #  nn.Module]` is not a function.
        self.cache_miss_counter[1] += miss_count

    def _update_tablewise_cache_miss(
        self,
        lxu_cache_locations: Tensor,
        linear_cache_indices: Tensor,
        offsets: Tensor,
    ) -> None:
        CACHE_MISS = -1
        CACHE_HIT = -2

        # pyre-ignore[6]:
        # Incompatible parameter type [6]: Expected `typing.Sized` for 1st
        # positional only parameter to call `len` but got `typing.Union[Tensor, nn.Module]`.
        num_tables = len(self.cache_hash_size_cumsum) - 1
        num_offsets_per_table = (len(offsets) - 1) // num_tables
        cache_missed_locations = torch.where(
            lxu_cache_locations == CACHE_MISS, linear_cache_indices, CACHE_HIT
        )

        for i in range(num_tables):
            start = offsets[i * num_offsets_per_table]
            end = offsets[(i + 1) * num_offsets_per_table]

            current_cache_missed_locations = cache_missed_locations[start:end]
            unique_ids_list = torch.unique(current_cache_missed_locations)
            unique_ids_count_list = torch.where(unique_ids_list == CACHE_HIT, 0, 1)

            miss_count = torch.sum(unique_ids_count_list)

            self.table_wise_cache_miss[i] += miss_count

    def init_embedding_weights_uniform(self, min_val: float, max_val: float) -> None:
        splits = self.split_embedding_weights()
        if self.weights_precision == SparseType.INT8:
            # TODO: add in-place FloatToFused8BitRowwiseQuantized conversion
            for emb in splits:
                assert (
                    len(emb.shape) == 2
                ), "Int8 embedding only supported for 2D weight tensors."
                shape = [emb.shape[0], emb.shape[1] - self.int8_emb_row_dim_offset]
                tmp_emb = torch.zeros(shape, device=self.current_device)
                tmp_emb.uniform_(min_val, max_val)
                tmp_emb_i8 = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(tmp_emb)
                emb.data.copy_(tmp_emb_i8)
        else:
            for param in splits:
                param.uniform_(min_val, max_val)

    @torch.jit.ignore
    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, (rows, dim, _, _) in enumerate(self.embedding_specs):
            if self.weights_precision == SparseType.INT8:
                dim += self.int8_emb_row_dim_offset
            # pyre-fixme[29]:
            #  `Union[BoundMethod[typing.Callable(Tensor.__getitem__)[[Named(self,
            #  Tensor), Named(item, typing.Any)], typing.Any], Tensor], Tensor,
            #  nn.Module]` is not a function.
            placement = self.weights_physical_placements[t]
            # pyre-fixme[29]:
            #  `Union[BoundMethod[typing.Callable(Tensor.__getitem__)[[Named(self,
            #  Tensor), Named(item, typing.Any)], typing.Any], Tensor], Tensor,
            #  nn.Module]` is not a function.
            offset = self.weights_physical_offsets[t]
            if placement == EmbeddingLocation.DEVICE.value:
                weights = self.weights_dev
            elif placement == EmbeddingLocation.HOST.value:
                weights = self.weights_host
            else:
                weights = self.weights_uvm
            splits.append(
                # pyre-fixme[29]:
                #  `Union[BoundMethod[typing.Callable(Tensor.detach)[[Named(self,
                #  Tensor)], Tensor], Tensor], Tensor, nn.Module]` is not a function.
                weights.detach()[offset : offset + rows * dim].view(rows, dim)
            )
        return splits

    @torch.jit.ignore
    def get_optimizer_buffer(self, state: str) -> torch.Tensor:
        for name, buffer in self.named_buffers():
            if name == state:
                return buffer
        return torch.tensor(0)

    @torch.jit.export
    def get_optimizer_state(self) -> List[Dict[str, torch.Tensor]]:
        r"""
        Get the optimizer state dict that matches the OSS Pytorch optims
        TODO: populate the supported list of optimizers
        """
        if (
            self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            or self.optimizer == OptimType.ROWWISE_ADAGRAD
        ):
            list_of_state_dict = [
                {"sum": _sum[0]} for _sum in self.split_optimizer_states()
            ]
        else:
            raise NotImplementedError(
                f"Getting optimizer state {self.optimizer} is not implmeneted"
            )

        return list_of_state_dict

    @torch.jit.ignore
    def split_optimizer_states(self) -> List[Tuple[torch.Tensor]]:
        """
        Returns a list of states, split by table
        """

        def get_optimizer_states(
            state_dev: Tensor,
            state_host: Tensor,
            state_uvm: Tensor,
            state_offsets: Tensor,
            state_placements: Tensor,
            rowwise: bool,
        ) -> List[torch.Tensor]:
            splits = []
            for t, (rows, dim, _, _) in enumerate(self.embedding_specs):
                offset = state_offsets[t]
                placement = state_placements[t]
                if placement == EmbeddingLocation.DEVICE:
                    state = state_dev
                elif placement == EmbeddingLocation.HOST:
                    state = state_host
                else:
                    state = state_uvm
                if not rowwise:
                    splits.append(
                        state.detach()[offset : offset + rows * dim].view(rows, dim)
                    )
                else:
                    splits.append(state.detach()[offset : offset + rows].view(rows))
            return splits

        states: List[List[torch.Tensor]] = []
        if self.optimizer not in (
            OptimType.SGD,
            OptimType.EXACT_SGD,
        ):
            states.append(
                get_optimizer_states(
                    self.momentum1_dev,
                    self.momentum1_host,
                    self.momentum1_uvm,
                    # pyre-fixme[6]: Expected `Tensor` for 4th param but got
                    #  `Union[Tensor, nn.Module]`.
                    self.momentum1_physical_offsets,
                    # pyre-fixme[6]: Expected `Tensor` for 5th param but got
                    #  `Union[Tensor, nn.Module]`.
                    self.momentum1_physical_placements,
                    rowwise=self.optimizer
                    in [OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.ROWWISE_ADAGRAD],
                )
            )
        if self.optimizer in (
            OptimType.ADAM,
            OptimType.PARTIAL_ROWWISE_ADAM,
            OptimType.LAMB,
            OptimType.PARTIAL_ROWWISE_LAMB,
        ):
            states.append(
                get_optimizer_states(
                    self.momentum2_dev,
                    self.momentum2_host,
                    self.momentum2_uvm,
                    # pyre-fixme[6]: Expected `Tensor` for 4th param but got
                    #  `Union[Tensor, nn.Module]`.
                    self.momentum2_physical_offsets,
                    # pyre-fixme[6]: Expected `Tensor` for 5th param but got
                    #  `Union[Tensor, nn.Module]`.
                    self.momentum2_physical_placements,
                    rowwise=self.optimizer
                    in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.PARTIAL_ROWWISE_LAMB),
                )
            )
        return list(zip(*states))

    @torch.jit.export
    def set_learning_rate(self, lr: float) -> None:
        """
        Sets the learning rate.
        """
        self._set_learning_rate(lr)

    @torch.jit.ignore
    def _set_learning_rate(self, lr: float) -> float:
        """
        Helper function to script `set_learning_rate`.
        Note that returning None does not work.
        """
        self.optimizer_args = self.optimizer_args._replace(learning_rate=lr)
        return 0.0

    @torch.jit.export
    def flush(self) -> None:
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.numel)[[Named(self, Tensor)],
        #  int], Tensor], Tensor, nn.Module]` is not a function.
        if not self.lxu_cache_weights.numel():
            return
        torch.ops.fb.lxu_cache_flush(
            self.weights_uvm,
            self.cache_hash_size_cumsum,
            self.cache_index_table_map,
            self.weights_offsets,
            self.D_offsets,
            self.total_D,
            self.lxu_cache_state,
            self.lxu_cache_weights,
            self.stochastic_rounding,
        )

    def _apply_split(
        self,
        split: SplitState,
        prefix: str,
        dtype: Type[torch.dtype],
        enforce_hbm: bool = False,
    ) -> None:
        setattr(self, f"{prefix}_physical_placements", split.placements)
        setattr(self, f"{prefix}_physical_offsets", split.offsets)

        offsets = [split.offsets[t] for t in self.feature_table_map]
        placements = [split.placements[t] for t in self.feature_table_map]
        self.register_buffer(
            f"{prefix}_offsets",
            torch.tensor(offsets, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            f"{prefix}_placements",
            torch.tensor(placements, device=self.current_device, dtype=torch.int32),
        )
        if split.dev_size > 0:
            self.register_buffer(
                f"{prefix}_dev",
                # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for 3rd
                #  param but got `Type[Type[torch._dtype]]`.
                torch.zeros(split.dev_size, device=self.current_device, dtype=dtype),
            )
        else:
            self.register_buffer(
                f"{prefix}_dev",
                torch.empty(0, device=self.current_device, dtype=dtype),
            )
        if split.host_size > 0:
            if dtype == torch.uint8:
                self.register_buffer(
                    f"{prefix}_host",
                    torch.zeros(
                        split.host_size,
                        device=self.current_device,
                        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for
                        #  3rd param but got `Type[Type[torch._dtype]]`.
                        dtype=dtype,
                    ),
                )
            else:
                setattr(
                    self,
                    f"{prefix}_host",
                    nn.Parameter(
                        torch.zeros(
                            split.host_size,
                            device=self.current_device,
                            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                            #  for 3rd param but got `Type[Type[torch._dtype]]`.
                            dtype=dtype,
                        )
                    ),
                )
        else:
            self.register_buffer(
                f"{prefix}_host",
                torch.empty(0, device=self.current_device, dtype=dtype),
            )
        if split.uvm_size > 0:
            assert not self.use_cpu
            if enforce_hbm:
                self.register_buffer(
                    f"{prefix}_uvm",
                    torch.zeros(
                        split.uvm_size,
                        device=self.current_device,
                        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for
                        #  3rd param but got `Type[Type[torch._dtype]]`.
                        dtype=dtype,
                    ),
                )
            else:
                self.register_buffer(
                    f"{prefix}_uvm",
                    torch.zeros(
                        split.uvm_size,
                        out=torch.ops.fb.new_managed_tensor(
                            # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                            #  for 3rd param but got `Type[Type[torch._dtype]]`.
                            torch.zeros(1, device=self.current_device, dtype=dtype),
                            [split.uvm_size],
                        ),
                    ),
                )
        else:
            self.register_buffer(
                f"{prefix}_uvm",
                torch.empty(0, device=self.current_device, dtype=dtype),
            )

    def _apply_cache_state(
        self,
        cache_state: CacheState,
        cache_algorithm: CacheAlgorithm,
        cache_load_factor: float,
        cache_sets: int,
        cache_reserved_memory: float,
        dtype: torch.dtype,
    ) -> None:
        self.cache_algorithm = cache_algorithm
        self.timestep = 1
        self.timesteps_prefetched = []

        self.max_prefetch_depth = MAX_PREFETCH_DEPTH
        self.lxu_cache_locations_list = []
        self.lxu_cache_locations_empty = torch.empty(
            0, device=self.current_device, dtype=torch.int32
        ).fill_(-1)

        # NOTE: no cache for CPU mode!
        if cache_state.total_cache_hash_size == 0 or self.use_cpu:
            self.register_buffer(
                "lxu_cache_weights",
                torch.zeros(0, 0, device=self.current_device, dtype=dtype),
            )
            # NOTE: make TorchScript work!
            self.register_buffer(
                "cache_hash_size_cumsum",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "total_cache_hash_size",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "cache_index_table_map",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "lxu_cache_state",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "lxu_state",
                torch.zeros(1, dtype=torch.int64, device=self.current_device),
                persistent=False,
            )
            self.register_buffer(
                "cache_miss_counter",
                torch.tensor([0, 0], dtype=torch.int64),
                persistent=False,
            )
            return

        assert cache_load_factor > 0
        element_size = 2 if dtype == torch.float16 else 4
        if cache_sets <= 0:
            total_memory = torch.cuda.get_device_properties(
                self.current_device
            ).total_memory
            free_memory = (
                total_memory
                - torch.cuda.memory_reserved(self.current_device)
                - int(cache_reserved_memory)
            )
            assert free_memory > 0
            cache_sets = (
                int(cache_state.total_cache_hash_size * cache_load_factor) + ASSOC - 1
            ) // ASSOC
            cache_size = cache_sets * ASSOC * element_size * self.max_D_cache
            if cache_size > free_memory:
                cache_sets = (
                    int(1.0 * free_memory / self.max_D_cache / element_size) + ASSOC - 1
                ) // ASSOC
        cache_load_factor = (
            1.0 * cache_sets * ASSOC / int(cache_state.total_cache_hash_size)
        )
        assert cache_sets > 0
        if cache_algorithm == CacheAlgorithm.LFU:
            assert cache_sets < 2 ** 24 - 1
        cache_size = cache_sets * 32 * element_size * self.max_D_cache
        logging.info(
            f"Using on-device cache with admission algorithm "
            f"{cache_algorithm}, {cache_sets} sets, "
            f"load_factor: {cache_load_factor : .3f}, "
            f"{cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB"
        )

        self.total_cache_hash_size = cache_state.total_cache_hash_size
        self.register_buffer(
            "cache_hash_size_cumsum",
            torch.tensor(
                cache_state.cache_hash_size_cumsum,
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "cache_index_table_map",
            torch.tensor(
                cache_state.cache_index_table_map,
                device=self.current_device,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "lxu_cache_state",
            torch.zeros(
                cache_sets, ASSOC, device=self.current_device, dtype=torch.int64
            ).fill_(-1),
        )
        self.register_buffer(
            "lxu_cache_weights",
            torch.zeros(
                cache_sets * ASSOC,
                self.max_D_cache,
                device=self.current_device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "lxu_state",
            # pyre-fixme[28]: Unexpected keyword argument `size`.
            torch.zeros(
                size=(self.total_cache_hash_size + 1,)
                if cache_algorithm == CacheAlgorithm.LFU
                else (cache_sets, ASSOC),
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "cache_miss_counter",
            torch.tensor([0, 0], device=self.current_device, dtype=torch.int64),
        )
        if cache_algorithm not in (CacheAlgorithm.LFU, CacheAlgorithm.LRU):
            raise ValueError(
                f"cache_algorithm must be {CacheAlgorithm.LRU} "
                f"or {CacheAlgorithm.LFU}"
            )

    def reset_cache_states(self) -> None:
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.numel)[[Named(self, Tensor)],
        #  int], Tensor], Tensor, nn.Module]` is not a function.
        if not self.lxu_cache_weights.numel():
            return
        self.lxu_cache_state.fill_(-1)
        self.lxu_state.fill_(0)
        self.timestep = 1


class DenseTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Table-batched version of nn.EmbeddingBag(sparse=False)
    """

    weights: Tensor
    weights_offsets: Tensor
    D_offsets: Tensor
    total_D: int
    max_D: int
    hash_size_cumsum: Tensor
    total_hash_size_bits: int
    embedding_specs: List[Tuple[int, int]]

    def __init__(
        self,
        embedding_specs: List[Tuple[int, int]],  # tuple of (rows, dims)
        feature_table_map: Optional[List[int]] = None,  # [T]
        pooling_mode: PoolingMode = PoolingMode.SUM,
        use_cpu: bool = False,
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super(DenseTableBatchedEmbeddingBagsCodegen, self).__init__()

        self.pooling_mode = pooling_mode

        self.use_cpu = use_cpu
        self.current_device: torch.device = (
            torch.device("cpu") if self.use_cpu else torch.cuda.current_device()
        )

        self.embedding_specs = embedding_specs
        (rows, dims) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        assert T_ > 0

        feature_table_map = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(feature_table_map)
        assert T_ <= T
        D_offsets = [dims[t] for t in feature_table_map]
        D_offsets = [0] + list(accumulate(D_offsets))
        self.total_D = D_offsets[-1]
        self.max_D = max(dims)
        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )
        assert self.D_offsets.numel() == T + 1

        hash_size_cumsum = [0] + list(accumulate(rows))
        self.total_hash_size_bits = int(log2(float(hash_size_cumsum[-1])) + 1)
        # The last element is to easily access # of rows of each table by
        # hash_size_cumsum[t + 1] - hash_size_cumsum[t]
        hash_size_cumsum = [hash_size_cumsum[t] for t in feature_table_map] + [
            hash_size_cumsum[-1]
        ]
        self.register_buffer(
            "hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )
        weights_offsets = [0] + list(
            accumulate([row * dim for (row, dim) in embedding_specs])
        )
        self.weights = nn.Parameter(
            torch.randn(
                weights_offsets[-1],
                device=self.current_device,
            )
        )
        for feature in range(T):
            t = feature_table_map[feature]
            row, dim = embedding_specs[t]
            if (
                self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()
                != row * dim
            ):
                logging.info(
                    f"row {row} dim {dim} feature {feature} t {t} {self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()}"
                )
            assert (
                self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()
                == row * dim
            )
            assert self.hash_size_cumsum[feature] == sum(
                row for (row, _) in embedding_specs[:t]
            )

        self.weights_physical_offsets: List[int] = weights_offsets
        weights_offsets = [weights_offsets[t] for t in feature_table_map]
        self.register_buffer(
            "weights_offsets",
            torch.tensor(
                weights_offsets, device=self.current_device, dtype=torch.int64
            ),
        )

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        (indices, offsets) = indices.long(), offsets.long()
        return torch.ops.fb.dense_embedding_codegen_lookup_function(
            dev_weights=self.weights,
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_D=self.max_D,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=indices,
            offsets=offsets,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        )

    @torch.jit.export
    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, (rows, dim) in enumerate(self.embedding_specs):
            offset = self.weights_physical_offsets[t]
            splits.append(
                self.weights.detach()[offset : offset + rows * dim].view(rows, dim)
            )
        return splits

    def init_embedding_weights_uniform(self, min_val: float, max_val: float) -> None:
        splits = self.split_embedding_weights()
        for param in splits:
            param.uniform_(min_val, max_val)


class SequenceEmbeddingCodegen(SplitTableBatchedEmbeddingBagsCodegen):
    """
    This class wraps around SplitTableBatchedEmbeddingBagsCodegen to get
    sequence embedding op: nn.EmbeddingBag(sparse=True)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        # assert T == 1
        assert "embedding_specs" in kwargs
        assert len(kwargs["embedding_specs"]) == 1
        super(SequenceEmbeddingCodegen, self).__init__(
            **kwargs,
        )

    # @torch.jit.ignore
    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        offsets = torch.arange(
            0,
            indices.numel() + 1,
            device=indices.device,
            dtype=torch.int64,
        )
        return super(SequenceEmbeddingCodegen, self).forward(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad,
        )


class DenseSequenceEmbeddingCodegen(DenseTableBatchedEmbeddingBagsCodegen):
    """
    This class wraps around DenseTableBatchedEmbeddingBagsCodegen to get
    sequence embedding op, nn.EmbeddingBag(sparse=False)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        # assert T == 1
        assert "embedding_specs" in kwargs
        assert len(kwargs["embedding_specs"]) == 1
        super(DenseSequenceEmbeddingCodegen, self).__init__(
            **kwargs,
        )

    # @torch.jit.ignore
    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        offsets = torch.arange(
            0,
            indices.numel() + 1,
            device=indices.device,
            dtype=torch.int64,
        )
        return super(DenseSequenceEmbeddingCodegen, self).forward(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad,
        )


def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def rounded_row_size_in_bytes(dim: int, weight_ty: SparseType) -> int:
    r = unpadded_row_size_in_bytes(dim, weight_ty)
    # align each row to 16-byte boundaries.
    return round_up(r, 16)


def unpadded_row_size_in_bytes(dim: int, weight_ty: SparseType) -> int:
    r = {
        SparseType.FP32.value: dim * 4,
        SparseType.FP16.value: dim * 2,
        SparseType.INT8.value: dim + 4,
        SparseType.INT4.value: dim // 2 + 4,
        SparseType.INT2.value: dim // 4 + 4,
    }[weight_ty.value]
    return r


def intn_construct_split_state(
    embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]],
    cacheable: bool,
) -> SplitState:
    placements = []
    offsets = []
    dev_size = 0
    host_size = 0
    uvm_size = 0
    for (_, num_embeddings, embedding_dim, weight_ty, location) in embedding_specs:

        def align_to_cacheline(a: int) -> int:
            # align each table to 128b cache line boundary.
            return round_up(a, 128)

        embedding_dim = rounded_row_size_in_bytes(embedding_dim, weight_ty)
        state_size = align_to_cacheline(num_embeddings * embedding_dim)
        if location == EmbeddingLocation.HOST:
            placements.append(EmbeddingLocation.HOST)
            offsets.append(host_size)
            host_size += state_size
        elif location == EmbeddingLocation.DEVICE:
            placements.append(EmbeddingLocation.DEVICE)
            offsets.append(dev_size)
            dev_size += state_size
        else:
            if cacheable and location == EmbeddingLocation.MANAGED_CACHING:
                placements.append(
                    EmbeddingLocation.MANAGED_CACHING
                )  # Note: this isn't supported yet.
                raise AssertionError("MANAGED_CACHING is not supported yet")
            else:
                placements.append(EmbeddingLocation.MANAGED)
            offsets.append(uvm_size)
            uvm_size += state_size
    assert len(placements) == len(offsets)
    return SplitState(
        dev_size=dev_size,
        host_size=host_size,
        uvm_size=uvm_size,
        placements=placements,
        offsets=offsets,
    )


class IntNBitTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Table-batched version of nn.EmbeddingBag(sparse=False)
    Inference version, with FP16/INT8/INT4 supports
    """

    def __init__(
        self,
        embedding_specs: List[
            Tuple[str, int, int, SparseType, EmbeddingLocation]
        ],  # tuple of (feature_names, rows, dims, SparseType, EmbeddingLocation/placement)
        feature_table_map: Optional[List[int]] = None,  # [T]
        index_remapping: Optional[List[Tensor]] = None,
        pooling_mode: PoolingMode = PoolingMode.SUM,
        device: Optional[Union[str, int, torch.device]] = None,
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        weight_lists: Optional[List[Tuple[Tensor, Tensor]]] = None,
        load_factor: float = 0.5,
        use_array_for_index_remapping: bool = True,
        output_dtype: SparseType = SparseType.FP16,
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super(IntNBitTableBatchedEmbeddingBagsCodegen, self).__init__()

        if device is None:
            self.current_device: torch.device = torch.device(
                torch.cuda.current_device()
            )
        elif isinstance(device, torch.device):
            self.current_device = device
        else:
            # pyre-ignore [6]
            self.current_device = torch.device(device)
        self.use_cpu: bool = self.current_device.type == "cpu"

        self.pooling_mode = pooling_mode
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self.embedding_specs = embedding_specs
        self.output_dtype: int = output_dtype.as_int()
        # (feature_names, rows, dims, weights_tys, locations) = zip(*embedding_specs)
        # Pyre workaround
        self.feature_names: List[str] = [e[0] for e in embedding_specs]
        rows: List[int] = [e[1] for e in embedding_specs]
        dims: List[int] = [e[2] for e in embedding_specs]
        self.dims: List[int] = dims
        weights_tys: List[SparseType] = [e[3] for e in embedding_specs]
        locations: List[EmbeddingLocation] = [e[4] for e in embedding_specs]

        assert not self.use_cpu or all(
            loc == EmbeddingLocation.HOST for loc in locations
        ), "ComputeDevice.CPU is only for EmbeddingLocation.HOST!"

        T_ = len(self.embedding_specs)

        assert T_ > 0
        for (dim, weight_ty) in zip(dims, weights_tys):
            assert (
                dim % weight_ty.align_size() == 0
            ), f"{dim} % {weight_ty.align_size() } != 0"

        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"
        D_offsets = [dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(D_offsets))
        self.total_D: int = D_offsets[-1]
        for weight_ty in weights_tys:
            if not weight_ty.is_float():
                assert (
                    self.total_D % 2 == 0
                ), "the total_D needs to be even so the FP16 output will be aligned with 4 bytes"
                break

        def max_ty_D(ty: SparseType) -> int:
            return max(
                [dim for dim, weight_ty in zip(dims, weights_tys) if weight_ty == ty],
                default=0,
            )

        self.max_int2_D: int = max_ty_D(SparseType.INT2)
        self.max_int4_D: int = max_ty_D(SparseType.INT4)
        self.max_int8_D: int = max_ty_D(SparseType.INT8)
        self.max_float16_D: int = max_ty_D(SparseType.FP16)
        self.max_float32_D: int = max_ty_D(SparseType.FP32)

        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )
        assert self.D_offsets.numel() == T + 1

        self.register_buffer(
            "rows_per_table",
            torch.tensor(
                [rows[t] for t in self.feature_table_map],
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "bounds_check_warning",
            torch.tensor([0], device=self.current_device, dtype=torch.int64),
        )

        def align_to_cacheline(a: int) -> int:
            # align each table to 128b cache line boundary.
            return round_up(a, 128)

        weights_tys_int = [weights_tys[t].as_int() for t in self.feature_table_map]
        self.register_buffer(
            "weights_tys",
            torch.tensor(
                weights_tys_int, device=self.current_device, dtype=torch.uint8
            ),
        )
        self.weight_initialized: bool = False

        self.weights_dev: torch.Tensor = torch.zeros(
            0,
            device=self.current_device,
            dtype=torch.uint8,
        )

        self.weights_host: torch.Tensor = torch.zeros(
            0, device=self.current_device, dtype=torch.uint8
        )

        self.weights_uvm: torch.Tensor = torch.empty(
            0, device=self.current_device, dtype=torch.uint8
        )

        weight_split: SplitState = intn_construct_split_state(
            self.embedding_specs,
            cacheable=True,
        )

        self.weights_physical_placements: List[int] = [
            t.value for t in weight_split.placements
        ]
        self.weights_physical_offsets: List[int] = weight_split.offsets
        self.host_size: int = weight_split.host_size
        self.dev_size: int = weight_split.dev_size
        self.uvm_size: int = weight_split.uvm_size

        # Assign weights after weights and weights_offsets are initialized.
        if weight_lists:
            self._apply_split(
                self.dev_size,
                self.host_size,
                self.uvm_size,
                self.weights_physical_placements,
                self.weights_physical_offsets,
            )
            self.assign_embedding_weights(weight_lists)  # type: ignore

        # Handle index remapping for embedding pruning.
        self.register_buffer(
            "index_remappings_array_offsets",
            torch.empty(0, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_remappings_array",
            torch.empty(0, device=self.current_device, dtype=torch.int32),
        )
        self.register_buffer(
            "index_remapping_hash_table_offsets",
            torch.empty(0, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_remapping_hash_table",
            torch.empty(0, device=self.current_device, dtype=torch.int32),
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.index_remapping_hash_table_cpu = None

        if index_remapping:
            self.set_index_remappings(
                index_remapping, load_factor, use_array_for_index_remapping
            )

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.weight_initialized
        if self.index_remapping_hash_table_cpu is not None:
            indices = self.index_remapping_hash_table_cpu.lookup(indices, offsets)
        elif self.index_remapping_hash_table.numel() > 0:
            # Convert from raw indices to pruned indices
            indices = torch.ops.fb.pruned_hashmap_lookup(
                indices,
                offsets,
                self.index_remapping_hash_table,
                self.index_remapping_hash_table_offsets,
            )
        elif self.index_remappings_array.numel() > 0:
            indices = torch.ops.fb.pruned_array_lookup(
                indices,
                offsets,
                self.index_remappings_array,
                self.index_remappings_array_offsets,
            )

        # We cast to int as a TorchScript workaround.
        if self.bounds_check_mode_int != BoundsCheckMode.NONE.value:
            torch.ops.fb.bounds_check_indices(
                self.rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode_int,
                self.bounds_check_warning,
            )
        # Note: CPU and CUDA ops use the same interface to facilitate JIT IR
        # generation for CUDA/CPU. For CPU op, we don't need weights_uvm and
        # weights_placements
        return torch.ops.fb.int_nbit_split_embedding_codegen_lookup_function(
            dev_weights=self.weights_host if self.host_size > 0 else self.weights_dev,
            uvm_weights=self.weights_uvm,
            weights_placements=self.weights_placements,
            weights_offsets=self.weights_offsets,
            weights_tys=self.weights_tys,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_int2_D=self.max_int2_D,
            max_int4_D=self.max_int4_D,
            max_int8_D=self.max_int8_D,
            max_float16_D=self.max_float16_D,
            max_float32_D=self.max_float32_D,
            output_dtype=self.output_dtype,
            indices=indices,
            offsets=offsets,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
        )

    def _apply_split(
        self,
        dev_size: int,
        host_size: int,
        uvm_size: int,
        placements: List[int],
        offsets: List[int],
    ) -> None:
        assert not self.weight_initialized, "Weights have already been initialized."
        self.weight_initialized = True
        self.weights_physical_placements = placements
        self.weights_physical_offsets = offsets

        self.host_size = host_size
        self.dev_size = dev_size
        self.uvm_size = uvm_size

        offsets = [offsets[t] for t in self.feature_table_map]
        placements = [placements[t] for t in self.feature_table_map]
        self.weights_offsets = torch.tensor(
            offsets, device=self.D_offsets.device, dtype=torch.int64
        )
        self.weights_placements = torch.tensor(
            placements, device=self.D_offsets.device, dtype=torch.int32
        )

        if dev_size > 0:
            self.weights_dev = torch.zeros(
                dev_size,
                device=self.D_offsets.device,
                dtype=torch.uint8,
            )

        if host_size > 0:
            self.weights_host = torch.zeros(
                host_size, device=self.D_offsets.device, dtype=torch.uint8
            )

        if uvm_size > 0:
            assert not self.use_cpu
            self.weights_uvm = torch.zeros(
                uvm_size,
                out=torch.ops.fb.new_managed_tensor(
                    torch.zeros(1, device=self.D_offsets.device, dtype=torch.uint8),
                    [uvm_size],
                ),
            )

    @torch.jit.export
    def split_embedding_weights(
        self, split_scale_shifts: bool = True
    ) -> List[Tuple[Tensor, Optional[Tensor]]]:
        """
        Returns a list of weights, split by table
        """
        assert self.weight_initialized
        splits: List[Tuple[Tensor, Optional[Tensor]]] = []
        for t, (_, rows, dim, weight_ty, _) in enumerate(self.embedding_specs):
            placement = self.weights_physical_placements[t]
            if placement == EmbeddingLocation.DEVICE.value:
                weights = self.weights_dev
            elif placement == EmbeddingLocation.HOST.value:
                weights = self.weights_host
            else:
                weights = self.weights_uvm
            offset = self.weights_physical_offsets[t]
            weights_shifts = weights.detach()[
                offset : offset + rows * rounded_row_size_in_bytes(dim, weight_ty)
            ].view(rows, rounded_row_size_in_bytes(dim, weight_ty))
            if split_scale_shifts:
                # remove the padding at the end of each row.
                weights_shifts = weights_shifts[
                    :, : unpadded_row_size_in_bytes(dim, weight_ty)
                ]
                if (
                    weight_ty == SparseType.INT8
                    or weight_ty == SparseType.INT4
                    or weight_ty == SparseType.INT2
                ):
                    splits.append(
                        (
                            weights_shifts[:, 4:],
                            weights_shifts[:, :4],
                        )
                    )
                else:
                    assert weight_ty == SparseType.FP16 or weight_ty == SparseType.FP32
                    splits.append(
                        (
                            weights_shifts,
                            None,
                        )
                    )
            else:
                splits.append((weights_shifts, None))

        return splits

    def initialize_weights(self) -> None:
        if not self.weight_initialized:
            self._apply_split(
                self.dev_size,
                self.host_size,
                self.uvm_size,
                self.weights_physical_placements,
                self.weights_physical_offsets,
            )
            self.weight_initialized: bool = True

    def fill_random_weights(self) -> None:
        """
        Fill the buffer with random weights, table by table
        FIXME: make it in-place fill.
        """
        self.initialize_weights()
        weights = self.split_embedding_weights()
        for dest_weight in weights:
            dest_weight[0].copy_(
                torch.randint(
                    0,
                    255,
                    size=dest_weight[0].shape,
                    dtype=torch.uint8,
                    device=self.current_device,
                )
            )

    def assign_embedding_weights(
        self, q_weight_list: List[Tuple[Tensor, Optional[Tensor]]]
    ) -> None:
        """
        Assigns self.split_embedding_weights() with values from the input list of weights and scale_shifts.
        """
        weights = self.split_embedding_weights()
        assert len(q_weight_list) == len(weights)

        for (dest_weight, input_weight) in zip(weights, q_weight_list):
            dest_weight[0].copy_(input_weight[0])
            if input_weight[1] is not None:
                assert dest_weight[1] is not None
                dest_weight[1].copy_(input_weight[1])
            else:
                assert dest_weight[1] is None

    def set_index_remappings(
        self,
        index_remapping: List[Tensor],
        load_factor: float = 0.5,
        use_array_for_index_remapping: bool = True,
    ) -> None:
        rows: List[int] = [e[1] for e in self.embedding_specs]
        T = len(self.embedding_specs)
        if not use_array_for_index_remapping:
            capacities = [
                round_up(int(row * 1.0 / load_factor), 32)
                if index_remap is not None
                else 0
                for (index_remap, row) in zip(index_remapping, rows)
            ]
            hash_table = torch.empty(
                (sum(capacities), 2),
                dtype=torch.int32,
            )
            hash_table[:, :] = -1
            hash_table_offsets = torch.tensor([0] + list(accumulate(capacities))).long()

            merged_index_remappings = [
                mapping if mapping is not None else Tensor(list(range(spec[1])))
                for (mapping, spec) in zip(index_remapping, self.embedding_specs)
            ]
            original_feature_rows = [
                mapping.numel() for mapping in merged_index_remappings
            ]
            dense_indices = torch.cat(merged_index_remappings, dim=0).int()
            indices = torch.cat(
                [torch.arange(row) for row in original_feature_rows], dim=0
            ).int()
            offsets = torch.tensor([0] + list(accumulate(original_feature_rows))).int()

            if self.use_cpu:
                self.index_remapping_hash_table_cpu = torch.classes.fb.PrunedMapCPU()
                self.index_remapping_hash_table_cpu.insert(
                    indices, dense_indices, offsets, T
                )
            else:
                torch.ops.fb.pruned_hashmap_insert(
                    indices, dense_indices, offsets, hash_table, hash_table_offsets
                )
                self.index_remapping_hash_table = hash_table.to(self.current_device)
                self.index_remapping_hash_table_offsets = hash_table_offsets.to(
                    self.current_device
                )
                self.index_remapping_hash_table_cpu = None
        else:
            index_remappings_array_offsets = [0]
            last_offset = 0
            for mapping in index_remapping:
                if mapping is not None:
                    last_offset += mapping.numel()
                index_remappings_array_offsets.append(last_offset)

            self.index_remappings_array_offsets = torch.tensor(
                index_remappings_array_offsets,
                device=self.current_device,
                dtype=torch.int64,
            )
            self.index_remappings_array = (
                torch.empty(0, dtype=torch.int32, device=self.current_device)
                if self.index_remappings_array_offsets[-1] == 0
                else torch.cat(
                    [mapping for mapping in index_remapping if mapping is not None]
                ).to(self.current_device)
            )
