#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import math
from typing import Callable, cast, Dict, List, Optional, Tuple, Type

import torch
from torch import distributed as dist, nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import none_throws

MIN_CW_DIM: int = 128


def get_default_sharders() -> List[ModuleSharder[nn.Module]]:
    return [
        cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], EmbeddingCollectionSharder()),
        cast(ModuleSharder[nn.Module], FusedEmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], QuantEmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], QuantEmbeddingCollectionSharder()),
    ]


def get_module_to_default_sharders() -> Dict[Type[nn.Module], ModuleSharder[nn.Module]]:
    return {sharder.module_type: sharder for sharder in get_default_sharders()}


def placement(
    compute_device: str,
    rank: int,
    local_size: int,
) -> str:
    param_device = compute_device
    if compute_device == "cuda":
        param_device = torch.device("cuda", rank % local_size)
    return f"rank:{rank}/{param_device}"


def calculate_shard_sizes_and_offsets(
    tensor: torch.Tensor,
    world_size: int,
    local_world_size: int,
    sharding_type: str,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Calculates sizes and offsets for tensor sharded according to provided sharding type.

    Args:
        tensor (torch.Tensor): tensor to be sharded.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        sharding_type (str): provided ShardingType value.
        col_wise_shard_dim (Optional[int]): dimension for column wise sharding split.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: shard sizes, represented as a list of the dimensions of the sharded tensor on each device, and shard offsets, represented as a list of coordinates of placement on each device.

    Raises:
        ValueError: If `sharding_type` is not a valid ShardingType.
    """

    (rows, columns) = tensor.shape

    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return [[rows, columns]] * world_size, [[0, 0]] * world_size
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return [[rows, columns]], [[0, 0]]
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _calculate_rw_shard_sizes_and_offsets(rows, world_size, columns)
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _calculate_rw_shard_sizes_and_offsets(rows, local_world_size, columns)
    elif (
        sharding_type == ShardingType.COLUMN_WISE.value
        or sharding_type == ShardingType.TABLE_COLUMN_WISE.value
    ):
        return _calculate_cw_shard_sizes_and_offsets(columns, rows, col_wise_shard_dim)

    raise ValueError(
        f"Unrecognized or unsupported sharding type provided: {sharding_type}"
    )


def _calculate_rw_shard_sizes_and_offsets(
    hash_size: int, num_devices: int, columns: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Sets prefix of shard_sizes to be ceil(hash_size/num_devices).

    For example if hash_size = 10, num_devices = 3, we will allocate the rows as 3,3,3,1
    (rather than 3,3,2,2).
    This is due to implementation in RW sharding that sets block_size_lists to be ceil.
    The balanced way is harder to support on GPU.
    For more details see https://fb.quip.com/xbgbAchCTOL0

    Also consider the example of hash_size = 5, num_devices = 4. The expected rows per
    rank is [2,2,1,0].
    """

    block_size: int = math.ceil(hash_size / num_devices)
    last_rank: int = hash_size // block_size
    last_block_size: int = hash_size - block_size * last_rank
    shard_sizes: List[List[int]] = []

    for rank in range(num_devices):
        if rank < last_rank:
            local_row: int = block_size
        elif rank == last_rank:
            local_row: int = last_block_size
        else:
            local_row: int = 0
        shard_sizes.append([local_row, columns])
    shard_offsets = [[0, 0]]

    for i in range(num_devices - 1):
        shard_offsets.append([shard_sizes[i][0] + shard_offsets[i][0], 0])

    return shard_sizes, shard_offsets


def _calculate_cw_shard_sizes_and_offsets(
    hash_size: int,
    rows: int,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    block_size: int = min(
        col_wise_shard_dim if col_wise_shard_dim else MIN_CW_DIM, hash_size
    )
    num_col_wise_shards, residual = divmod(hash_size, block_size)

    shard_sizes: List[List[int]] = [[rows, block_size]] * (num_col_wise_shards - 1)
    shard_sizes.append([rows, block_size + residual])

    shard_offsets: List[List[int]] = [
        [0, block_size * rank] for rank in range(num_col_wise_shards)
    ]
    return shard_sizes, shard_offsets


def _get_parameter_size_offsets(
    param: torch.nn.Parameter,
    sharding_type: ShardingType,
    local_size: int,
    world_size: int,
    col_wise_shard_dim: Optional[int] = None,
) -> List[Tuple[List[int], List[int]]]:
    (shard_sizes, shard_offsets,) = calculate_shard_sizes_and_offsets(
        tensor=none_throws(param),
        world_size=world_size,
        local_world_size=local_size,
        sharding_type=sharding_type.value,
        col_wise_shard_dim=col_wise_shard_dim,
    )
    return list(zip(shard_sizes, shard_offsets))


def _get_compute_kernel(
    sharder: ModuleSharder[nn.Module],
    param: nn.Parameter,
    sharding_type: str,
    device_type: str,
) -> str:
    # TODO add placement support for compute_kernel
    compute_kernels = sharder.compute_kernels(sharding_type, device_type)

    if sharding_type == ShardingType.DATA_PARALLEL or not hasattr(
        param, "_optimizer_class"
    ):
        if EmbeddingComputeKernel.DENSE.value in compute_kernels:
            return EmbeddingComputeKernel.DENSE.value
        elif EmbeddingComputeKernel.QUANT.value in compute_kernels:
            return EmbeddingComputeKernel.QUANT.value
    else:
        if EmbeddingComputeKernel.FUSED.value in compute_kernels:
            return EmbeddingComputeKernel.FUSED.value
        elif EmbeddingComputeKernel.QUANT.value in compute_kernels:
            return EmbeddingComputeKernel.QUANT.value

    raise ValueError(f"Could not find compute kernel for sharding_type={sharding_type}")


def _get_parameter_sharding(
    param: nn.Parameter,
    sharding_type: str,
    size_offset_ranks: List[Tuple[List[int], List[int], int]],
    local_size: int,
    device_type: str,
    sharder: ModuleSharder[nn.Module],
) -> ParameterSharding:
    return ParameterSharding(
        sharding_spec=None
        if sharding_type == ShardingType.DATA_PARALLEL.value
        else EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_sizes=size,
                    shard_offsets=offset,
                    placement=placement(
                        device_type,
                        none_throws(rank),
                        none_throws(local_size),
                    ),
                )
                for (size, offset, rank) in (size_offset_ranks)
            ]
        ),
        sharding_type=sharding_type,
        compute_kernel=_get_compute_kernel(sharder, param, sharding_type, device_type),
        ranks=[rank for (_, _, rank) in size_offset_ranks],
    )


ParameterShardingGenerator = Callable[
    [nn.Parameter, int, int, str, ModuleSharder[nn.Module]], ParameterSharding
]


def data_parallel_sharding() -> ParameterShardingGenerator:
    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.DATA_PARALLEL,
            local_size,
            world_size,
        )
        size_offset_ranks = []

        assert len(size_and_offsets) == world_size
        for (size, offset), rank in zip(size_and_offsets, range(world_size)):
            size_offset_ranks.append((size, offset, rank))

        return _get_parameter_sharding(
            param,
            ShardingType.DATA_PARALLEL.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def table_wise_sharding(
    rank: int,
) -> ParameterShardingGenerator:
    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.TABLE_WISE,
            local_size,
            world_size,
        )
        assert len(size_and_offsets) == 1
        (size, offset) = size_and_offsets[0]
        size_offset_ranks = [(size, offset, rank)]

        return _get_parameter_sharding(
            param,
            ShardingType.TABLE_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def row_wise_sharding() -> ParameterShardingGenerator:
    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.ROW_WISE,
            local_size,
            world_size,
        )
        assert len(size_and_offsets) <= world_size
        size_offset_ranks = []
        for (size, offset), rank in zip(size_and_offsets, range(world_size)):
            size_offset_ranks.append((size, offset, rank))

        return _get_parameter_sharding(
            param,
            ShardingType.ROW_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def column_wise_sharding(
    shard_dim: int = 128,
    ranks: Optional[List[int]] = None,
) -> ParameterShardingGenerator:
    """
    if ranks is not specified, shards will be placed in a uniform fashion.
    If there are not enough shards, only the first ceil(embedding_dim/shard_dim) ranks will get
    shards
    """

    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param, ShardingType.COLUMN_WISE, local_size, world_size, shard_dim
        )

        size_offset_ranks = []
        if ranks is None:
            for (size, offset), rank in zip(size_and_offsets, range(world_size)):
                size_offset_ranks.append((size, offset, rank))
        else:
            assert len(size_offset_ranks) <= len(ranks)
            for (size, offset), rank in zip(size_and_offsets, ranks):
                size_offset_ranks.append((size, offset, rank))

        return _get_parameter_sharding(
            param,
            ShardingType.COLUMN_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def table_row_wise_sharding(
    node_index: int,
) -> ParameterShardingGenerator:
    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.TABLE_ROW_WISE,
            local_size,
            world_size,
        )

        size_offset_ranks = []
        assert len(size_and_offsets) <= local_size
        for (size, offset), rank in zip(size_and_offsets, range(local_size)):
            size_offset_ranks.append((size, offset, node_index + rank))

        return _get_parameter_sharding(
            param,
            ShardingType.TABLE_ROW_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def construct_module_sharding_plan(
    module: nn.Module,
    per_param_sharding: Dict[str, ParameterShardingGenerator],
    sharder: Optional[ModuleSharder[nn.Module]] = None,
    local_size: Optional[int] = None,
    world_size: Optional[int] = None,
    device_type: str = "cuda",
) -> Dict[str, ParameterSharding]:
    """
    Helper function to create sharding plans (Dict[str, ParameterSharding]) for an module
    Args:
        module (nn.Module): module to create plan for.
        per_param_sharding: Dict[str, Callable[[nn.Parameter, int, int, str], ParameterSharding]]: A mapping of parameter names to a generator function
        that takes in [parameter, local_size, world_size, device_type] and returns a ParameterSharding. We recommend using one of the predefined generator functions
        e.g. table_wise_sharding, row_wise_sharding, etc,
        sharder: Optional[ModuleSharder[nn.Module]]: Sharder that we are creating a plan for. If set to none, we will try to derive it from the module. We recommend setting this to None.
        local_size: Optional[int] = None: process group local size
        world_size: Optional[int] = None: process_group world_size
        device_type: str : Torch device type,

    Example::
        ebc = EmbeddingBagCollection(...)
        plan = construct_parameter_sharding_plan(
            ebc,
            {
                "table_0": data_parallel_sharding(),
                "table_1": row_wise_sharding(),
                "table_2": column_wise_sharding(),
                "table_3": column_wise_sharding(ranks=[0,1,2]),
                "table_4": table_row_wise_sharding(host_index=2),
            },
            EmbeddingBagCollectionSharder()
        )
    """
    if sharder is None:
        sharder = get_module_to_default_sharders().get(type(module), None)
    assert (
        sharder is not None
    ), f"Could not find a valid sharder type for {type(module)}"

    assert isinstance(
        module, sharder.module_type
    ), f"Incorrect sharder for module type {type(module)}"
    shardable_parameters = sharder.shardable_parameters(module)
    assert (
        shardable_parameters.keys() == per_param_sharding.keys()
    ), "per_param_sharding_config doesn't match the shardable parameters of the module"

    local_size = local_size or get_local_size()
    world_size = world_size or dist.get_world_size()

    per_parameter_sharding: Dict[str, ParameterSharding] = {}

    param = None
    for table_name, sharding_plan_generator in per_param_sharding.items():
        param_name = f"{table_name}.weight"
        for name, _param in module.named_parameters():
            if param_name in name:
                param = _param
                break
        assert (
            param is not None
        ), f"Could not find parameter {param_name} in module's named_parameters()"

        per_parameter_sharding[table_name] = sharding_plan_generator(
            param, local_size, world_size, device_type, sharder
        )

    return per_parameter_sharding


def uniform_parameter_sharding(
    module: nn.Module,
    parameter_sharding_generator: ParameterShardingGenerator,
    sharder: Optional[ModuleSharder[nn.Module]] = None,
) -> Dict[str, ParameterShardingGenerator]:
    """
    Convenience function to generate a uniform sharding scheme for construct_module_sharding_plan.

    Example::
    ebc = EmbeddingBagCollection(...)
    sharder = EmbeddingBagCollectionSharder()
    plan = construct_parameter_sharding_plan(
        ebc,
        uniform_parameter_sharding(ebc, row_wise_sharding(), sharder),
        EmbeddingBagCollectionSharder()
    )

    """
    if sharder is None:
        # pyre-ignore
        sharder = get_module_to_default_sharders.get(type(module), None)
    else:
        assert isinstance(
            module, sharder.module_type
        ), f"Incorrect sharder for module type {type(module)}"

    assert (
        sharder is not None
    ), f"Could not find a valid sharder type for {type(module)}"

    shardable_parameters = sharder.shardable_parameters(module)
    return {
        param_name: parameter_sharding_generator for param_name in shardable_parameters
    }
