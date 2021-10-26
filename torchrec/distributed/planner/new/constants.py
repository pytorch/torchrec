#!/usr/bin/env python3

from enum import Enum

HBM_CAP_DEFAULT: int = 32 * 1024 * 1024 * 1024  # 32 GB
DDR_CAP_DEFAULT: int = 2 * 1024 * 1024 * 1024 * 1024  # 2 TB

INTRA_NODE_BANDWIDTH: int = 600
CROSS_NODE_BANDWIDTH: int = 12

DEFAULT_CW_DIM: int = 32
DEFAULT_POOLING_FACTOR: float = 1.0

BIGINT_DTYPE: float = 8.0


class PartitionByType(Enum):
    """
    Well-known partition types
    """

    # Partitioning based on device
    DEVICE = "device"
    # Partitioning based on host
    HOST = "host"
    # Uniform, (ie. fixed layout)
    UNIFORM = "uniform"
