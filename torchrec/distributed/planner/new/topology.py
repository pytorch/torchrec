#!/usr/bin/env python3

from typing import Optional

from torchrec.distributed.planner.new.constants import (
    INTRA_NODE_BANDWIDTH,
    CROSS_NODE_BANDWIDTH,
)
from torchrec.distributed.planner.new.types import Topology


class SMCTopology(Topology):
    def __init__(
        self,
        world_size: int,
        compute_device: str,
        hbm_cap: Optional[int] = None,
        ddr_cap: Optional[int] = None,
        local_world_size: int = 8,
        intra_host_bw: Optional[int] = None,
        inter_host_bw: Optional[int] = None,
    ) -> None:
        super().__init__(world_size, compute_device, hbm_cap, ddr_cap)
        self._local_world_size = local_world_size
        self._intra_host_bw: int = (
            intra_host_bw if intra_host_bw else INTRA_NODE_BANDWIDTH
        )
        self._inter_host_bw: int = (
            inter_host_bw if inter_host_bw else CROSS_NODE_BANDWIDTH
        )

    @property
    def local_world_size(self) -> int:
        return self._local_world_size

    @property
    def intra_host_bw(self) -> int:
        return self._intra_host_bw

    @property
    def inter_host_bw(self) -> int:
        return self._inter_host_bw
