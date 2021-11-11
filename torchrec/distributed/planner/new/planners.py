#!/usr/bin/env python3

from typing import Dict, Optional, List

import torch.distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import (
    invoke_on_rank_and_broadcast_result,
)
from torchrec.distributed.planner.new.calculators import EmbeddingWTCostCalculator
from torchrec.distributed.planner.new.enumerators import ShardingEnumerator
from torchrec.distributed.planner.new.partitioners import GreedyCostPartitioner
from torchrec.distributed.planner.new.placers import EmbeddingPlacer
from torchrec.distributed.planner.new.rankers import DepthRanker, TotalWorkRanker
from torchrec.distributed.planner.new.stats import EmbeddingShardingStats
from torchrec.distributed.planner.new.types import (
    PlannerConstraints,
    InputStats,
    Enumerator,
    Placer,
    Ranker,
    Calculator,
    Partitioner,
    Topology,
    Stats,
)
from torchrec.distributed.types import (
    ShardingPlan,
    ShardingPlanner,
    ModuleSharder,
)


def _reserve_storage_percentage(topology: Topology, percent: float) -> None:
    for device in topology.devices:
        device.storage.hbm = int((1 - percent) * device.storage.hbm)
        device.storage.ddr = int((1 - percent) * device.storage.ddr)


class EmbeddingShardingPlanner(ShardingPlanner):
    def __init__(
        self,
        topology: Topology,
        components: Optional[Dict[str, object]] = None,
        constraints: Optional[Dict[str, PlannerConstraints]] = None,
        input_stats: Optional[Dict[str, InputStats]] = None,
    ) -> None:
        self._topology = topology
        self._input_stats = input_stats

        if components is None:
            components = {}

        self._enumerator: Enumerator = components.get(
            "enumerator",
            ShardingEnumerator(
                topology=topology,
                constraints=constraints,
                input_stats=input_stats,
            ),
        )
        self._calculator: Calculator = components.get(
            "calculator", EmbeddingWTCostCalculator(topology=topology)
        )
        self._partitioners: List[Partitioner] = components.get(
            "paritioners", [GreedyCostPartitioner()]
        )
        self._rankers: List[Ranker] = components.get(
            "rankers", [DepthRanker(), TotalWorkRanker()]
        )

        self._placer: Placer = components.get(
            "placer",
            EmbeddingPlacer(
                topology=topology,
                partitioners=self._partitioners,
                rankers=self._rankers,
            ),
        )
        self._stats: Stats = components.get("stats", EmbeddingShardingStats())

    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        pg: dist.ProcessGroup,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        return invoke_on_rank_and_broadcast_result(
            pg,
            0,
            self.plan,
            module,
            sharders,
        )

    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:

        # TODO: actually estimate storage of non-sharded modules:
        _reserve_storage_percentage(self._topology, 0.40)

        sharding_options = self._enumerator.run(module=module, sharders=sharders)
        self._calculator.run(sharding_options=sharding_options)
        sharding_plan = self._placer.run(sharding_options=sharding_options)
        self._stats.run(
            sharding_plan=sharding_plan,
            topology=self._topology,
            placer_stats=self._placer.stats,
            input_stats=self._input_stats,
        )

        return sharding_plan
