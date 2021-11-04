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
from torchrec.distributed.planner.new.rankers import FlatRanker
from torchrec.distributed.planner.new.stats import EmbeddingShardingStats
from torchrec.distributed.planner.new.types import (
    PlannerConstraints,
    InputStats,
    Enumerator,
    Placer,
    Ranker,
    CostCalc,
    Partitioner,
    Topology,
    Stats,
)
from torchrec.distributed.types import (
    ShardingPlan,
    ShardingPlanner,
    ModuleSharder,
)


class EmbeddingShardingPlanner(ShardingPlanner):
    def __init__(
        self,
        topology: Topology,
        batch_size: int = 512,
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
                batch_size=batch_size,
            ),
        )
        self._calculator: CostCalc = components.get(
            "calculator", EmbeddingWTCostCalculator(topology=topology)
        )
        self._ranker: Ranker = components.get(
            "ranker", FlatRanker(calculator=self._calculator)
        )
        self._partitioner: Partitioner = components.get(
            "paritioner", GreedyCostPartitioner()
        )
        self._placer: Placer = components.get(
            "placer", EmbeddingPlacer(topology=topology, partitioner=self._partitioner)
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
        sharding_options = self._enumerator.run(module=module, sharders=sharders)
        rank_stack = self._ranker.run(sharding_options=sharding_options)
        sharding_plan = self._placer.run(rank_stack=rank_stack)

        self._stats.run(
            sharding_plan=sharding_plan,
            topology=self._topology,
            placer_stats=self._placer.stats,
            input_stats=self._input_stats,
        )

        return sharding_plan
