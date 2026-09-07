#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, List, Optional

import torch
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.sharding.twrw_sharding import BaseTwRwEmbeddingSharding
from torchrec.distributed.types import ParameterSharding, ShardingType, ShardMetadata
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingTableConfig

_LOCAL_SIZE: int = 4


class _PlacementResolver:
    """
    `_resolve_placement_ranks` with only the two attributes it reads.

    The real constructor needs a live `dist` world, and `object.__new__` is out
    too: `EmbeddingSharding` is an ABC that `BaseTwRwEmbeddingSharding` does not
    fully implement. Borrowing the function keeps these tests on shipped code.
    """

    _resolve_placement_ranks = BaseTwRwEmbeddingSharding._resolve_placement_ranks

    def __init__(self, local_size: int, is_2D_parallel: bool = False) -> None:
        self._local_size = local_size
        self._is_2D_parallel = is_2D_parallel


def _resolve(
    num_nodes: int,
    plan_ranks: List[int],
    num_shards: Optional[int] = None,
    table_node: int = 0,
    local_size: int = _LOCAL_SIZE,
    is_2D_parallel: bool = False,
) -> List[int]:
    resolver = _PlacementResolver(local_size, is_2D_parallel)
    # pyrefly: ignore[bad-argument-type]
    return resolver._resolve_placement_ranks(
        table_name="table_0",
        num_nodes=num_nodes,
        plan_ranks=plan_ranks,
        num_shards=len(plan_ranks) if num_shards is None else num_shards,
        table_node=table_node,
    )


class ResolvePlacementRanksTest(unittest.TestCase):
    """
    Every rejection in `_resolve_placement_ranks`, plus the two accepted shapes.

    It is the last check before a collective constructor: past it, a
    contradictory `num_nodes` becomes a duplicate bucket destination or an
    out-of-range rank inside an all-to-all, which hangs rather than raises.
    """

    def test_shard_and_rank_counts_must_match(self) -> None:
        """Multi-node only, where `ranks[i]` pairs with `shards[i]`: a mismatch
        reads past the end of `shards` or leaves a block unplaced."""
        for placed, num_shards in ((8, 7), (0, 8)):
            with self.subTest(placed=placed):
                with self.assertRaisesRegex(
                    ValueError, rf"{placed} placement ranks for {num_shards} shards"
                ):
                    _resolve(
                        num_nodes=2,
                        plan_ranks=list(range(placed)),
                        num_shards=num_shards,
                    )

    def test_single_node_shard_shortfall_is_rejected(self) -> None:
        """The single-node branch fills the node from `table_node`, so too few
        shards is a bare IndexError in `_shard`."""
        with self.assertRaisesRegex(ValueError, r"2 shards for a node of 4 ranks"):
            _resolve(num_nodes=1, plan_ranks=[0, 1], num_shards=2)

    def test_num_nodes_below_one_is_rejected(self) -> None:
        """0 makes the width check `num_nodes * local_size` demand zero ranks;
        a negative demands a negative number."""
        for num_nodes in (0, -1):
            with self.subTest(num_nodes=num_nodes):
                with self.assertRaisesRegex(
                    ValueError, rf"num_nodes={num_nodes} must be >= 1"
                ):
                    _resolve(num_nodes=num_nodes, plan_ranks=[0, 1, 2, 3])

    def test_multi_node_under_2d_parallelism_is_rejected(self) -> None:
        """Under 2D `plan_ranks` is global while `_shard` maps it through the
        peer group, so a multi-node placement has no defined meaning yet."""
        with self.assertRaisesRegex(
            ValueError, r"num_nodes=2 is not supported under 2D parallelism"
        ):
            _resolve(
                num_nodes=2,
                plan_ranks=list(range(8)),
                is_2D_parallel=True,
            )

    def test_single_node_placed_wider_than_a_node_is_rejected(self) -> None:
        """The case `num_nodes` exists to disambiguate: 8 ranks at a node width
        of 4 is either a two-node placement or a planner/runtime disagreement
        about node width. Undeclared, it is the second, and inferring the first
        would activate the feature on a table that never asked for it."""
        with self.assertRaisesRegex(
            ValueError,
            r"the plan places it on 8 ranks, more than the 4 in a node",
        ):
            _resolve(num_nodes=1, plan_ranks=list(range(8)))

    def test_rank_count_must_match_num_nodes_times_local_size(self) -> None:
        """The same node-width disagreement from the opt-in side: `num_nodes=2`
        at a runtime node width of 4 needs exactly 8 ranks, under or over."""
        for placed in (6, 12):
            with self.subTest(placed=placed):
                with self.assertRaisesRegex(
                    ValueError,
                    r"num_nodes=2 needs 8 ranks at a node width of 4, but the "
                    rf"plan places it on {placed}",
                ):
                    _resolve(num_nodes=2, plan_ranks=list(range(placed)))

    def test_repeated_rank_is_rejected(self) -> None:
        """Two row blocks on one rank, breaking the `ranks[i]` to `shards[i]`
        pairing. Whole-node reuse is what an oversized `num_nodes` produces:
        `_multi_hosts_partition` selects hosts circularly."""
        for num_nodes, plan_ranks in (
            (2, [0, 1, 2, 2, 4, 5, 6, 7]),
            (2, [0, 1, 2, 3, 0, 1, 2, 3]),
            (3, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]),
        ):
            with self.subTest(plan_ranks=plan_ranks):
                with self.assertRaisesRegex(ValueError, r"repeat a rank"):
                    _resolve(num_nodes=num_nodes, plan_ranks=plan_ranks)

    def test_span_wider_than_num_nodes_is_rejected(self) -> None:
        """The right rank count, all distinct, but spread wider than declared.
        Distinct ranks cannot span fewer, so this is the only shape left once
        the duplicate check above has run."""
        with self.assertRaisesRegex(ValueError, r"8 ranks spread over 3 nodes"):
            _resolve(num_nodes=2, plan_ranks=[0, 1, 2, 3, 4, 5, 8, 9])

    def test_single_node_derives_the_whole_node(self) -> None:
        """The stock placement: the table owns its node's ranks, derived from
        `table_node` rather than echoed from the plan, so an unsorted plan
        still comes back in node order."""
        self.assertEqual(
            _resolve(num_nodes=1, plan_ranks=[8, 9, 10, 11], table_node=2),
            [8, 9, 10, 11],
        )
        self.assertEqual(
            _resolve(num_nodes=1, plan_ranks=[0, 1, 2, 3], table_node=0),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            _resolve(num_nodes=1, plan_ranks=[11, 10, 9, 8], table_node=2),
            [8, 9, 10, 11],
        )

    def test_single_node_accepts_a_first_rank_only_plan(self) -> None:
        """`_shard` only ever read `ranks[0]` for TABLE_ROW_WISE, so a plan
        listing one rank has always been enough; requiring one per shard now
        would reject plans that load today."""
        self.assertEqual(
            _resolve(num_nodes=1, plan_ranks=[8], num_shards=4, table_node=2),
            [8, 9, 10, 11],
        )

    def test_single_node_wider_than_a_node_is_allowed_under_2d(self) -> None:
        """2D is exempt from the width check: `plan_ranks` is a global
        placement that `_shard` remaps, so it legitimately exceeds the sharding
        group's node width. Dropping the guard would break every 2D
        TABLE_ROW_WISE model, which no other case here catches."""
        self.assertEqual(
            _resolve(
                num_nodes=1,
                plan_ranks=list(range(8)),
                table_node=1,
                is_2D_parallel=True,
            ),
            [4, 5, 6, 7],
        )

    def test_multi_node_returns_plan_order_unsorted(self) -> None:
        """`ranks[i]` pairs with `shards[i]`, so sorting would pair a rank with
        a different row range."""
        self.assertEqual(
            _resolve(
                num_nodes=2,
                plan_ranks=[2, 3, 0, 1],
                # Would give a different answer via the single-node branch.
                table_node=1,
                local_size=2,
            ),
            [2, 3, 0, 1],
        )
        self.assertEqual(
            _resolve(num_nodes=3, plan_ranks=[4, 5, 0, 1, 2, 3], local_size=2),
            [4, 5, 0, 1, 2, 3],
        )

        # Ascending comes back unchanged too, so the above is order
        # preservation and not the function reversing.
        self.assertEqual(
            _resolve(num_nodes=2, plan_ranks=[0, 1, 2, 3], local_size=2),
            [0, 1, 2, 3],
        )


class _Sharder:
    """`_shard` with only the attributes it reads; see `_PlacementResolver`."""

    _resolve_placement_ranks = BaseTwRwEmbeddingSharding._resolve_placement_ranks
    _shard = BaseTwRwEmbeddingSharding._shard

    def __init__(self, world_size: int, local_size: int) -> None:
        self._world_size = world_size
        self._local_size = local_size
        self._is_2D_parallel = False
        self._pg = None
        self._env: Any = type("_Env", (), {"output_dtensor": False})()


def _sharding_info(
    name: str,
    rows: int,
    dim: int,
    ranks: List[int],
    num_shards: int,
    num_nodes: Optional[int] = None,
) -> EmbeddingShardingInfo:
    rows_per_shard = rows // num_shards
    return EmbeddingShardingInfo(
        embedding_config=EmbeddingTableConfig(
            num_embeddings=rows,
            embedding_dim=dim,
            name=name,
            feature_names=[f"f_{name}"],
            embedding_names=[f"f_{name}"],
        ),
        param_sharding=ParameterSharding(
            sharding_type=ShardingType.TABLE_ROW_WISE.value,
            compute_kernel="dense",
            ranks=ranks,
            sharding_spec=EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=[rows_per_shard, dim],
                        shard_offsets=[i * rows_per_shard, 0],
                        placement=f"rank:{i}/cpu",
                    )
                    for i in range(num_shards)
                ]
            ),
            num_nodes=num_nodes,
        ),
        param=torch.empty(rows, dim, device="meta"),
    )


class ShardPlacementTest(unittest.TestCase):
    """
    `_shard` itself, which `ResolvePlacementRanksTest` does not reach.

    Without it, an implementation that resolved the placement and then threw it
    away, keeping the loop `_resolve_placement_ranks` replaced, passes every
    other test here.
    """

    def test_single_node_pairs_shards_with_the_derived_node(self) -> None:
        """Row block `i` lands on the `i`th rank of the derived node."""
        sharder = _Sharder(world_size=8, local_size=4)
        # One rank in the plan and four shards: all `_shard` ever read.
        info = _sharding_info("t", rows=400, dim=8, ranks=[4], num_shards=4)
        # pyrefly: ignore[bad-argument-type]
        per_rank = sharder._shard([info])

        placed = {r: t for r, t in enumerate(per_rank) if t}
        self.assertEqual(sorted(placed), [4, 5, 6, 7])
        self.assertEqual(
            [
                none_throws(placed[r][0].local_metadata).shard_offsets[0]
                for r in sorted(placed)
            ],
            [0, 100, 200, 300],
        )

    def test_multi_node_is_refused_until_the_forward_path_exists(self) -> None:
        """The placement resolves, but nothing bucketizes ids across nodes or
        sums the per-node partials yet."""
        sharder = _Sharder(world_size=8, local_size=4)
        info = _sharding_info(
            "t", rows=800, dim=8, ranks=list(range(8)), num_shards=8, num_nodes=2
        )
        with self.assertRaisesRegex(NotImplementedError, r"num_nodes=2"):
            # pyrefly: ignore[bad-argument-type]
            sharder._shard([info])

    def test_multi_node_shard_rank_mismatch_raises(self) -> None:
        """`_shard` surfaces the resolver's rejections rather than swallowing
        them."""
        sharder = _Sharder(world_size=8, local_size=4)
        info = _sharding_info(
            "t", rows=800, dim=8, ranks=list(range(8)), num_shards=8, num_nodes=2
        )
        info.param_sharding.ranks = list(range(7))
        with self.assertRaisesRegex(ValueError, r"7 placement ranks for 8 shards"):
            # pyrefly: ignore[bad-argument-type]
            sharder._shard([info])
