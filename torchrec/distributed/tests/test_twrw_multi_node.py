#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from typing import Dict, List, Optional, Tuple

import torch
from torchrec.distributed.embedding_sharding import bucketize_kjt_before_all2all
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.sharding.twrw_sharding import (
    _BucketGroup,
    _FeatureBucket,
    _FeatureLayout,
    _InputPlan,
    TwRwSparseFeaturesDist,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.types import DataType


def _all_feature_buckets(
    num_buckets_per_feature: List[int],
) -> List[_FeatureBucket]:
    return [
        _FeatureBucket(feature, bucket)
        for feature, num_buckets in enumerate(num_buckets_per_feature)
        for bucket in range(num_buckets)
    ]


def _bucket_values(kjt: KeyedJaggedTensor) -> List[List[int]]:
    """Values in each key of a one-sample-per-key bucketize output."""
    values = kjt.values().tolist()
    out: List[List[int]] = []
    start = 0
    for length in kjt.lengths().tolist():
        out.append(values[start : start + length])
        start += length
    return out


def _feature_names(table: str, num_features: int) -> List[str]:
    return (
        [table] if num_features == 1 else [f"{table}f{i}" for i in range(num_features)]
    )


def _layout(
    tables_per_rank: List[List[str]],
    dim_by_table: Dict[str, int],
    local_size: int = 1,
    features_by_table: Optional[Dict[str, int]] = None,
) -> _FeatureLayout:
    """
    Builds a layout from grouped configs, one feature per table named after it
    unless `features_by_table` asks for more.

    Defaults to one rank per node, so `tables_per_rank` then reads as the
    tables each node holds.
    """
    table_placement_ranks: Dict[str, List[int]] = {}
    for rank, tables in enumerate(tables_per_rank):
        for table in tables:
            table_placement_ranks.setdefault(table, []).append(rank)
    names_by_table = {
        table: _feature_names(table, (features_by_table or {}).get(table, 1))
        for tables in tables_per_rank
        for table in tables
    }
    return _FeatureLayout(
        grouped_configs_per_rank=[
            [
                GroupedEmbeddingConfig(
                    data_type=DataType.FP32,
                    pooling=PoolingType.SUM,
                    is_weighted=False,
                    has_feature_processor=False,
                    compute_kernel=EmbeddingComputeKernel.DENSE,
                    embedding_tables=[
                        ShardedEmbeddingTable(
                            name=table,
                            num_embeddings=16,
                            embedding_dim=dim_by_table[table],
                            local_cols=dim_by_table[table],
                            feature_names=names_by_table[table],
                            embedding_names=names_by_table[table],
                        )
                        for table in tables
                    ],
                )
            ]
            for tables in tables_per_rank
        ],
        table_placement_ranks=table_placement_ranks,
        local_size=local_size,
    )


def _total_dim(tables_per_node: List[List[str]], dim_by_table: Dict[str, int]) -> int:
    return sum(dim_by_table[table] for tables in tables_per_node for table in tables)


class InputPlanTest(unittest.TestCase):
    def test_single_count_is_the_ungrouped_layout(self) -> None:
        """When every feature wants the same bucket count (every single-node
        TABLE_ROW_WISE and GRID_SHARD instance), grouping is a no-op and the
        index of `(feature, bucket)` is still `bucket * num_features +
        feature`, which the pre-existing staggered shuffle assumes."""
        num_features, buckets = 5, 4
        feature_buckets = _all_feature_buckets([buckets] * num_features)
        plan = _InputPlan(
            [buckets] * num_features, feature_buckets, default_num_buckets=buckets
        )

        self.assertEqual(
            plan.bucket_groups, [_BucketGroup(buckets, list(range(num_features)))]
        )
        for index, fb in zip(plan.rank_feature_bucket_indices, feature_buckets):
            self.assertEqual(index, fb.bucket * num_features + fb.feature)

    def test_uniform_order_matches_the_staggered_shuffle(self) -> None:
        """The single-node path is the general path: `local_size` buckets each
        and an order that reproduces the original staggered shuffle."""
        features_per_rank, local_size = [3, 3, 2, 2], 2
        num_features = 5
        plan = _InputPlan.uniform(num_features, features_per_rank, local_size)

        node_offsets = [0, 3, 5]
        self.assertEqual(
            plan.rank_feature_bucket_indices,
            [
                bucket * num_features + feature
                for node in range(2)
                for bucket in range(local_size)
                for feature in range(node_offsets[node], node_offsets[node + 1])
            ],
        )

    def test_mixed_counts_assign_every_index_by_feature_bucket(self) -> None:
        """Four features of single-node tables at 2 buckets, two of a
        multi-node table at 4."""
        num_buckets_per_feature = [2, 4, 2, 2, 4, 2]
        feature_buckets = _all_feature_buckets(num_buckets_per_feature)
        plan = _InputPlan(
            num_buckets_per_feature, feature_buckets, default_num_buckets=2
        )

        # Groups in order of first use, every feature in exactly one.
        self.assertEqual(
            plan.bucket_groups, [_BucketGroup(2, [0, 2, 3, 5]), _BucketGroup(4, [1, 4])]
        )

        # Every (feature, bucket) has one unique index in the concatenated
        # bucketized KJT.
        self.assertEqual(
            sorted(plan.rank_feature_bucket_indices),
            list(range(len(feature_buckets))),
        )

    def test_no_features_still_yields_one_group(self) -> None:
        """An empty instance keeps `local_size` rather than yielding no group."""
        plan = _InputPlan([], [], default_num_buckets=8)
        self.assertEqual(plan.bucket_groups, [_BucketGroup(8, [])])
        self.assertEqual(plan.rank_feature_bucket_indices, [])


class BucketizeByBucketCountTest(unittest.TestCase):
    """
    A single-node table must bucketize exactly as it would without a
    multi-node table in the same sharding instance.

    `block_bucketize_sparse_features` takes per-feature block sizes but one
    scalar bucket count, and that scalar routes ids at or above
    `block_size * num_buckets` to rank `id % num_buckets` — a path fbgemm
    documents as supported. One shared call at the largest count would
    therefore move the out-of-range ids of every table that asked for fewer.
    """

    # One feature lives on a 2-rank node; the other spans both nodes.
    HASH_SIZE: int = 16
    SINGLE_NODE_BUCKETS: int = 2
    MULTI_NODE_BUCKETS: int = 4
    # 20 and 21 are at or above `block_size * num_buckets` at this feature's own
    # count, so they take the out-of-range path.
    SINGLE_NODE_IDS: List[int] = [0, 9, 20, 21]
    MULTI_NODE_IDS: List[int] = [1, 5, 11, 15]

    def _kjt(self, keys: List[str], ids_per_key: List[List[int]]) -> KeyedJaggedTensor:
        values: List[int] = [i for ids in ids_per_key for i in ids]
        return KeyedJaggedTensor(
            keys=keys,
            values=torch.tensor(values, dtype=torch.int64),
            lengths=torch.tensor([len(ids) for ids in ids_per_key], dtype=torch.int32),
        )

    def _block_sizes(self, num_buckets_per_feature: List[int]) -> torch.Tensor:
        return torch.tensor(
            [
                math.ceil(self.HASH_SIZE / buckets)
                for buckets in num_buckets_per_feature
            ],
            dtype=torch.int64,
        )

    def _grouped(
        self, num_buckets_per_feature: List[int], ids_per_key: List[List[int]]
    ) -> Tuple[List[List[int]], Dict[_FeatureBucket, int]]:
        """Runs the production grouping + bucketize over a mixed KJT."""
        feature_buckets = _all_feature_buckets(num_buckets_per_feature)
        plan = _InputPlan(
            num_buckets_per_feature,
            feature_buckets,
            default_num_buckets=min(num_buckets_per_feature),
        )
        index_by_feature_bucket = dict(
            zip(feature_buckets, plan.rank_feature_bucket_indices)
        )
        group_feature_indices = [
            feature for group in plan.bucket_groups for feature in group.features
        ]
        bucketizer = TwRwSparseFeaturesDist.__new__(TwRwSparseFeaturesDist)
        torch.nn.Module.__init__(bucketizer)
        bucketizer._bucket_groups = plan.bucket_groups
        bucketizer._group_feature_indices_tensor = torch.tensor(
            group_feature_indices, dtype=torch.int32
        )
        bucketizer._feature_block_sizes_tensor = self._block_sizes(
            [num_buckets_per_feature[feature] for feature in group_feature_indices]
        )
        out = bucketizer._bucketize(
            self._kjt(
                [f"f{i}" for i in range(len(num_buckets_per_feature))], ids_per_key
            ),
            bucketize_pos=False,
        )
        return _bucket_values(out), index_by_feature_bucket

    def _alone(
        self,
        num_buckets: int,
        ids: List[int],
        block_size_for: Optional[int] = None,
    ) -> List[List[int]]:
        """
        The reference: that feature bucketized on its own.

        `block_size_for` is the bucket count whose block size to use, so a
        caller can vary the scalar while holding the block size fixed;
        production passes block sizes per feature and only the count is shared.
        """
        out = bucketize_kjt_before_all2all(
            self._kjt(["f"], [ids]),
            num_buckets=num_buckets,
            block_sizes=self._block_sizes(
                [num_buckets if block_size_for is None else block_size_for]
            ),
            output_permute=False,
            bucketize_pos=False,
        )[0]
        return _bucket_values(out)

    def test_in_range_ids_ignore_the_bucket_count(self) -> None:
        """`rank = id / block_size`, `row = id % block_size`: no `num_buckets`
        anywhere, so both counts give the same answer, which is why raising
        the count looked safe."""
        in_range = [0, 9]
        self.assertEqual(self._alone(self.SINGLE_NODE_BUCKETS, in_range), [[0], [1]])
        self.assertEqual(
            self._alone(
                self.MULTI_NODE_BUCKETS,
                in_range,
                block_size_for=self.SINGLE_NODE_BUCKETS,
            )[: self.SINGLE_NODE_BUCKETS],
            [[0], [1]],
        )

    def test_shared_max_bucket_count_moves_out_of_range_ids(self) -> None:
        """The defect with no grouping in sight: one feature, one block size,
        only the scalar changes."""
        at_own_count = self._alone(self.SINGLE_NODE_BUCKETS, self.SINGLE_NODE_IDS)
        at_shared_max = self._alone(
            self.MULTI_NODE_BUCKETS,
            self.SINGLE_NODE_IDS,
            block_size_for=self.SINGLE_NODE_BUCKETS,
        )[: self.SINGLE_NODE_BUCKETS]
        self.assertNotEqual(
            at_own_count,
            at_shared_max,
            "if these matched, the bucket count would not affect routing and "
            "the per-count grouping would be unnecessary",
        )

    def test_grouping_leaves_the_single_node_table_untouched(self) -> None:
        """The single-node table's feature is unaffected by the multi-node
        table's."""
        bucket_values, index_by_feature_bucket = self._grouped(
            [self.SINGLE_NODE_BUCKETS, self.MULTI_NODE_BUCKETS],
            [self.SINGLE_NODE_IDS, self.MULTI_NODE_IDS],
        )

        single_node = [
            bucket_values[index_by_feature_bucket[_FeatureBucket(0, bucket)]]
            for bucket in range(self.SINGLE_NODE_BUCKETS)
        ]
        self.assertEqual(
            single_node,
            self._alone(self.SINGLE_NODE_BUCKETS, self.SINGLE_NODE_IDS),
        )

        multi_node = [
            bucket_values[index_by_feature_bucket[_FeatureBucket(1, bucket)]]
            for bucket in range(self.MULTI_NODE_BUCKETS)
        ]
        self.assertEqual(
            multi_node,
            self._alone(self.MULTI_NODE_BUCKETS, self.MULTI_NODE_IDS),
        )
        self.assertEqual(
            sum(len(bucket) for bucket in single_node + multi_node),
            len(self.SINGLE_NODE_IDS) + len(self.MULTI_NODE_IDS),
        )

    def test_single_group_matches_the_ungrouped_call(self) -> None:
        """A single bucket count takes the original bucketization path."""
        ids = [self.SINGLE_NODE_IDS, [2, 7, 30]]
        bucket_values, index_by_feature_bucket = self._grouped(
            [self.SINGLE_NODE_BUCKETS] * 2, ids
        )

        ungrouped = _bucket_values(
            bucketize_kjt_before_all2all(
                self._kjt(["f0", "f1"], ids),
                num_buckets=self.SINGLE_NODE_BUCKETS,
                block_sizes=self._block_sizes([self.SINGLE_NODE_BUCKETS] * 2),
                output_permute=False,
                bucketize_pos=False,
            )[0]
        )
        self.assertEqual(bucket_values, ungrouped)


class FeatureLayoutTest(unittest.TestCase):
    # (local_size, features of each table on each node). No table spans nodes,
    # so every case must route exactly as the pre-existing single-node path,
    # which GRID_SHARD still takes.
    SINGLE_NODE_TOPOLOGIES: List[Tuple[int, List[List[int]]]] = [
        (1, [[1]]),  # one rank holding one table
        (2, [[1, 1]]),  # one node of two ranks
        (1, [[1], [1]]),  # two nodes of one rank
        (2, [[1, 1], [1]]),  # uneven table counts per node
        (2, [[1, 1, 1], []]),  # a node holding nothing
        (2, [[2], [1]]),  # a table with two features
        (3, [[3, 1], [2], [1]]),  # mixed feature counts, three nodes
        (2, [[], []]),  # no tables at all
    ]

    def test_single_node_plans_match_the_uniform_plan(self) -> None:
        for local_size, features_per_table_by_node in self.SINGLE_NODE_TOPOLOGIES:
            with self.subTest(local_size=local_size, nodes=features_per_table_by_node):
                features_by_table = {
                    f"n{node}t{table}": num_features
                    for node, tables in enumerate(features_per_table_by_node)
                    for table, num_features in enumerate(tables)
                }
                tables_per_rank = [
                    [f"n{node}t{table}" for table in range(len(tables))]
                    for node, tables in enumerate(features_per_table_by_node)
                    for _ in range(local_size)
                ]
                features_per_rank = [
                    sum(features_by_table[table] for table in tables)
                    for tables in tables_per_rank
                ]

                feature_layout = _layout(
                    tables_per_rank,
                    dict.fromkeys(features_by_table, 4),
                    local_size,
                    features_by_table,
                )
                input_plan = feature_layout.input_plan
                uniform_input_plan = _InputPlan.uniform(
                    sum(features_by_table.values()), features_per_rank, local_size
                )

                self.assertEqual(
                    input_plan.bucket_groups, uniform_input_plan.bucket_groups
                )
                self.assertEqual(
                    input_plan.rank_feature_bucket_indices,
                    uniform_input_plan.rank_feature_bucket_indices,
                )

    def test_derives_the_feature_list_and_both_orderings(self) -> None:
        """2 nodes of 2 ranks: `a` on node 0 only, `b` on both."""
        feature_layout = _layout(
            [["a", "b"], ["a", "b"], ["b"], ["b"]],
            {"a": 4, "b": 6},
            local_size=2,
        )

        # Each feature once, in the order the ranks first reach it, cut into as
        # many buckets as it has row blocks.
        self.assertEqual(
            [
                (feature.feature_name, feature.num_buckets, feature.embedding_dim)
                for feature in feature_layout.features
            ],
            [("a", 2, 4), ("b", 4, 6)],
        )
        # Node 0's segment carries a and b, node 1's carries b again.
        self.assertEqual(feature_layout._feature_indices, [0, 1, 1])
        # One feature bucket per rank per table it holds, in rank order:
        # (a, 0) (b, 0) | (a, 1) (b, 1) | (b, 2) | (b, 3), resolved against a
        # bucketize output of a's 2 buckets then b's 4.
        self.assertEqual(
            feature_layout.input_plan.rank_feature_bucket_indices,
            [0, 2, 1, 3, 4, 5],
        )


class FeatureLayoutCombineTest(unittest.TestCase):
    """
    Checks the combine against a reference that adds one partial at a time,
    independent of the coalescing production does. Each node can hold a
    different set of features, so a feature's partials are not necessarily
    adjacent.
    """

    # (tables each node holds, embedding dim per table), one rank per node.
    NODE_PLACEMENTS: List[Tuple[List[List[str]], Dict[str, int]]] = [
        ([["a"]], {"a": 4}),  # nothing repeats
        ([["a"], ["a"]], {"a": 4}),  # one table on both nodes
        ([["a", "b"], ["b"]], {"a": 3, "b": 2}),  # repeat at the end
        ([["a", "b"], ["a"]], {"a": 3, "b": 2}),  # repeat at the start
        ([["a", "b", "c"], ["b"]], {"a": 4, "b": 6, "c": 5}),  # repeat in the middle
        ([["a", "b"], ["a", "b"]], {"a": 3, "b": 2}),  # a whole node repeats
        ([["a", "b"], ["b", "a"]], {"a": 3, "b": 2}),  # reordered, defeats coalescing
        ([["a"], ["a"], ["a"]], {"a": 2}),  # three nodes
        ([["a", "b"], [], ["b"]], {"a": 1, "b": 1}),  # a node holding nothing
        ([["a", "b", "c"], ["a", "c"]], {"a": 2, "b": 3, "c": 1}),  # a subset repeats
    ]

    @staticmethod
    def _reference(
        tables_per_node: List[List[str]],
        dim_by_table: Dict[str, int],
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Adds one partial at a time into the final feature order."""
        destination_by_table: Dict[str, int] = {}
        offset = 0
        for table in dict.fromkeys(
            table for tables in tables_per_node for table in tables
        ):
            destination_by_table[table] = offset
            offset += dim_by_table[table]
        expected = torch.zeros(tensor.shape[0], offset)
        source = 0
        for tables in tables_per_node:
            for table in tables:
                destination = destination_by_table[table]
                embedding_dim = dim_by_table[table]
                expected[:, destination : destination + embedding_dim] += tensor[
                    :, source : source + embedding_dim
                ]
                source += embedding_dim
        return expected

    @staticmethod
    def _combine(feature_layout: _FeatureLayout, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the combine, which is skipped when no feature repeats."""
        callback = feature_layout.combine_callback()
        return tensor if callback is None else callback(tensor)

    def _assert_matches_reference(
        self, tables_per_node: List[List[str]], dim_by_table: Dict[str, int]
    ) -> None:
        feature_layout = _layout(tables_per_node, dim_by_table)
        total_dim = _total_dim(tables_per_node, dim_by_table)
        tensor = torch.arange(3 * total_dim, dtype=torch.float32).view(3, total_dim)
        torch.testing.assert_close(
            self._combine(feature_layout, tensor),
            self._reference(tables_per_node, dim_by_table, tensor),
        )

    def test_variable_batch_combine_flattens_per_feature_batches(self) -> None:
        """b repeats, so its two partials carry b's batch size and are summed
        inside a flattened tensor."""
        feature_layout = _layout([["a", "b", "c"], ["b"]], {"a": 2, "b": 3, "c": 1})

        batch_size_per_partial, callback = feature_layout.variable_batch_combine(
            [2, 1, 3]
        )
        self.assertEqual(batch_size_per_partial, [2, 1, 3, 1])

        tensor = torch.arange(13, dtype=torch.float32)
        expected = torch.cat((tensor[:4], tensor[4:7] + tensor[10:13], tensor[7:10]))
        torch.testing.assert_close(none_throws(callback)(tensor), expected)

    def test_variable_batch_rejects_a_wrong_length(self) -> None:
        """The batch sizes come from the sharding context, so a count that
        disagrees with the layout would silently misalign the flattened
        tensor."""
        feature_layout = _layout([["a", "b"]], {"a": 3, "b": 5})

        with self.assertRaisesRegex(ValueError, r"expected 2 feature batch sizes"):
            feature_layout.variable_batch_combine([1, 2, 3])

    def test_gradient_reaches_every_partial(self) -> None:
        tables_per_node = [["a", "b"], ["b"]]
        dim_by_table = {"a": 3, "b": 2}
        feature_layout = _layout(tables_per_node, dim_by_table)

        total_dim = _total_dim(tables_per_node, dim_by_table)
        # `requires_grad` after the view, so `tensor` is a leaf and keeps a grad.
        tensor = torch.arange(2 * total_dim, dtype=torch.float32).view(2, total_dim)
        tensor.requires_grad_(True)
        self._combine(feature_layout, tensor).sum().backward()
        torch.testing.assert_close(none_throws(tensor.grad), torch.ones(2, total_dim))

    def test_matches_reference_across_node_placements(self) -> None:
        for tables_per_node, dim_by_table in self.NODE_PLACEMENTS:
            with self.subTest(nodes=tables_per_node):
                self._assert_matches_reference(tables_per_node, dim_by_table)

    def test_single_node_layout_is_a_pass_through(self) -> None:
        feature_layout = _layout([["a", "b"]], {"a": 3, "b": 5})

        # Nothing to sum, so no callback is attached to either awaitable and
        # the variable batch sizes pass through.
        self.assertIsNone(feature_layout.combine_callback())
        batch_size_per_partial, callback = feature_layout.variable_batch_combine([2, 3])
        self.assertEqual(batch_size_per_partial, [2, 3])
        self.assertIsNone(callback)
