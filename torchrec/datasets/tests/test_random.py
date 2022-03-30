#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchrec.datasets.random import RandomRecDataset


class RandomDataLoader(unittest.TestCase):
    def test_hash_per_feature_ids_per_feature(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_sizes=[100, 200],
            ids_per_features=[100, 200],
            num_dense=5,
        )

        example = next(iter(dataset))
        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 100)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 200)

    def test_hash_ids_per_feature(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100,
            ids_per_features=[100, 200],
            num_dense=5,
        )

        example = next(iter(dataset))
        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 100)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 200)

    def test_hash_ids(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100,
            ids_per_feature=50,
            num_dense=5,
        )

        example = next(iter(dataset))
        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 50)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 50)

    def test_on_fly_batch_generation(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100,
            ids_per_feature=50,
            num_dense=5,
            num_generated_batches=-1,
        )

        it = iter(dataset)

        example = next(it)
        example = next(it)
        example = next(it)
        example = next(it)

        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 50)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 50)
