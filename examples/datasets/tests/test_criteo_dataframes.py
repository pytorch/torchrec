#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torcharrow as ta
from torch.utils.data import IterDataPipe
from torchrec.datasets.test_utils.criteo_test_utils import CriteoTest

from ..criteo_dataframes import criteo_dataframes_from_tsv


class CriteoDataFramesTest(CriteoTest):
    BATCH_SIZE = 4

    def test_single_file(self) -> None:
        with self._create_dataset_tsv() as dataset_pathname:
            dataset = criteo_dataframes_from_tsv(
                dataset_pathname, batch_size=self.BATCH_SIZE
            )

            self._validate_dataset(dataset, 10)

    def test_multiple_files(self) -> None:
        with contextlib.ExitStack() as stack:
            pathnames = [
                stack.enter_context(self._create_dataset_tsv()) for _ in range(3)
            ]
            dataset = criteo_dataframes_from_tsv(pathnames, batch_size=self.BATCH_SIZE)

            self._validate_dataset(dataset, 30)

    def _validate_dataset(
        self, dataset: IterDataPipe, expected_total_length: int
    ) -> None:
        last_batch = False
        total_length = 0

        for df in dataset:
            self.assertFalse(last_batch)
            self.assertTrue(isinstance(df, ta.DataFrame))
            self.assertLessEqual(len(df), self.BATCH_SIZE)

            total_length += len(df)
            if len(df) < self.BATCH_SIZE:
                last_batch = True

            self._validate_dataframe(df)
        self.assertEqual(total_length, expected_total_length)

    def _validate_dataframe(self, df: ta.DataFrame, train: bool = True) -> None:
        if train:
            self.assertEqual(len(df.columns), 3)
            labels = df["labels"]
            for label_val in labels:
                self.assertTrue(
                    self.LABEL_VAL_RANGE[0] <= label_val <= self.LABEL_VAL_RANGE[1]
                )
        else:
            self.assertEqual(len(df.columns), 2)

        # Validations for both train and test
        dense_features = df["dense_features"]
        for idx in range(self.INT_FEATURE_COUNT):
            int_vals = dense_features[f"int_{idx}"]
            for int_val in int_vals:
                self.assertTrue(
                    self.INT_VAL_RANGE[0] <= int_val <= self.INT_VAL_RANGE[1]
                )

        sparse_features = df["sparse_features"]
        for idx in range(self.CAT_FEATURE_COUNT):
            cat_vals = sparse_features[f"cat_{idx}"]
            for cat_val in cat_vals:
                # stored as int32
                self.assertTrue(-(2**31) <= cat_val <= 2**31 - 1)
