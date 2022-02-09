#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import tempfile
from typing import Optional, List, Any, Dict

import numpy as np
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import (
    BinaryCriteoUtils,
    InMemoryBinaryCriteoIterDataPipe,
    INT_FEATURE_COUNT,
    CAT_FEATURE_COUNT,
)
from torchrec.datasets.criteo import criteo_kaggle, criteo_terabyte
from torchrec.datasets.test_utils.criteo_test_utils import CriteoTest
from torchrec.datasets.utils import Batch


class CriteoTerabyteTest(CriteoTest):
    def test_single_file(self) -> None:
        with self._create_dataset_tsv() as dataset_pathname:
            dataset = criteo_terabyte((dataset_pathname,))
            for sample in dataset:
                self._validate_sample(sample)
            self.assertEqual(len(list(iter(dataset))), 10)

    def test_multiple_files(self) -> None:
        with contextlib.ExitStack() as stack:
            pathnames = [
                stack.enter_context(self._create_dataset_tsv()) for _ in range(3)
            ]
            dataset = criteo_terabyte(pathnames)
            for sample in dataset:
                self._validate_sample(sample)
            self.assertEqual(len(list(iter(dataset))), 30)


class CriteoKaggleTest(CriteoTest):
    def test_train_file(self) -> None:
        with self._create_dataset_tsv() as path:
            dataset = criteo_kaggle(path)
            for sample in dataset:
                self._validate_sample(sample)
            self.assertEqual(len(list(iter(dataset))), 10)

    def test_test_file(self) -> None:
        with self._create_dataset_tsv(train=False) as path:
            dataset = criteo_kaggle(path)
            for sample in dataset:
                self._validate_sample(sample, train=False)
            self.assertEqual(len(list(iter(dataset))), 10)


class CriteoDataLoaderTest(CriteoTest):
    def _validate_dataloader_sample(
        self,
        sample: Dict[str, List[Any]],  # pyre-ignore[2]
        batch_size: int,
        train: bool = True,
    ) -> None:
        unbatched_samples = [{} for _ in range(self._sample_len(sample))]
        for k, batched_values in sample.items():
            for (idx, value) in enumerate(batched_values):
                unbatched_samples[idx][k] = value
        for sample in unbatched_samples:
            self._validate_sample(sample, train=train)

    def _sample_len(
        self,
        sample: Dict[str, List[Any]],  # pyre-ignore[2]
    ) -> int:
        return len(next(iter(sample.values())))

    def _test_dataloader(
        self,
        num_workers: int = 0,
        batch_size: int = 1,
        num_tsvs: int = 1,
        num_rows_per_tsv: int = 10,
        train: bool = True,
    ) -> None:
        with contextlib.ExitStack() as stack:
            pathnames = [
                stack.enter_context(
                    self._create_dataset_tsv(num_rows=num_rows_per_tsv, train=train)
                )
                for _ in range(num_tsvs)
            ]
            dataset = criteo_terabyte(pathnames)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers
            )
            total_len = 0
            for sample in dataloader:
                sample_len = self._sample_len(sample)
                total_len += sample_len
                self._validate_dataloader_sample(
                    sample, batch_size=batch_size, train=train
                )
            self.assertEqual(total_len, len(list(iter(dataset))))

    def test_multiple_train_workers(self) -> None:
        self._test_dataloader(
            num_workers=4, batch_size=16, num_tsvs=5, num_rows_per_tsv=32
        )

    def test_fewer_tsvs_than_workers(self) -> None:
        self._test_dataloader(
            num_workers=2, batch_size=16, num_tsvs=1, num_rows_per_tsv=16
        )

    def test_single_worker(self) -> None:
        self._test_dataloader(batch_size=16, num_tsvs=2, num_rows_per_tsv=16)


class TestBinaryCriteoUtils(CriteoTest):
    def test_tsv_to_npys(self) -> None:
        num_rows = 10
        with self._create_dataset_tsv(num_rows=num_rows) as in_file:
            out_files = [tempfile.NamedTemporaryFile(delete=False) for _ in range(3)]
            for out_file in out_files:
                out_file.close()

            BinaryCriteoUtils.tsv_to_npys(
                in_file, out_files[0].name, out_files[1].name, out_files[2].name
            )

            dense = np.load(out_files[0].name)
            sparse = np.load(out_files[1].name)
            labels = np.load(out_files[2].name)

            self.assertEqual(dense.shape, (num_rows, INT_FEATURE_COUNT))
            self.assertEqual(dense.dtype, np.float32)
            self.assertEqual(sparse.shape, (num_rows, CAT_FEATURE_COUNT))
            self.assertEqual(sparse.dtype, np.int32)
            self.assertEqual(labels.shape, (num_rows, 1))
            self.assertEqual(labels.dtype, np.int32)

            for out_file in out_files:
                os.remove(out_file.name)

    def test_get_shape_from_npy(self) -> None:
        num_rows = 10
        with self._create_dataset_npys(num_rows=num_rows) as (
            dense_path,
            sparse_path,
            labels_path,
        ):
            dense_shape = BinaryCriteoUtils.get_shape_from_npy(dense_path)
            sparse_shape = BinaryCriteoUtils.get_shape_from_npy(sparse_path)
            labels_shape = BinaryCriteoUtils.get_shape_from_npy(labels_path)
            self.assertEqual(dense_shape, (num_rows, INT_FEATURE_COUNT))
            self.assertEqual(sparse_shape, (num_rows, CAT_FEATURE_COUNT))
            self.assertEqual(labels_shape, (num_rows, 1))

    def test_get_file_idx_to_row_range(self) -> None:
        lengths = [14, 17, 20]
        world_size = 3
        expected = [{0: (0, 13), 1: (0, 2)}, {1: (3, 16), 2: (0, 2)}, {2: (3, 19)}]

        for i in range(world_size):
            self.assertEqual(
                expected[i],
                BinaryCriteoUtils.get_file_idx_to_row_range(
                    lengths=lengths,
                    rank=i,
                    world_size=world_size,
                ),
            )

    def test_load_npy_range(self) -> None:
        num_rows = 10
        start_row = 2
        num_rows_to_select = 4
        with self._create_dataset_npys(
            num_rows=num_rows, generate_sparse=False, generate_labels=False
        ) as (dense_path,):
            full = np.load(dense_path)
            partial = BinaryCriteoUtils.load_npy_range(
                dense_path, start_row=start_row, num_rows=num_rows_to_select
            )
            np.testing.assert_array_equal(
                full[start_row : start_row + num_rows_to_select], partial
            )

    def test_sparse_to_contiguous_ids(self) -> None:
        # Build the day .npy files. 3 days, 3 columns, 9 rows.
        unprocessed_data = [
            np.array([[10, 70, 10], [20, 80, 20], [30, 90, 30]]),  # day 0
            np.array([[20, 70, 40], [30, 80, 50], [40, 90, 60]]),  # day 1
            np.array([[20, 70, 70], [20, 80, 80], [30, 90, 90]]),  # day 2
        ]

        expected_data_no_freq_threshold = [
            np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),  # day 0
            np.array([[3, 2, 5], [4, 3, 6], [5, 4, 7]]),  # day 1
            np.array([[3, 2, 8], [3, 3, 9], [4, 4, 10]]),  # day 2
        ]
        self._validate_sparse_to_contiguous_preproc(
            unprocessed_data, expected_data_no_freq_threshold, 0, 3
        )

        expected_data_freq_threshold_2 = [
            np.array([[1, 2, 1], [2, 3, 1], [3, 4, 1]]),  # day 0
            np.array([[2, 2, 1], [3, 3, 1], [1, 4, 1]]),  # day 1
            np.array([[2, 2, 1], [2, 3, 1], [3, 4, 1]]),  # day 2
        ]
        self._validate_sparse_to_contiguous_preproc(
            unprocessed_data, expected_data_freq_threshold_2, 2, 3
        )

    def _validate_sparse_to_contiguous_preproc(
        self,
        unprocessed_data: List[np.ndarray],
        expected_data: List[np.ndarray],
        freq_threshold: int,
        columns: int,
    ) -> None:
        # Save the unprocessed data to temporary directory.
        temp_input_dir: str
        temp_output_dir: str
        with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
            input_files = []
            for i, data in enumerate(unprocessed_data):
                file = os.path.join(temp_input_dir, f"day_{i}_sparse.npy")
                input_files.append(file)
                np.save(file, data)

            BinaryCriteoUtils.sparse_to_contiguous(
                input_files, temp_output_dir, freq_threshold, columns
            )

            output_files = list(
                map(
                    lambda f: os.path.join(temp_output_dir, f),
                    os.listdir(temp_output_dir),
                )
            )
            output_files.sort()
            for day, file in enumerate(output_files):
                processed_data = np.load(file)
                self.assertTrue(np.array_equal(expected_data[day], processed_data))


class TestInMemoryBinaryCriteoIterDataPipe(CriteoTest):
    def _validate_batch(
        self, batch: Batch, batch_size: int, hashes: Optional[List[int]] = None
    ) -> None:
        self.assertEqual(
            tuple(batch.dense_features.size()), (batch_size, INT_FEATURE_COUNT)
        )
        self.assertEqual(
            tuple(batch.sparse_features.values().size()),
            (batch_size * CAT_FEATURE_COUNT,),
        )
        self.assertEqual(tuple(batch.labels.size()), (batch_size,))
        if hashes is not None:
            hashes_np = np.array(hashes).reshape((CAT_FEATURE_COUNT, 1))
            self.assertTrue(
                np.all(
                    batch.sparse_features.values().reshape(
                        (CAT_FEATURE_COUNT, batch_size)
                    )
                    < hashes_np
                )
            )

    def _test_dataset(
        self, rows_per_file: List[int], batch_size: int, world_size: int
    ) -> None:
        with contextlib.ExitStack() as stack:
            files = [
                stack.enter_context(self._create_dataset_npys(num_rows=num_rows))
                for num_rows in rows_per_file
            ]
            hashes = [i + 1 for i in range(CAT_FEATURE_COUNT)]

            lens = []
            for rank in range(world_size):
                datapipe = InMemoryBinaryCriteoIterDataPipe(
                    dense_paths=[f[0] for f in files],
                    sparse_paths=[f[1] for f in files],
                    labels_paths=[f[2] for f in files],
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                    hashes=hashes,
                )
                datapipe_len = len(datapipe)

                len_ = 0
                for x in datapipe:
                    self._validate_batch(x, batch_size=batch_size)
                    len_ += 1

                # Check that dataset __len__ matches true length.
                self.assertEqual(datapipe_len, len_)
                lens.append(len_)

            # Ensure all ranks' datapipes return the same number of batches.
            self.assertEqual(len(set(lens)), 1)

    def test_dataset_small_files(self) -> None:
        self._test_dataset([1] * 20, 4, 2)

    def test_dataset_random_sized_files(self) -> None:
        random.seed(0)
        self._test_dataset([random.randint(1, 100) for _ in range(100)], 16, 3)
