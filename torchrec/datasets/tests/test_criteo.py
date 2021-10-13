#!/usr/bin/env python3

import contextlib
import csv
import os
import random
import tempfile
import unittest
from typing import List, Any, Dict, Generator

from torch.utils.data import DataLoader
from torchrec.datasets.criteo import criteo_kaggle, criteo_terabyte


class _CriteoTest(unittest.TestCase):
    INT_FEATURE_COUNT = 13
    CAT_FEATURE_COUNT = 26

    LABEL_VAL_RANGE = (0, 1)
    INT_VAL_RANGE = (0, 100)
    CAT_VAL_RANGE = (0, 1000)

    @classmethod
    @contextlib.contextmanager
    def _create_dataset_tsv(
        cls,
        num_rows: int = 10,
        train: bool = True,
        filename: str = "criteo",
    ) -> Generator[str, None, None]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "w") as f:
                rows = []
                for _ in range(num_rows):
                    row = []
                    if train:
                        row.append(str(random.randint(*cls.LABEL_VAL_RANGE)))
                    row += [
                        *(
                            str(random.randint(*cls.INT_VAL_RANGE))
                            for _ in range(cls.INT_FEATURE_COUNT)
                        ),
                        *(
                            (
                                "%x"
                                % abs(hash(str(random.randint(*cls.CAT_VAL_RANGE))))
                            ).zfill(8)[:8]
                            for _ in range(cls.CAT_FEATURE_COUNT)
                        ),
                    ]
                    rows.append(row)
                # pyre-ignore[6]
                cf = csv.writer(f, delimiter="\t")
                cf.writerows(rows)
            yield path

    def _validate_sample(self, sample: Dict[str, Any], train: bool = True) -> None:
        if train:
            self.assertEqual(
                len(sample), self.INT_FEATURE_COUNT + self.CAT_FEATURE_COUNT + 1
            )
            label_val = sample["label"]
            self.assertTrue(
                self.LABEL_VAL_RANGE[0] <= label_val <= self.LABEL_VAL_RANGE[1]
            )
        else:
            self.assertEqual(
                len(sample), self.INT_FEATURE_COUNT + self.CAT_FEATURE_COUNT
            )
        for idx in range(self.INT_FEATURE_COUNT):
            int_val = sample[f"int_{idx}"]
            self.assertTrue(self.INT_VAL_RANGE[0] <= int_val <= self.INT_VAL_RANGE[1])
        for idx in range(self.CAT_FEATURE_COUNT):
            cat_val = int(sample[f"cat_{idx}"], 16)
            self.assertTrue(0 <= cat_val <= 16 ** 8 - 1)


class CriteoTerabyteTest(_CriteoTest):
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


class CriteoKaggleTest(_CriteoTest):
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


class CriteoDataLoaderTest(_CriteoTest):
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
