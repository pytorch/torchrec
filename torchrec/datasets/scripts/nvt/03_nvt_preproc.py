#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import shutil
import time
from contextlib import ExitStack
from typing import List, Iterable

import numpy as np
import numpy.lib.format as fmt
import nvtabular as nvt
import pyarrow.parquet as pq
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
    INT_FEATURE_COUNT,
    CAT_FEATURE_COUNT,
    DAYS,
)


class NumpyWriter(object):
    def __init__(self, path: str, dtype: np.dtype, shape: Iterable[int]):
        assert len(shape) == 2, "Only works with two dimensional arrays"
        self.path = path
        self.dtype = dtype
        self.shape = shape
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "wb")
        header = {
            "descr": fmt.dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": self.shape,
        }
        fmt.write_array_header_2_0(self.f, header)
        return self

    def append(self, x: np.array):
        self.f.write(x.tobytes("C"))

    def __exit__(self, *args):
        self.f.close()


def get_total_entries(files: List[str]) -> int:
    return sum([pq.read_table(f, columns=[]).num_rows for f in files])


def create_in_mem(input_path: str, output_path: str, day: int) -> None:
    files = glob.glob(os.path.join(input_path, f"day_{day}", "*.parquet"))

    entries = get_total_entries(files)

    input_dataset = nvt.Dataset(files)

    with ExitStack() as stack:
        dense_writer = stack.enter_context(
            NumpyWriter(
                os.path.join(output_path, f"day_{day}_dense.npy"),
                np.dtype(np.float32),
                (entries, INT_FEATURE_COUNT),
            )
        )
        sparse_writer = stack.enter_context(
            NumpyWriter(
                os.path.join(output_path, f"day_{day}_sparse.npy"),
                np.dtype(np.int32),
                (entries, CAT_FEATURE_COUNT),
            )
        )
        label_writer = stack.enter_context(
            NumpyWriter(
                os.path.join(output_path, f"day_{day}_labels.npy"),
                np.dtype(np.int32),
                (entries, 1),
            )
        )

        for t in input_dataset.to_iter():
            dense_writer.append(t[DEFAULT_INT_NAMES].to_numpy())
            sparse_writer.append(t[DEFAULT_CAT_NAMES].to_numpy())
            label_writer.append(t[DEFAULT_LABEL_NAME].to_numpy())


def parse_args():
    parser = argparse.ArgumentParser(description="Create criteo in memory dataset")
    parser.add_argument("--base_path", "-b", dest="base_path", help="Base path")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    input_path = os.path.join(args.base_path, "criteo_preproc")

    assert os.path.exists(input_path), f"Input path {input_path} does not exist"

    output_path = os.path.join(args.base_path, "in_mem")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    start_time = time.time()
    for day in range(DAYS):
        create_in_mem(input_path, output_path, day)
    print(f"Processing took {time.time()-start_time:.2f} sec")
