#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import time

import numpy as np
import nvtabular as nvt
from torchrec.datasets.criteo import DAYS, DEFAULT_COLUMN_NAMES, DEFAULT_LABEL_NAME, DEFAULT_INT_NAMES, DEFAULT_CAT_NAMES
from utils.dask import setup_dask

dtypes = {c: np.int32 for c in DEFAULT_INT_NAMES + [DEFAULT_LABEL_NAME]}
dtypes.update({c: "hex" for c in DEFAULT_CAT_NAMES})


def convert_tsv_to_parquet(input_path: str, output_base_path: str):
    output_path = os.path.join(output_base_path, "criteo_parquet")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    input_paths = [os.path.join(input_path, f"day_{day}") for day in range(DAYS)]

    tsv_dataset = nvt.Dataset(
        input_paths, 
        engine="csv",
        names=DEFAULT_COLUMN_NAMES,
        part_memory_fraction= 0.1,
        sep="\t",
        dtypes=dtypes,
    )

    tsv_dataset.to_parquet(
        output_path,
        preserve_files=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert criteo tsv to parquet")
    parser.add_argument(
        "--input_path", "-i", dest="input_path", help="Input path containing tsv files"
    )
    parser.add_argument(
        "--output_base_path", "-o", dest="output_base_path", help="Output base path"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    dask_workdir = os.path.join(args.output_base_path, "dask_workdir")
    client = setup_dask(dask_workdir)

    assert os.path.exists(
        args.input_path
    ), f"Input path {args.input_path} does not exist"

    start_time = time.time()
    convert_tsv_to_parquet(args.input_path, args.output_base_path)
    print(f"Conversion from tsv to parquet took {time.time()-start_time:.2f} sec")
