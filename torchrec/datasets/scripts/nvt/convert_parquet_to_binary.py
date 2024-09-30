#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import tqdm
from joblib import delayed, Parallel
from utils.criteo_constant import (
    DEFAULT_CAT_NAMES,
    DEFAULT_COLUMN_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
)


def process_file(f, dst):
    data = pd.read_parquet(f)
    data = data[DEFAULT_COLUMN_NAMES]

    data[DEFAULT_LABEL_NAME] = data[DEFAULT_LABEL_NAME].astype(np.int32)
    data[DEFAULT_INT_NAMES] = data[DEFAULT_INT_NAMES].astype(np.float32)
    data[DEFAULT_CAT_NAMES] = data[DEFAULT_CAT_NAMES].astype(np.int32)

    data = data.to_records(index=False)
    data = data.tobytes()

    dst_file = dst + "/" + f.split("/")[-1] + ".bin"
    with open(dst_file, "wb") as dst_fd:
        dst_fd.write(data)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--intermediate_dir", type=str)
    parser.add_argument("--dst_dir", type=str)
    parser.add_argument("--parallel_jobs", default=20, type=int)
    args = parser.parse_args()

    print("Processing train files...")
    train_src_files = glob.glob(args.src_dir + "/train/*.parquet")
    train_intermediate_dir = os.path.join(args.intermediate_dir, "train")
    os.makedirs(train_intermediate_dir, exist_ok=True)

    Parallel(n_jobs=args.parallel_jobs)(
        delayed(process_file)(f, train_intermediate_dir)
        for f in tqdm.tqdm(train_src_files)
    )

    print("Train files conversion done")

    print("Processing test files...")
    test_src_files = glob.glob(args.src_dir + "/test/*.parquet")
    test_intermediate_dir = os.path.join(args.intermediate_dir, "test")
    os.makedirs(test_intermediate_dir, exist_ok=True)

    Parallel(n_jobs=args.parallel_jobs)(
        delayed(process_file)(f, test_intermediate_dir)
        for f in tqdm.tqdm(test_src_files)
    )
    print("Test files conversion done")

    print("Processing validation files...")
    valid_src_files = glob.glob(args.src_dir + "/validation/*.parquet")
    valid_intermediate_dir = os.path.join(args.intermediate_dir, "validation")
    os.makedirs(valid_intermediate_dir, exist_ok=True)

    Parallel(n_jobs=args.parallel_jobs)(
        delayed(process_file)(f, valid_intermediate_dir)
        for f in tqdm.tqdm(valid_src_files)
    )
    print("Validation files conversion done")

    os.makedirs(args.dst_dir, exist_ok=True)

    print("Concatenating train files")
    os.system(f"cat {train_intermediate_dir}/*.bin > {args.dst_dir}/train_data.bin")

    print("Concatenating test files")
    os.system(f"cat {test_intermediate_dir}/*.bin > {args.dst_dir}/test_data.bin")

    print("Concatenating validation files")
    os.system(
        f"cat {valid_intermediate_dir}/*.bin > {args.dst_dir}/validation_data.bin"
    )
    print(f"Processing took {time.time()-start_time:.2f} sec")


if __name__ == "__main__":
    main()
