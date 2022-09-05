#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script preprocesses Criteo dataset tsv files to binary (npy) files.

import argparse
import os
import sys
from multiprocessing import Manager, Process
from typing import List

import numpy as np
from torchrec.datasets.criteo import BinaryCriteoUtils

DAYS = 24
COLUMNS = 40  # 13 dense, 26 sparse, 1 label, in that order
INT_COLUMNS = 13


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle preprocessed npy dataset.")
    parser.add_argument(
        "--input_dir_labels_and_dense",
        type=str,
        required=True,
        help="Input directory containing labels and dense features.",
    )
    parser.add_argument(
        "--input_dir_sparse",
        type=str,
        required=True,
        help="Input directory with sparse features. Sometimes these"
        " features can be stored in a separate directory from the"
        " labels and dense features as extra pre-processing was"
        " applied to them.",
    )
    parser.add_argument(
        "--output_dir_full_set",
        type=str,
        default=None,
        help="If specified, store the full dataset (unshuffled).",
    )
    parser.add_argument(
        "--output_dir_shuffled",
        type=str,
        required=True,
        help="Output directory to store split shuffled npy files.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="random seed for the dataset shuffle",
    )
    return parser.parse_args(argv)


def count_rows(rows_per_file, path, day):
    day_file = os.path.join(path, f"day_{day}_labels.npy")
    data = np.load(day_file)
    num_rows = data.shape[0]

    rows_per_file[day] = num_rows
    print(f"counted {num_rows} for {day_file}")


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    input_dir_labels_and_dense = args.input_dir_labels_and_dense
    input_dir_sparse = args.input_dir_sparse
    output_dir_full_set = args.output_dir_full_set
    output_dir_shuffled = args.output_dir_shuffled

    # Count num rows in each day file.
    rows_per_file = Manager().dict()

    # Adjust the number of processes here if <24 processes available to run
    # simultaneously.
    processes = [
        Process(
            target=count_rows,
            name="count_rows:day%i" % i,
            args=(rows_per_file, input_dir_labels_and_dense, i),
        )
        for i in range(0, DAYS - 1)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    BinaryCriteoUtils.shuffle(
        input_dir_labels_and_dense,
        input_dir_sparse,
        output_dir_shuffled,
        rows_per_file,
        output_dir_full_set,
        random_seed=args.random_seed,
        days=DAYS,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
