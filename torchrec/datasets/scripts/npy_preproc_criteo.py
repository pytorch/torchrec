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
from typing import List

from torchrec.datasets.criteo import BinaryCriteoUtils


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Criteo tsv -> npy preprocessing script."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing Criteo tsv files. Files in the directory "
        "should be named day_{0-23}.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to store npy files.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """
    This function preprocesses the raw Criteo tsvs into the format (npy binary)
    expected by InMemoryBinaryCriteoIterDataPipe.

    Args:
        argv (List[str]): Command line args.

    Returns:
        None.
    """

    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir

    for i in range(24):
        in_file_path = os.path.join(input_dir, f"day_{i}")
        if not os.path.exists(in_file_path):
            continue
        dense_out_file_path = os.path.join(output_dir, f"day_{i}_dense.npy")
        sparse_out_file_path = os.path.join(output_dir, f"day_{i}_sparse.npy")
        labels_out_file_path = os.path.join(output_dir, f"day_{i}_labels.npy")
        print(
            f"Processing {in_file_path}.\nOutput will be saved to\n{dense_out_file_path}"
            f"\n{sparse_out_file_path}\n{labels_out_file_path}"
        )
        BinaryCriteoUtils.tsv_to_npys(
            in_file_path,
            dense_out_file_path,
            sparse_out_file_path,
            labels_out_file_path,
        )
        print(f"Done processing {in_file_path}.")


if __name__ == "__main__":
    main(sys.argv[1:])
