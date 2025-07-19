#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is used to parse the KuaiRand datasets to
# remap the input values into [0, hash_size - 1] range with
# sparse distribution.
# The hash modulo is used for the remapping.
# This is an example of pre-hash solution to deal with imbalanced
# input values distribution. Other datasets can be pre-procesed
# following the same manner.

import argparse
import json
import time

import pandas as pd
from tqdm import tqdm

# load the sequence one line at a time
# for each user id x, perform the following hash
# 1. x = int(x) ^ 0xDEADBEEF
# 2. x = x * 2654435761
# 3. x = x ^ (x >> 16)
# 4. x = x % hash_size


def hash_modulo(x, hash_size):
    """
    hash the user id and video id
    Args:
        x: user id or video id
        hash_size: hash size
    Returns:
        hashed user id or video id
    """
    x = int(x) ^ 0xDEADBEEF
    x = x * 2654435761
    x = x ^ (x >> 16)
    x = x % hash_size
    return x


def hash_list_modulo(x_list, hash_size):
    """
    hash the video id when the input is a list
    Args:
        x_list: list of video id
        hash_size: hash size
    Returns:
        hashed video id list
    """
    for x in x_list:
        x = hash_modulo(x, hash_size)
    return x_list


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_seq_path",
        type=str,
        default="data/processed_seqs.csv",
        help="path to the processed sequence",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed_seqs_hashed.csv",
        help="path to the output file",
    )
    parser.add_argument(
        "--hash_size",
        type=int,
        default=100000,
        help="hash size",
    )
    args = parser.parse_args()
    # read the sequence
    print("load the sequence...")
    # read the csv file whose first row is the header
    start_read_time = time.perf_counter()
    df = pd.read_csv(args.processed_seq_path, header=0)
    end_read_time = time.perf_counter()
    print(f"read csv time: {end_read_time - start_read_time} seconds")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        user_id = row["user_id"]
        video_id = json.loads(row["video_id"])
        hashed_user_id = hash_modulo(user_id, args.hash_size)
        hashed_video_id = hash_list_modulo(video_id, args.hash_size)
        df.at[idx, "user_id"] = hashed_user_id
        df.at[idx, "video_id"] = json.dumps(hashed_video_id)
    print("save to csv...")
    df.to_csv(args.output_path, header=True, index=False)
