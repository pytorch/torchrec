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
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
)

from utils.dask import setup_dask


def process_criteo_day(input_path, output_path, day, num_embeddings_per_feature):
    input_dataset = nvt.Dataset(os.path.join(input_path, f"day_{day}.parquet"))

    cat_features = (
        DEFAULT_CAT_NAMES
        >> nvt.ops.FillMissing()
        >> nvt.ops.HashBucket(
            {
                cat_name: num_embeddings
                for cat_name, num_embeddings in zip(
                    DEFAULT_CAT_NAMES, num_embeddings_per_feature
                )
            }
        )
    )

    cont_features = (
        DEFAULT_INT_NAMES
        >> nvt.ops.FillMissing()
        >> nvt.ops.LambdaOp(lambda col: col + 2)
        >> nvt.ops.LogOp()
    )
    features = cat_features + cont_features + [DEFAULT_LABEL_NAME]
    workflow = nvt.Workflow(features)

    workflow.fit(input_dataset)

    target_dtypes = {c: np.float32 for c in DEFAULT_INT_NAMES + [DEFAULT_LABEL_NAME]}
    target_dtypes.update({c: np.int64 for c in DEFAULT_CAT_NAMES})

    workflow.transform(input_dataset).to_parquet(
        output_path=os.path.join(output_path, f"day_{day}"),
        dtypes=target_dtypes,
        cats=DEFAULT_CAT_NAMES,
        conts=DEFAULT_INT_NAMES,
        labels=[DEFAULT_LABEL_NAME],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess criteo dataset")
    parser.add_argument("--base_path", "-b", dest="base_path", help="Base path")
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument("--days", "-d", type=int, default=26, help="days to process")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    num_embeddings_per_feature = [args.num_embeddings] * len(DEFAULT_CAT_NAMES)
    if args.num_embeddings_per_feature is not None:
        maybe_num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        if len(maybe_num_embeddings_per_feature) != len(DEFAULT_CAT_NAMES):
            raise ValueError(
                f"num_embedding_per_feature must have exactly {len(DEFAULT_CAT_NAMES)} values"
            )
        num_embeddings_per_feature = maybe_num_embeddings_per_feature
        
    dask_workdir = os.path.join(args.base_path, "dask_workdir")
    client = setup_dask(dask_workdir)

    output_path = os.path.join(args.base_path, "criteo_preproc")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    input_path = os.path.join(args.base_path, "criteo_parquet")
    assert os.path.exists(
        input_path
    ), f"Criteo parquet path {input_path} does not exist"

    start_time = time.time()
    for day in range(args.days):
        process_criteo_day(input_path, output_path, day, num_embeddings_per_feature)
    print(f"Processing took {time.time()-start_time:.2f} sec")