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
    DEFAULT_COLUMN_NAMES,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
    DAYS,
)
from utils.dask import setup_dask


def process_criteo_day(input_path, output_path, day):
    input_dataset = nvt.Dataset(os.path.join(input_path, f"day_{day}.parquet"))

    cat_features = DEFAULT_CAT_NAMES >> nvt.ops.FillMissing()
    cont_features = (
        DEFAULT_INT_NAMES
        >> nvt.ops.FillMissing()
        >> nvt.ops.LambdaOp(lambda col: col + 2)
        >> nvt.ops.LogOp()
    )
    features = cat_features + cont_features + [DEFAULT_LABEL_NAME]
    workflow = nvt.Workflow(features)

    workflow.fit(input_dataset)

    target_dtypes = {
        c: np.float32 for c in DEFAULT_COLUMN_NAMES[:14] + [DEFAULT_LABEL_NAME]
    }
    target_dtypes.update({c: "hex" for c in DEFAULT_COLUMN_NAMES[14:]})

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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

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
    for day in range(DAYS):
        process_criteo_day(input_path, output_path, day)
    print(f"Processing took {time.time()-start_time:.2f} sec")
