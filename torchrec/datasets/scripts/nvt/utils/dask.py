#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import numba
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from nvtabular.utils import device_mem_size


def setup_dask(dask_workdir):
    if os.path.exists(dask_workdir):
        shutil.rmtree(dask_workdir)
    os.makedirs(dask_workdir)

    device_limit_frac = 0.8  # Spill GPU-Worker memory to host at this limit.
    device_pool_frac = 0.7

    # Use total device size to calculate device limit and pool_size
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)

    cluster = LocalCUDACluster(
        protocol="tcp",
        n_workers=len(numba.cuda.gpus),
        CUDA_VISIBLE_DEVICES=range(len(numba.cuda.gpus)),
        device_memory_limit=device_limit,
        local_directory=dask_workdir,
        rmm_pool_size=(device_pool_size // 256) * 256,
    )

    return Client(cluster)
