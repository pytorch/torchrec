#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, List, Tuple, Union

import numpy as np
import torch.utils.data.datapipes as dp
import torcharrow as ta
import torcharrow.dtypes as dt
from torch.utils.data import IterDataPipe
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    CriteoIterDataPipe,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    INT_FEATURE_COUNT,
)
from torchrec.datasets.utils import safe_cast

DTYPE = dt.Struct(
    [
        dt.Field("labels", dt.int8),
        dt.Field(
            "dense_features",
            dt.Struct(
                [
                    dt.Field(int_name, dt.Int32(nullable=True))
                    for int_name in DEFAULT_INT_NAMES
                ]
            ),
        ),
        dt.Field(
            "sparse_features",
            dt.Struct(
                [
                    dt.Field(cat_name, dt.Int32(nullable=True))
                    for cat_name in DEFAULT_CAT_NAMES
                ]
            ),
        ),
    ]
)


def _torcharrow_row_mapper(
    row: List[str],
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    # TODO: Fix safe_cast type annotation
    label = int(safe_cast(row[0], int, 0))
    dense = tuple(
        (int(safe_cast(row[i], int, 0)) for i in range(1, 1 + INT_FEATURE_COUNT))
    )
    sparse = tuple(
        (
            int(safe_cast(row[i], str, "0") or "0", 16)
            for i in range(
                1 + INT_FEATURE_COUNT, 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT
            )
        )
    )
    # TorchArrow doesn't support uint32, but we can save memory
    # by not using int64. Numpy will automatically handle sparse values >= 2 ** 31.
    sparse = tuple(np.array(sparse, dtype=np.int32).tolist())

    return (label, dense, sparse)


def criteo_dataframes_from_tsv(
    paths: Union[str, Iterable[str]],
    *,
    batch_size: int = 128,
) -> IterDataPipe:
    """
    Load Criteo dataset (Kaggle or Terabyte) as TorchArrow DataFrame streams from TSV file(s)

    This implementaiton is inefficient and is used for prototype and test only.

    Args:
        paths (str or Iterable[str]): local paths to TSV files that constitute
            the Kaggle or Criteo 1TB dataset.

    Example::

        datapipe = criteo_dataframes_from_tsv(
            ["/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv"]
        )
        for df in datapipe:
           print(df)
    """
    if isinstance(paths, str):
        paths = [paths]

    datapipe = CriteoIterDataPipe(paths, row_mapper=_torcharrow_row_mapper)
    datapipe = dp.iter.Batcher(datapipe, batch_size)
    datapipe = dp.iter.Mapper(datapipe, lambda batch: ta.dataframe(batch, dtype=DTYPE))

    return datapipe
