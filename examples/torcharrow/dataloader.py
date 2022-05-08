#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torcharrow as ta
from torch.utils.data import DataLoader
from torcharrow import functional
from torchdata.datapipes.iter import FileLister
from torchrec.datasets.criteo import (
    DEFAULT_INT_NAMES,
    DEFAULT_CAT_NAMES,
    DEFAULT_LABEL_NAME,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def get_dataloader(
    parquet_directory, world_size, rank, num_embeddings=4096, salt=0, batch_size=16
):
    source_dp = FileLister(parquet_directory, masks="*.parquet")
    # TODO support batch_size for load_parquet_as_df.
    # TODO use OSSArrowDataPipe once it is ready
    parquet_df_dp = source_dp.load_parquet_as_df()

    def preproc(df, max_idx=num_embeddings, salt=salt):

        for feature_name in DEFAULT_INT_NAMES + DEFAULT_CAT_NAMES:
            df[feature_name] = df[feature_name].fill_null(0)

        # construct a sprase index from a dense one
        df["bucketize_int_0"] = functional.array_constructor(
            functional.bucketize(df["int_0"], [0.5, 1.0, 1.5])
        )

        # flatten several columns into one
        df["dense_features"] = ta.dataframe(
            {int_name: df[int_name] for int_name in DEFAULT_INT_NAMES}
        )
        df["dense_features"] = (df["dense_features"] + 3).log()

        for cat_name in DEFAULT_CAT_NAMES:
            # hash our embedding index into our embedding tables
            df[cat_name] = functional.sigrid_hash(df[cat_name], salt, max_idx)
            df[cat_name] = functional.array_constructor(df[cat_name])
            df[cat_name] = functional.firstx(df[cat_name], 1)

        df["sparse_features"] = ta.dataframe(
            {
                cat_name: df[cat_name]
                for cat_name in DEFAULT_CAT_NAMES + ["bucketize_int_0"]
            }
        )

        df = df.drop(
            list(
                set(df.columns)
                - set("dense_features", "sparse_features", DEFAULT_LABEL_NAME)
            )
        )
        return df

    parquet_df_dp = parquet_df_dp.map(preproc).sharding_filter()
    parquet_df_dp.apply_sharding(world_size, rank)

    # TODO use ArrowDataPipe::collate(conversion={tap.rec.Dense(), tap.rec.Sparse(is_jagged=True)}) once it is ready.
    def criteo_collate(df):
        dense_features = torch.tensor(df["dense_features"])
        labels = torch.tensor(df[DEFAULT_LABEL_NAME])

        kjt_keys = df["sparse_features"].columns
        kjt_values = []
        kjt_lengths = []
        for row in df["sparse_features"]:
            for idx, _column in enumerate(df["sparse_features"].columns):
                value = row[idx]
                kjt_values.extend(value)
                kjt_lengths.append(len(value))
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=kjt_keys,
            values=torch.tensor(kjt_values),
            lengths=torch.tensor(kjt_lengths),
        )

        return dense_features, kjt, labels

    return DataLoader(
        parquet_df_dp,
        batch_size=None,
        collate_fn=criteo_collate,
        drop_last=False,
        pin_memory=True,
    )
