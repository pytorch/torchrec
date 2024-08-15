#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @nolint
# pyre-ignore-all-errors

import unittest
from argparse import Namespace

from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES

from torchrec.inference.dlrm_predict import (
    create_training_batch,
    DLRMModelConfig,
    DLRMPredictFactory,
)


class InferenceTest(unittest.TestCase):
    def test_dlrm_inference_package(self) -> None:
        args = Namespace()
        args.batch_size = 10
        args.num_embedding_features = 26
        args.num_dense_features = len(DEFAULT_INT_NAMES)
        args.dense_arch_layer_sizes = "512,256,64"
        args.over_arch_layer_sizes = "512,512,256,1"
        args.sparse_feature_names = ",".join(DEFAULT_CAT_NAMES)
        args.num_embeddings = 100_000
        args.num_embeddings_per_feature = ",".join(
            [str(args.num_embeddings)] * args.num_embedding_features
        )

        batch = create_training_batch(args)

        model_config = DLRMModelConfig(
            dense_arch_layer_sizes=list(
                map(int, args.dense_arch_layer_sizes.split(","))
            ),
            dense_in_features=args.num_dense_features,
            embedding_dim=64,
            id_list_features_keys=args.sparse_feature_names.split(","),
            num_embeddings_per_feature=list(
                map(int, args.num_embeddings_per_feature.split(","))
            ),
            num_embeddings=args.num_embeddings,
            over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
            sample_input=batch,
        )

        # Create torchscript model for inference
        DLRMPredictFactory(model_config).create_predict_module(
            world_size=1, device="cpu"
        )
