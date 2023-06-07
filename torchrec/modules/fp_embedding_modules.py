#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import FeatureProcessor
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


@torch.fx.wrap
def apply_feature_processors_to_kjt(
    features: KeyedJaggedTensor,
    feature_processors: nn.ModuleDict,
) -> KeyedJaggedTensor:

    if len(features.keys()) == 0:
        return features

    processed_weights = []
    features_dict = features.to_dict()

    for key in features.keys():
        jt = features_dict[key]
        if key in feature_processors:
            fp_jt = feature_processors[key](jt)
        else:
            fp_jt = jt
        processed_weights.append(fp_jt.weights())

    return KeyedJaggedTensor(
        keys=features.keys(),
        values=features.values(),
        weights=torch.cat(processed_weights),
        lengths=features.lengths(),
        offsets=features._offsets,
        stride=features._stride,
        length_per_key=features._length_per_key,
        offset_per_key=features._offset_per_key,
        index_per_key=features._index_per_key,
    )


class FeatureProcessedEmbeddingBagCollection(nn.Module):
    """
    FeatureProcessedEmbeddingBagCollection represents a EmbeddingBagCollection module and a set of feature processor modules.
    The inputs into the FP-EBC will first be modified by the feature processor before being passed into the embedding bag collection.

    For details of input and output types, see EmbeddingBagCollection


    Args:
        embedding_bag_collection (EmbeddingBagCollection): ebc module
        feature_processors (Dict[str, FeatureProcessor]): feature processors
    Example::
        fp_ebc = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(...),
            {
                "feature_1": FeatureProcessorModule(...),
                "feature_2": FeatureProcessorModule2(...),
            }
        )

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        >>> fp_ebc(features).to_dict()
        {
            "feature_1": torch.Tensor(...)
            "feature_2": torch.Tensor(...)
        }
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        feature_processors: Dict[str, FeatureProcessor],
    ) -> None:
        super().__init__()
        self._embedding_bag_collection = embedding_bag_collection
        self._feature_processors = nn.ModuleDict(feature_processors)

        assert set(
            sum(
                [
                    config.feature_names
                    for config in self._embedding_bag_collection.embedding_bag_configs()
                ],
                [],
            )
        ) == set(
            feature_processors.keys()
        ), "Passed in feature processors do not match feature names of embedding bag"

        assert (
            embedding_bag_collection.is_weighted()
        ), "EmbeddingBagCollection must accept weighted inputs for feature processor"

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """

        fp_features = apply_feature_processors_to_kjt(
            features, self._feature_processors
        )
        return self._embedding_bag_collection(fp_features)
