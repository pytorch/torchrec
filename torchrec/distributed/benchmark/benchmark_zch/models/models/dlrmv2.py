#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pyre_extensions import none_throws

from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES

from torchrec.datasets.utils import Batch

from torchrec.distributed.comm import get_local_size
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mc_adapter import McEmbeddingBagCollectionAdapter


class DLRMv2(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dlrm = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
        )
        self.train_model = DLRMTrain(self.dlrm)
        self.table_configs: List[EmbeddingBagConfig] = list(
            embedding_bag_collection.embedding_bag_configs()
        )

    def forward(
        self, batch: Batch
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:

        loss, (loss_values, pred_logits, labels) = self.train_model(batch)
        dummy_weights = torch.ones_like(pred_logits)
        return loss, (loss_values, pred_logits, labels, dummy_weights)

    def eval(self) -> None:
        self.train_model.eval()


def make_model_dlrmv2(
    args: argparse.Namespace, configs: Dict[str, Any], device: torch.device
) -> nn.Module:
    ebc_configs = [
        EmbeddingBagConfig(
            name=f"{feature_name}",
            embedding_dim=configs["embedding_dim"],
            num_embeddings=(
                none_throws(configs["num_embeddings_per_feature"])[feature_name]
                if args.num_embeddings is None
                else args.num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    if args.zch_method == "" or args.zch_method is None:
        ebc = EmbeddingBagCollection(tables=ebc_configs, device=torch.device("meta"))
    elif args.zch_method == "mpzch":
        ebc = (
            McEmbeddingBagCollectionAdapter(  # TODO: add switch for other ZCH or no ZCH
                tables=ebc_configs,
                input_hash_size=args.input_hash_size,
                device=torch.device("meta"),
                world_size=get_local_size(),
                zch_method="mpzch",
                mpzch_num_buckets=args.num_buckets,
                mpzch_max_probe=args.max_probe,
            )
        )
    else:
        raise NotImplementedError(f"ZCH method {args.zch_method} is not supported yet.")

    dlrm_model = DLRMv2(
        # pyre-ignore [6] # NOTE: Pyre reports that DLRM model's _embedding_bag_collection is EmbeddingBagCollection, but here we assign it with an EmbeddingBagCollectionAdapter.
        # This is because we want to implement managed collision functions without changing the DLRM class. The EmbeddingBagCollectionAdapter will simulate all the
        # APIs for EmbeddingBagCollection, and we can use it to replace the EmbeddingBagCollection in DLRM for managed collision functions.
        embedding_bag_collection=ebc,
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=[int(x) for x in configs["dense_arch_layer_sizes"]],
        over_arch_layer_sizes=[int(x) for x in configs["over_arch_layer_sizes"]],
        dense_device=device,
    )

    return dlrm_model
