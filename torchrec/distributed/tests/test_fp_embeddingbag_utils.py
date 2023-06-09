#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import cast, Dict, List, Tuple

import torch
from torch import nn
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import (
    FeatureProcessor,
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        use_fp_collection: bool,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._fp_ebc: FeatureProcessedEmbeddingBagCollection = (
            FeatureProcessedEmbeddingBagCollection(
                EmbeddingBagCollection(
                    tables=tables,
                    device=device,
                    is_weighted=True,
                ),
                cast(
                    Dict[str, FeatureProcessor],
                    {
                        "feature_0": PositionWeightedModule(max_feature_length=10),
                        "feature_1": PositionWeightedModule(max_feature_length=10),
                        "feature_2": PositionWeightedModule(max_feature_length=12),
                        "feature_3": PositionWeightedModule(max_feature_length=12),
                    },
                )
                if not use_fp_collection
                else PositionWeightedModuleCollection(
                    max_feature_lengths={
                        "feature_0": 10,
                        "feature_1": 10,
                        "feature_2": 12,
                        "feature_3": 12,
                    }
                ),
            ).to(device)
        )

    def forward(self, kjt: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fp_ebc_out = self._fp_ebc(kjt)
        pred = torch.cat(
            [
                fp_ebc_out[key]
                for key in ["feature_0", "feature_1", "feature_2", "feature_3"]
            ],
            dim=1,
        )
        loss = pred.mean()
        return loss, pred


def create_module_and_freeze(
    tables: List[EmbeddingBagConfig],
    use_fp_collection: bool,
    device: torch.device,
) -> SparseArch:

    sparse_arch = SparseArch(tables, use_fp_collection, device)

    torch.manual_seed(0)
    for param in sparse_arch.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    torch.manual_seed(0)

    return sparse_arch
