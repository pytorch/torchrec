#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.sharding.cw_sharding import CwPooledEmbeddingSharding
from torchrec.distributed.types import (
    ParameterSharding,
)
from torchrec.modules.embedding_configs import EmbeddingTableConfig


class TwCwPooledEmbeddingSharding(CwPooledEmbeddingSharding):
    """
    Shards embedding bags table-wise column-wise, i.e.. a given embedding table is
    distributed by specified number of columns and table slices are placed on all ranks
    within a host group.
    """

    def __init__(
        self,
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
    ) -> None:
        super().__init__(
            embedding_configs, pg, device, permute_embeddings=permute_embeddings
        )
