#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3
from dataclasses import dataclass
from typing import cast, Dict, Iterable, List, Optional, Union

import torch

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class EmbeddingTableProps:
    """
    Properties of an embedding table.

    Args:
        embedding_table_config: Config of the embedding table of Union(EmbeddingConfig or EmbeddingBagConfig)
        sharding (ShardingType): sharding type of the table
        weight_type (WeightedType): weight
    """

    embedding_table_config: Union[EmbeddingConfig, EmbeddingBagConfig]
    sharding: ShardingType
    is_weighted: bool = False
