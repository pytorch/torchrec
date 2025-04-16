#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import List, Optional

import torch

from torchrec.modules.embedding_configs import EmbeddingConfig

from torchrec.modules.embedding_modules import EmbeddingCollection


class MappedEmbeddingCollection(EmbeddingCollection):
    """ """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
    ) -> None:
        super().__init__(tables=tables, need_indices=need_indices, device=device)
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
