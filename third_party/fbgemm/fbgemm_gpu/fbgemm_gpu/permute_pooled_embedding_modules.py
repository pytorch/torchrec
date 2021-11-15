#!/usr/bin/env python3

import logging
from itertools import accumulate
from typing import List, Optional

import torch
from torch import nn

torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu")


class PermutePooledEmbeddings(nn.Module):
    def __init__(
        self,
        embs_dims: List[int],
        permute: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super(PermutePooledEmbeddings, self).__init__()
        logging.info("Using Permute Pooled Embeddings")

        self.register_buffer(
            "_offset_dim_list",
            torch.tensor(
                [0] + list(accumulate(embs_dims)), device=device, dtype=torch.int64
            ),
        )
        self.register_buffer(
            "_permute", torch.tensor(permute, device=device, dtype=torch.int64)
        )

        inv_permute: List[int] = [0] * len(permute)
        for i, p in enumerate(permute):
            inv_permute[p] = i

        self.register_buffer(
            "_inv_permute", torch.tensor(inv_permute, device=device, dtype=torch.int64)
        )

        #  `Union[BoundMethod[typing.Callable(torch.Tensor.tolist)[[Named(self,
        #  torch.Tensor)], List[typing.Any]], torch.Tensor], nn.Module, torch.Tensor]`
        #  is not a function.

        inv_embs_dims = [embs_dims[i] for i in permute]

        self.register_buffer(
            "_inv_offset_dim_list",
            torch.tensor(
                [0] + list(accumulate(inv_embs_dims)), device=device, dtype=torch.int64
            ),
        )

    def forward(self, pooled_embs: torch.Tensor) -> torch.Tensor:
        result = torch.ops.fbgemm.permute_pooled_embs_auto_grad(
            pooled_embs,
            self._offset_dim_list.to(device=pooled_embs.device),
            self._permute.to(device=pooled_embs.device),
            self._inv_offset_dim_list.to(device=pooled_embs.device),
            self._inv_permute.to(device=pooled_embs.device),
        )
        return result
