#!/usr/bin/env python3

# pyre-unsafe

from math import sqrt
from typing import List

import torch

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")

def wrap_weight_to_parameter(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    for i, v in enumerate(weights):
        if not isinstance(v, torch.nn.Parameter):
            weights[i] = torch.nn.Parameter(v)
    return weights


class BatchedUnaryEmbeddingBag(torch.nn.Module):
    def __init__(self, num_tasks: int, hash_sizes: List[int], long_index: bool = False):
        super().__init__()
        self.num_tasks = num_tasks
        self.hash_sizes = hash_sizes
        # [N][sum(E)][1]
        embedding_data = torch.randn(size=(num_tasks, sum(self.hash_sizes), 1))
        self.weight = torch.nn.Parameter(embedding_data)
        index_dtype = torch.int64 if long_index else torch.int32
        table_offsets_tensor = torch.cat(
            [
                torch.tensor([0], dtype=index_dtype),
                torch.cumsum(
                    torch.tensor(hash_sizes),
                    dim=0,
                    dtype=index_dtype,
                ),
            ]
        )
        self.register_buffer("table_offsets_tensor", table_offsets_tensor)
        self.init_parameters()

    def forward(self, offsets: torch.Tensor, input: torch.Tensor):
        # output is [N][B][T]
        return torch.ops.fbgemm.batched_unary_embeddings(
            self.weight,
            self.table_offsets_tensor,
            offsets,
            input,
        )

    @torch.jit.export
    def split_embedding_weights(self):
        embedding_weights = []
        for n in range(self.num_tasks):
            for t in range(len(self.hash_sizes)):
                embedding_weights.append(
                    self.weight.detach()[
                        n,
                        self.table_offsets_tensor[t] : self.table_offsets_tensor[t + 1],
                        :,
                    ]
                )
        return embedding_weights

    @torch.jit.export
    def init_parameters(self):
        for (num_emb, param) in zip(
            self.hash_sizes * self.num_tasks,
            wrap_weight_to_parameter(self.split_embedding_weights())
        ):
            assert param.shape == (num_emb, 1)
            param.data.uniform_(-sqrt(1 / num_emb), sqrt(1 / num_emb))
