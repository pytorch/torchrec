#!/usr/bin/env python3

# pyre-unsafe

import unittest
from math import sqrt
from typing import List, Tuple

import numpy as np
import torch
import deeplearning.fbgemm.fbgemm_gpu.fbgemm_gpu.batched_unary_embeddings_ops as batched_unary_embeddings_ops

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")

class TableBatchedEmbeddingsTest(unittest.TestCase):
    class RefEmb(torch.nn.Module):
        def __init__(self, num_tasks: int, hash_sizes: List[int]) -> None:
            super().__init__()
            self.num_tasks = num_tasks
            self.hash_sizes = hash_sizes
            self.emb_modules = torch.nn.ModuleList()
            for _ in range(num_tasks):
                for h in self.hash_sizes:
                    emb = torch.nn.EmbeddingBag(
                        num_embeddings=h,
                        embedding_dim=1,
                        mode="sum",
                        sparse=False,
                        include_last_offset=True,
                    )
                    emb.weight = torch.nn.Parameter(
                        torch.empty([h, 1]).uniform_(-sqrt(1 / h), sqrt(1 / h))
                    )
                    self.emb_modules.append(emb)

        def forward(
            self, offsets: List[torch.Tensor], indices: List[torch.Tensor]
        ) -> torch.Tensor:
            tt_list = []
            for n in range(self.num_tasks):
                t_list = []
                for i in range(len(self.hash_sizes)):
                    t = self.emb_modules[n * len(self.hash_sizes) + i](
                        offsets=offsets[i].long(), input=indices[i].long()
                    )
                    t_list.append(t)
                tt = torch.cat(t_list, dim=1)
                tt_list.append(tt)
            return torch.cat(tt_list).view(self.num_tasks, -1, len(self.hash_sizes))

    def _generate_unary_features(
        self, batch_size: int, num_embeddings: int
    ) -> Tuple[List, List, List]:
        lengths = []
        offsets = []
        indices = []
        offset = 0
        for _ in range(batch_size):
            n_indices = 1
            indices += np.round(
                np.random.random(n_indices) * (num_embeddings - 1)
            ).tolist()
            offsets.append(offset)
            offset += 1
            lengths.append(n_indices)
        offsets.append(offset)
        return (lengths, offsets, indices)

    def _test_main(self, gpu_infer: bool):
        if gpu_infer:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        batch_size = 128
        hash_sizes = [100, 200]
        num_tasks = 3
        # generate unary features
        lengths = []
        offsets = []
        indices = []
        for h in hash_sizes:
            l, o, i = self._generate_unary_features(batch_size, h)
            lengths.append(torch.IntTensor(l).to(device))
            offsets.append(torch.IntTensor(o).to(device))
            indices.append(torch.IntTensor(i).to(device))
        lengths_tensor = torch.cat(lengths)
        indices_tensor = torch.cat(indices)
        offsets_tensor = torch.zeros(
            lengths_tensor.numel() + 1,
            dtype=lengths_tensor.dtype,
            device=lengths_tensor.device,
        )
        offsets_tensor[1:] = torch.ops.fbgemm.asynchronous_inclusive_cumsum(
            lengths_tensor.view(-1)
        )
        # forward with int_32
        ref_emb = self.RefEmb(num_tasks, hash_sizes).to(device)
        unary_emb = batched_unary_embeddings_ops.BatchedUnaryEmbeddingBag(
            num_tasks, hash_sizes
        ).to(device)
        for i, param in enumerate(unary_emb.split_embedding_weights()):
            param.detach().copy_(ref_emb.emb_modules[i].weight)
        output_ref = ref_emb(offsets, indices)
        output = unary_emb(offsets_tensor, indices_tensor)
        torch.testing.assert_allclose(output_ref, output)

        # forward with int_64
        ref_emb = self.RefEmb(num_tasks, hash_sizes).to(device)
        unary_emb = batched_unary_embeddings_ops.BatchedUnaryEmbeddingBag(
            num_tasks=num_tasks, hash_sizes=hash_sizes, long_index=True
        ).to(device)
        for i, param in enumerate(unary_emb.split_embedding_weights()):
            param.detach().copy_(ref_emb.emb_modules[i].weight)
        output_ref = ref_emb(offsets, indices)
        output = unary_emb(offsets_tensor.long(), indices_tensor.long())
        torch.testing.assert_allclose(output_ref, output)

        # No implementation for CPU backprop yet
        if not gpu_infer:
            return

        d_output = (
            torch.randn([num_tasks, batch_size, len(hash_sizes)]).to(device) * 0.1
        )
        output_ref.backward(d_output)
        output.backward(d_output)
        d_weight_ref = []
        for emb in ref_emb.emb_modules:
            d_weight_ref.append(emb.weight.grad)
        d_weight_ref = torch.cat(d_weight_ref).view(num_tasks, sum(hash_sizes), -1)
        d_weight = unary_emb.weight.grad
        torch.testing.assert_allclose(d_weight_ref, d_weight)

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    def test_gpu(self):
        self._test_main(gpu_infer=True)

    def test_cpu(self):
        self._test_main(gpu_infer=False)
