#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import List, Iterator

import torch
import torch.distributed as dist
from torchrec import EmbeddingBagCollection
from torchrec import KeyedJaggedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper


class RandomIterator(Iterator):
    def __init__(
        self, batch_size: int, num_dense: int, num_sparse: int, num_embeddings: int
    ) -> None:
        self.batch_size = batch_size
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.sparse_keys = [f"feature{id}" for id in range(self.num_sparse)]
        self.num_embeddings = num_embeddings
        self.num_ids_per_feature = 3
        self.num_ids_to_generate = (
            self.num_sparse * self.num_ids_per_feature * self.batch_size
        )

    def __next__(self) -> (torch.Tensor, KeyedJaggedTensor, torch.Tensor):
        float_features = torch.randn(
            self.batch_size,
            self.num_dense,
        )
        labels = torch.randint(
            low=0,
            high=2,
            size=(self.batch_size,),
        )
        sparse_ids = torch.randint(
            high=self.num_sparse,
            size=(self.num_ids_to_generate,),
        )
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=self.sparse_keys,
            values=sparse_ids,
            offsets=torch.tensor(
                list(range(0, self.num_ids_to_generate + 1, self.num_ids_per_feature)),
                dtype=torch.int32,
            ),
        )
        return (float_features, sparse_features, labels)


def main(argv: List[str]) -> None:
    batch_size = 1024
    num_dense = 1000
    num_sparse = 20
    num_embeddings = 1000000

    configs = [
        EmbeddingBagConfig(
            name=f"table{id}",
            embedding_dim=64,
            num_embeddings=num_embeddings,
            feature_names=[f"feature{id}"],
        )
        for id in range(num_sparse)
    ]

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        backend = "gloo"
        print(
            "\033[92m"
            + "Warning: CUDA not available! Is this meant to be a CPU installation?"
        )

    dist.init_process_group(backend=backend)

    model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=configs, device=torch.device("meta")
        ),
        dense_in_features=num_dense,
        dense_arch_layer_sizes=[500, 64],
        over_arch_layer_sizes=[32, 16, 1],
        dense_device=device,
    )
    model = DistributedModelParallel(
        module=model,
        device=device,
    )
    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.SGD(params, lr=0.01),
    )

    random_iterator = RandomIterator(batch_size, num_dense, num_sparse, num_embeddings)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(10):
        (dense_features, sparse_features, labels) = next(random_iterator)
        dense_features = dense_features.to(device)
        sparse_features = sparse_features.to(device)
        output = model(dense_features, sparse_features)
        labels = labels.to(device)
        loss = loss_fn(output.squeeze(), labels.float())
        torch.sum(loss, dim=0).backward()
        optimizer.zero_grad()
        optimizer.step()

    print(
        "\033[92m" + "Successfully ran a few epochs for DLRM. Installation looks good!"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
