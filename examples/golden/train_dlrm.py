#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional
from hypothesis import target

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torchrec import EmbeddingBagCollection
from torchrec.datasets.random import RandomRecDataset
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.fused_embedding_modules import fuse_embedding_optimizer
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.metrics.ne import NEMetric

from torchrec.metrics.metrics_config import DefaultTaskInfo

logger: logging.Logger = logging.getLogger(__name__)


def get_dataset(
    batch_size: int,
    num_batches: int,
    num_dense_features: int,
    embedding_bag_configs: List[EmbeddingBagConfig],
    pooling_factors: Optional[Dict[str, int]] = None,
) -> RandomRecDataset:
    keys = []
    ids_per_features = []
    hash_sizes = []

    if pooling_factors is None:
        pooling_factors = {}

    for table in embedding_bag_configs:
        for feature_name in table.feature_names:
            keys.append(feature_name)
            ids_per_features.append(pooling_factors.get(feature_name, 4))
            hash_sizes.append(table.num_embeddings)

    return RandomRecDataset(
        keys=keys,
        batch_size=batch_size,
        hash_sizes=hash_sizes,
        ids_per_features=ids_per_features,
        num_dense=num_dense_features,
        num_batches=num_batches,
    )


@record
def main(argv: List[str]) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        backend = "gloo"
        print("\033[92m" + f"WARNING: Running in CPU mode.")
    dist.init_process_group(backend=backend)

    batch_size = 512
    num_batches = 500
    num_dense_features = 1000
    num_sparse_features = 20
    sparse_num_embeddings = 1000000
    num_epochs = 3

    embedding_bag_configs = [
        EmbeddingBagConfig(
            name=f"table{id}",
            embedding_dim=64,
            num_embeddings=sparse_num_embeddings,
            feature_names=[f"feature_{id}"],
        )
        for id in range(num_sparse_features)
    ]

    dataset = get_dataset(
        batch_size,
        num_batches,
        num_dense_features,
        embedding_bag_configs,
    )

    model = DLRMTrain(
        DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=embedding_bag_configs, device=torch.device("meta")
            ),
            dense_in_features=num_dense_features,
            dense_arch_layer_sizes=[500, 64],
            over_arch_layer_sizes=[32, 16, 1],
            dense_device=device,
        )
    )

    model = fuse_embedding_optimizer(
        model,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={
            "lr": 0.02,
        },
        device=torch.device("meta"),
    )

    sharded_model = DistributedModelParallel(
        module=model,
        device=device,
    )

    optimizer = CombinedOptimizer(
        [
            KeyedOptimizerWrapper(
                dict(model.named_parameters()),
                lambda params: torch.optim.SGD(params, lr=0.01),
            ),
            sharded_model.fused_optimizer,
        ]
    )

    training_pipeline = TrainPipelineSparseDist(
        model=sharded_model, optimizer=optimizer, device=device
    )

    # Training Loop
    for epoch in range(num_epochs):
        it = iter(dataset)
        batch_idx = 0
        while True:
            try:
                loss, logits, targets = training_pipeline.progress(it)
                predictions = logits.sigmoid()
                if batch_idx % 100 == 0:
                    logger.info(f"Loss is {loss}")
            except StopIteration:
                logger.info(f"Reached stop iteration for epoch {epoch}")


if __name__ == "__main__":
    main(sys.argv[1:])
