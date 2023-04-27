#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from torch.autograd import profiler
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torchrec import distributed as trec_dist
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.utils import Batch
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)

from torchrec.distributed.shard import shard, shard_modules
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.distributed.types import (
    ModuleSharder,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from torch import nn
import torch


class SamplingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        batch 
    ):
        sparse_features = batch.sparse_features
        dense_features = batch.dense_features
        labels = batch.labels
        with record_function("in sampling module"):
            new_sparse_features = KeyedJaggedTensor.from_lengths_sync(
                keys=sparse_features.keys(),
                values=sparse_features.values().repeat(2),
                lengths=sparse_features.lengths().repeat(2),
            )
            new_dense_features = dense_features.repeat(2, 1)
            new_labels = labels.repeat(2)
            return Batch(new_dense_features, new_sparse_features, new_labels)


class RetrievalModule(nn.Module):
    def __init__(self, embedding_bag_configs, float_dim):
        super().__init__()
        self._ebc = EmbeddingBagCollection(embedding_bag_configs, device="meta")
        self._dense = nn.Linear(float_dim, 10)
        overarch_in = 10 + sum(table.embedding_dim for table in embedding_bag_configs)
        self._overarch = nn.Linear(overarch_in, 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        sparse_features = batch.sparse_features
        dense_features = batch.dense_features
        labels = batch.labels

        with record_function("model forward"):
            sparse_embs = self._ebc(sparse_features).values()
            dense_embs = self._dense(dense_features)
            pred = self._overarch(torch.cat([sparse_embs, dense_embs], 1)).sigmoid()

            return pred.sum(), pred

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import os
import sys

@record
def main(argv: List[str]) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    backend = "nccl"
    dist.init_process_group(backend=backend)

    pg = dist.GroupMember.WORLD
    assert pg is not None, "Process group is not initialized"
    env = ShardingEnv.from_process_group(pg)

    embedding_bag_config = [
        EmbeddingBagConfig(
            name="table_0",
            feature_names=["feature_0"],
            embedding_dim=4,
            num_embeddings=100,
        ),
        EmbeddingBagConfig(
            name="table_1",
            feature_names=["feature_1"],
            embedding_dim=4,
            num_embeddings=100,
        ),
    ]

    B = 8
    dataset = RandomRecDataset(
        keys=["feature_0", "feature_1"],
        batch_size=B,
        hash_sizes=[100, 100],
        ids_per_features=[4, 8],
        num_dense=16,
    )
    model = RetrievalModule(embedding_bag_config, 16)

    # planner = EmbeddingShardingPlanner(
    #     topology=Topology(
    #         world_size, ctx.device.type, local_world_size=ctx.local_size
    #     ),
    #     constraints=constraints,
    # )

    model = RetrievalModule(embedding_bag_config, 16)
    # plan: ShardingPlan = planner.collective_plan(model, [sharder], ctx.pg)

    sharded_model = shard_modules(
        module=model,
        env=env,
        # plan=plan,
        # sharders=[sharder],
        device=device,
    )
    sharded_model = sharded_model.to(device)

    opt = torch.optim.Adam(sharded_model.parameters(), lr=1e-3, weight_decay=1e-5)

    pipeline = TrainPipelineSparseDist(
        sharded_model,
        sampling_module = SamplingModule(),
        optimizer=opt,
        device=device,
    )

    it = iter(dataset)
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/ylgh/torchrec/pipeline_sample_overlap")
    ) as profiler:
        for _ in range(20):
            pipeline.progress(it)
            profiler.step()
        # profiler.export_chrome_trace("debug_pipeline_trace.json")

if __name__ == "__main__":
    main(sys.argv[1:])
