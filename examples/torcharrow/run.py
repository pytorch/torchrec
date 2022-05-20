#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import click
import dataloader as torcharrow_dataloader
import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch.distributed.elastic.multiprocessing.errors import record
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, INT_FEATURE_COUNT
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper


@record
@click.command()
@click.option("--batch_size", default=256)
@click.option("--num_embeddings", default=2048)
@click.option("--sigrid_hash_salt", default=0)
@click.option("--parquet_directory", default="/data/criteo_preproc")
def main(
    batch_size,
    num_embeddings,
    sigrid_hash_salt,
    parquet_directory,
) -> None:
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
            + f"WARNING: Running in CPU mode. cuda availablility {torch.cuda.is_available()}."
        )

    dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()

    dataloader = torcharrow_dataloader.get_dataloader(
        parquet_directory,
        world_size,
        rank,
        batch_size=batch_size,
        num_embeddings=num_embeddings,
        salt=sigrid_hash_salt,
    )
    it = iter(dataloader)

    model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name=f"table_{cat_name}",
                    embedding_dim=64,
                    num_embeddings=num_embeddings,
                    feature_names=[cat_name],
                )
                for cat_name in DEFAULT_CAT_NAMES + ["bucketize_int_0"]
            ],
            device=torch.device("meta"),
        ),
        dense_in_features=INT_FEATURE_COUNT,
        dense_arch_layer_sizes=[64],
        over_arch_layer_sizes=[32, 1],
        dense_device=device,
    )

    fused_params = {
        "learning_rate": 0.02,
        "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
    }

    sharded_model = DistributedModelParallel(
        module=model,
        device=device,
        sharders=[
            EmbeddingBagCollectionSharder(fused_params=fused_params),
        ],
    )

    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.SGD(params, lr=0.01),
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    print_example = dist.get_rank() == 0
    for (dense_features, kjt, labels) in it:
        if print_example:
            print("Example dense_features", dense_features)
            print("Example KJT input", kjt)
            print_example = False

        dense_features = dense_features.to(device)
        kjt = kjt.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = sharded_model(dense_features, kjt)
        loss = loss_fn(preds.squeeze(), labels.squeeze())
        loss.sum().backward()

        optimizer.step()

    print("\033[92m" + "DLRM run with torcharrow last-mile preprocessing finished!")


if __name__ == "__main__":
    main()
