#!/usr/bin/env python3
import argparse
import os
from typing import List

from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset


def get_dataloader(args: argparse.Namespace, backend: str) -> DataLoader:
    if args.num_embeddings_per_feature is not None:
        num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        num_embeddings = None
    else:
        num_embeddings_per_feature = None
        num_embeddings = args.num_embeddings

    pin_memory = (backend == "nccl") if args.pin_memory is None else args.pin_memory

    if args.in_memory_binary_criteo_path is None:
        dataloader = DataLoader(
            RandomRecDataset(
                keys=DEFAULT_CAT_NAMES,
                batch_size=args.batch_size,
                hash_size=num_embeddings,
                hash_sizes=num_embeddings_per_feature,
                manual_seed=args.seed,
                ids_per_feature=1,
                num_dense=len(DEFAULT_INT_NAMES),
            ),
            batch_size=None,
            batch_sampler=None,
            pin_memory=pin_memory,
            num_workers=args.num_workers,
        )
    else:
        files = os.listdir(args.in_memory_binary_criteo_path)
        split_files: List[List[str]] = [
            sorted(
                map(
                    lambda x: os.path.join(args.in_memory_binary_criteo_path, x),
                    filter(lambda s: kind in s, files),
                )
            )
            for kind in ["dense", "sparse", "labels"]
        ]
        dataloader = DataLoader(
            InMemoryBinaryCriteoIterDataPipe(
                *split_files,  # pyre-ignore[6]
                batch_size=args.batch_size,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                hashes=num_embeddings_per_feature
                if num_embeddings is None
                else ([num_embeddings] * 26),
            ),
            batch_size=None,
            pin_memory=pin_memory,
            collate_fn=lambda x: x,
        )

    return dataloader
