import torch
from torch.utils.data import DataLoader

from typing import List

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch

import nvtabular as nvt
from nvtabular.loader.torch import TorchAsyncItr

from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
)


def seed_fn():
    """
    Generate consistent dataloader shuffle seeds across workers
    Reseeds each worker's dataloader each epoch to get fresh a shuffle
    that's consistent across workers.

    TODO there is something wrong with the seed_fn example. Return 0 for now
    """
    return 0


def get_nvt_criteo_dataloader(
    paths: List[str],
    batch_size: int,
    world_size: int,
    rank: int,
):
    dataset = TorchAsyncItr(
        nvt.Dataset(paths, part_size="144MB"),
        batch_size=batch_size,
        cats=DEFAULT_CAT_NAMES,
        conts=DEFAULT_INT_NAMES,
        labels=[DEFAULT_LABEL_NAME],
        device=rank,
        global_size=world_size,
        global_rank=rank,
        shuffle=True,
        seed_fn=seed_fn,
    )

    def collate_fn(attr_dict):
        batch_features, labels = attr_dict
        # We know that all categories are one-hot. However, this may not generalize
        # We should work with nvidia to allow nvtabular to natively transform to
        # a KJT format.
        return Batch(
            dense_features=torch.cat(
                [batch_features[feature] for feature in DEFAULT_INT_NAMES], dim=1
            ),
            sparse_features=KeyedJaggedTensor.from_lengths_sync(
                keys=DEFAULT_CAT_NAMES,
                values=torch.cat(
                    [batch_features[feature] for feature in DEFAULT_CAT_NAMES]
                ).view(-1),
                lengths=torch.ones(
                    (len(DEFAULT_CAT_NAMES) * batch_size), dtype=torch.int32
                ),
            ),
            labels=labels,
        )

    # Don't pin memory since the batches are already on cuda!
    # Num worker is set to zero as well, because it is on GPU
    return DataLoader(
        dataset,
        batch_size=None,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0,
    )
