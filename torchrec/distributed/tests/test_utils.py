#!/usr/bin/env python3

import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
from torchrec.distributed.embedding import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.utils import get_unsharded_module_names
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.tests.utils import get_free_port


class UtilsTest(unittest.TestCase):
    def test_get_unsharded_module_names(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        device = torch.device("cpu")
        backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(2)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(2)
        ]
        m = TestSparseNN(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=device,
            sparse_device=device,
        )
        dmp = DistributedModelParallel(
            module=m,
            init_data_parallel=False,
            device=device,
            sharders=[
                EmbeddingBagCollectionSharder(),
            ],
        )

        np.testing.assert_array_equal(
            sorted(get_unsharded_module_names(dmp)),
            sorted(["module.over", "module.dense"]),
        )
