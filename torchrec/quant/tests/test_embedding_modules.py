#!/usr/bin/env python3

import unittest

import torch
from torchrec.distributed.embedding_lookup import GroupedEmbeddingBag
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    EmbeddingComputeKernel,
    ShardedEmbeddingTable,
)
from torchrec.modules.embedding_configs import (
    DataType,
    PoolingType,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class EmbeddingBagCollectionTest(unittest.TestCase):
    def test_ebc(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        embeddings = ebc(features)

        # test forward
        # pyre-ignore [16]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        quantized_embeddings = qebc(features.to(torch.device("cuda")))

        self.assertEqual(embeddings.keys(), quantized_embeddings.keys())
        self.assertEqual(embeddings["f1"].shape, quantized_embeddings["f1"].shape)
        self.assertTrue(
            torch.allclose(
                embeddings["f1"].cpu(),
                quantized_embeddings["f1"].cpu().float(),
                atol=1,
            )
        )
        self.assertTrue(
            torch.allclose(
                embeddings["f2"].cpu(),
                quantized_embeddings["f2"].cpu().float(),
                atol=1,
            )
        )

        # test state dict
        state_dict = ebc.state_dict()
        quantized_state_dict = qebc.state_dict()
        self.assertEqual(state_dict.keys(), quantized_state_dict.keys())

    def test_grouped_ebc(self) -> None:
        config = GroupedEmbeddingConfig(
            data_type=DataType.FP32,
            pooling=PoolingType.MEAN,
            is_weighted=False,
            compute_kernel=EmbeddingComputeKernel.DENSE,
            embedding_tables=[
                ShardedEmbeddingTable(
                    embedding_names=["f1"],
                    pooling=PoolingType.MEAN,
                    is_weighted=False,
                    name="t1",
                    embedding_dim=16,
                    local_cols=16,
                    num_embeddings=10,
                    local_rows=10,
                    feature_names=["f1"],
                ),
                ShardedEmbeddingTable(
                    embedding_names=["f2"],
                    pooling=PoolingType.MEAN,
                    is_weighted=False,
                    name="t2",
                    embedding_dim=16,
                    local_cols=16,
                    num_embeddings=10,
                    local_rows=10,
                    feature_names=["f2"],
                ),
            ],
        )
        gebc = GroupedEmbeddingBag(
            config=config,
            sparse=False,
        )

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        embeddings = gebc(features)

        # test forward
        # pyre-ignore [16]
        gebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qgebc = QuantEmbeddingBagCollection.from_float(gebc)
        quantized_embeddings = qgebc(features.to(torch.device("cuda")))

        self.assertEqual(embeddings.keys(), quantized_embeddings.keys())
        self.assertEqual(embeddings["f1"].shape, quantized_embeddings["f1"].shape)
        self.assertTrue(
            torch.allclose(
                embeddings["f1"].cpu(),
                quantized_embeddings["f1"].cpu().float(),
                atol=1,
            )
        )
        self.assertTrue(
            torch.allclose(
                embeddings["f2"].cpu(),
                quantized_embeddings["f2"].cpu().float(),
                atol=1,
            )
        )

        # test state dict
        state_dict = gebc.state_dict()
        quantized_state_dict = qgebc.state_dict()
        self.assertEqual(state_dict.keys(), quantized_state_dict.keys())
