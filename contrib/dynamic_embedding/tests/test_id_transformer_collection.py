import unittest
from distutils.command.config import config

import torch
from torchrec import EmbeddingBagConfig, EmbeddingConfig, KeyedJaggedTensor
from torchrec_dynamic_embedding import IDTransformerCollection


class TestIDTransformer(unittest.TestCase):
    def testTransform(self):
        configs = [
            EmbeddingConfig(name="A", num_embeddings=8, embedding_dim=32),
            EmbeddingConfig(name="B", num_embeddings=8, embedding_dim=32),
        ]
        transformer_collection = IDTransformerCollection(configs)
        global_kjt = KeyedJaggedTensor(
            keys=["B", "A"],
            values=torch.tensor([1, 3, 2, 3, 4, 3, 2]),
            lengths=torch.tensor([4, 3]),
        )
        cache_kjt = transformer_collection.transform(global_kjt)
        self.assertEqual(cache_kjt.keys(), global_kjt.keys())
        self.assertTrue(torch.all(cache_kjt.lengths() == global_kjt.lengths()))
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 2, 1, 0, 1, 2]))
        )

    def testFeatureNames(self):
        configs = [
            EmbeddingBagConfig(
                num_embeddings=8, embedding_dim=32, feature_names=["A", "B"]
            ),
            EmbeddingBagConfig(name="C", num_embeddings=8, embedding_dim=32),
        ]
        transformer_collection = IDTransformerCollection(configs)
        global_kjt = KeyedJaggedTensor(
            keys=["A", "C", "B"],
            values=torch.tensor([1, 3, 2, 3, 4, 3, 2, 3, 2, 1]),
            lengths=torch.tensor([4, 3, 3]),
        )
        cache_kjt = transformer_collection.transform(global_kjt)
        self.assertEqual(cache_kjt.keys(), global_kjt.keys())
        self.assertTrue(torch.all(cache_kjt.lengths() == global_kjt.lengths()))
        self.assertTrue(
            torch.all(
                cache_kjt.values() == torch.tensor([0, 1, 2, 1, 0, 1, 2, 1, 2, 0])
            )
        )
