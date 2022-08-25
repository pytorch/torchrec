import os
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec_dynamic_embedding import get_ps, IDTransformerCollection
from utils import init_dist, register_memory_io


register_memory_io()


class TestPSCollection(unittest.TestCase):
    def testExtractTensor(self):
        init_dist()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        class Model(nn.Module):
            def __init__(self, configs):
                super().__init__()
                self.emb = EmbeddingCollection(
                    tables=configs, device=torch.device("meta")
                )
                # this is incorrect, but doesn't matter.
                self.dense = nn.Linear(10, 10)

            def forward(self, x):
                return x

        configs = [
            EmbeddingConfig(
                name="AB", num_embeddings=4, embedding_dim=32, feature_names=["A", "B"]
            ),
            EmbeddingConfig(name="C", num_embeddings=8, embedding_dim=32),
        ]
        model = Model(configs=configs)
        model = DMP(module=model, device=device)
        ps_dict = get_ps(model, 0, "memory://")
        self.assertEqual(len(ps_dict), 1)
        self.assertTrue("emb" in ps_dict)
        ps_collection = ps_dict["emb"]
        keys = ps_collection.keys()
        self.assertEqual(len(keys), 2)
        self.assertTrue("AB" in keys)
        self.assertTrue("C" in keys)

    def testModuleEviction(self):
        init_dist()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        embedding_dim = 8
        configs = [
            EmbeddingConfig(
                name="AB",
                num_embeddings=4,
                embedding_dim=embedding_dim,
                feature_names=["A", "B"],
                # to check reinit
                weight_init_min=2,
                weight_init_max=2,
            ),
        ]
        model = EmbeddingCollection(tables=configs, device=torch.device("meta"))
        model = DMP(module=model, device=device)
        ps_dict = get_ps(model, 0, "memory://")
        transformer_collection = IDTransformerCollection(
            configs, ps_collection=ps_dict[""]
        )

        # extract weight manually
        weight = (
            model.module.state_dict()["embeddings.AB.weight"].local_shards()[0].tensor
        )
        self.assertTrue(weight.shape[0] == 4 and weight.shape[1] == embedding_dim)

        # init to zero
        weight[:] = torch.zeros_like(weight)

        global_kjt_1 = KeyedJaggedTensor(
            keys=["A", "B"],
            values=torch.tensor([1, 2, 3, 4, 1, 2]),
            lengths=torch.tensor([4, 2]),
        )
        cache_kjt = transformer_collection.transform(global_kjt_1)
        embedding = model(cache_kjt.to(device))
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 2, 3, 0, 1]))
        )
        self.assertTrue(
            torch.allclose(
                embedding["A"].values(), torch.zeros_like(embedding["A"].values())
            )
        )
        self.assertTrue(
            torch.allclose(
                embedding["B"].values(), torch.zeros_like(embedding["B"].values())
            )
        )

        global_kjt_2 = KeyedJaggedTensor(
            keys=["A", "B"],
            values=torch.tensor([1, 2, 1, 5, 5, 6]),
            lengths=torch.tensor([3, 3]),
        )

        # this will evict 3 and 4
        cache_kjt = transformer_collection.transform(global_kjt_2)
        embedding = model(cache_kjt.to(device))
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 0, 2, 2, 3]))
        )
        self.assertTrue(
            torch.allclose(
                embedding["A"].values(), torch.zeros_like(embedding["A"].values())
            )
        )
        self.assertTrue(
            torch.allclose(
                embedding["B"].values(), 2 * torch.ones_like(embedding["B"].values())
            )
        )

        # update to one
        weight[:] = torch.ones_like(weight)
        global_kjt_3 = KeyedJaggedTensor(
            keys=["A", "B"],
            values=torch.tensor([1, 2, 1, 4, 3, 4]),
            lengths=torch.tensor([3, 3]),
        )
        cache_kjt = transformer_collection.transform(global_kjt_3)
        embedding = model(cache_kjt.to(device))
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 0, 2, 3, 2]))
        )
        self.assertTrue(
            torch.allclose(
                embedding["A"].values(), torch.ones_like(embedding["A"].values())
            )
        )
        self.assertTrue(
            torch.allclose(
                embedding["B"].values(), torch.zeros_like(embedding["B"].values())
            )
        )


if __name__ == "__main__":
    unittest.main()
