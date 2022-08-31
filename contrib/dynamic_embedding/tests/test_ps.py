import os
import unittest

import torch
from torchrec_dynamic_embedding import PS
from utils import register_memory_io


register_memory_io()


class TestPS(unittest.TestCase):
    def testEvictFetch(self):
        cache_ids = [0, 2, 4, 8]
        ids = torch.tensor([[100, 0], [101, 2], [102, 4], [103, 8]], dtype=torch.long)
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://")
        ps.evict(ids)
        tensor[:, :] = 0
        ps.fetch(ids, 0).wait()
        self.assertTrue(torch.allclose(tensor[cache_ids], origin_tensor[cache_ids]))

    def testOS(self):
        cache_ids = [1, 3, 6]
        ids = torch.tensor([[100, 1], [101, 3], [102, 6]], dtype=torch.long)
        tensor = torch.rand((10, 4))
        optim1 = torch.rand((10, 4))
        optim2 = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        origin_optim1 = optim1.clone()
        origin_optim2 = optim2.clone()
        ps = PS("table", [tensor, optim1, optim2], "memory://")
        ps.evict(ids)
        tensor[:, :] = 0
        optim1[:, :] = 0
        optim2[:, :] = 0
        ps.fetch(ids, 0).wait()
        self.assertTrue(torch.allclose(tensor[cache_ids], origin_tensor[cache_ids]))
        self.assertTrue(torch.allclose(optim1[cache_ids], origin_optim1[cache_ids]))
        self.assertTrue(torch.allclose(optim2[cache_ids], origin_optim2[cache_ids]))

    def testFetchToDifferentCacheID(self):
        cache_ids = [0, 2, 4, 8]
        evict_ids = torch.tensor(
            [[100, 0], [101, 2], [102, 4], [103, 8]], dtype=torch.long
        )
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://")
        ps.evict(evict_ids)
        tensor[:, :] = 0
        new_cache_ids = [1, 3, 5, 7]
        fetch_ids = torch.tensor(
            [[100, 1], [101, 3], [102, 5], [103, 7]], dtype=torch.long
        )
        ps.fetch(fetch_ids, 0).wait()
        self.assertTrue(torch.allclose(tensor[new_cache_ids], origin_tensor[cache_ids]))

    def testFetchNonExist(self):
        cache_ids = [0, 2, 4]
        evict_ids = torch.tensor([[100, 0], [101, 2], [102, 4]], dtype=torch.long)
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://")
        ps.evict(evict_ids)
        tensor[:, :] = 0
        addition_cache_ids = [3, 9]
        additional_fetch_ids = torch.tensor([[103, 3], [104, 9]], dtype=torch.long)
        ps.fetch(torch.cat([evict_ids, additional_fetch_ids]), 0).wait()
        self.assertTrue(torch.allclose(tensor[cache_ids], origin_tensor[cache_ids]))
        self.assertTrue(
            torch.allclose(
                tensor[addition_cache_ids],
                torch.zeros_like(tensor[addition_cache_ids]),
            )
        )


if __name__ == "__main__":
    unittest.main()
