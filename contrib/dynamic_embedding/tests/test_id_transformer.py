import unittest

import torch
import torchrec_dynamic_embedding
from torchrec_dynamic_embedding import IDTransformer


class PythonIdTransformer:
    def __init__(self, num_embedding, num_threads):
        self._num_embedding = num_embedding
        self._num_threads = num_threads
        self.dict = [{} for _ in range(num_threads)]
        embedding_per_mapper = num_embedding // num_threads
        self._mapper_start = [i * embedding_per_mapper for i in range(num_threads)] + [
            num_embedding
        ]
        self.overflow = False

    def transform(self, global_ids: torch.Tensor):
        global_id_list = global_ids.flatten().tolist()
        cache_id_list = [0] * len(global_id_list)
        for i in range(len(global_id_list)):
            gid = global_id_list[i]
            mapper_id = gid % self._num_threads
            if gid in self.dict[mapper_id]:
                cid = self.dict[mapper_id][gid]
            else:
                cid = len(self.dict[mapper_id]) + self._mapper_start[mapper_id]
                self.dict[mapper_id][gid] = cid
                if (
                    len(self.dict[mapper_id])
                    >= self._mapper_start[mapper_id + 1] - self._mapper_start[mapper_id]
                ):
                    self.overflow = True
            cache_id_list[i] = cid
        cache_ids = torch.tensor(cache_id_list, dtype=torch.long)
        return cache_ids


class TestIDTransformer(unittest.TestCase):
    def testSkeleton(self):
        self.assertIsNotNone(IDTransformer(1024))

    def testTransform(self):
        num_embedding = 1024
        num_threads = 4
        shape = (1024,)
        transformer = IDTransformer(
            num_embedding,
            transform_config={
                "type": "thread",
                "underlying": {"type": "naive"},
                "num_threads": num_threads,
            },
        )
        global_ids = torch.empty(shape, dtype=torch.int64)
        global_ids.random_(0, 512)

        cache_ids = torch.empty_like(global_ids)
        num_transformed, ids_to_fetch = transformer.transform(global_ids, cache_ids)
        self.assertEqual(num_transformed, global_ids.numel())

        python_transformer = PythonIdTransformer(num_embedding, num_threads)
        python_cache_ids = python_transformer.transform(global_ids)
        self.assertTrue(torch.all(cache_ids == python_cache_ids))

        ids_map = {global_id: cache_id for global_id, cache_id in ids_to_fetch.tolist()}
        for global_id, cache_id in zip(global_ids.tolist(), cache_ids.tolist()):
            self.assertTrue(global_id in ids_map)
            self.assertEqual(cache_id, ids_map[global_id])

    def testEvict(self):
        num_embedding = 9
        num_threads = 2
        transformer = IDTransformer(
            num_embedding,
            transform_config={
                "type": "thread",
                "underlying": {"type": "naive"},
                "num_threads": num_threads,
            },
        )
        global_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        cache_ids = torch.empty_like(global_ids)
        num_transformed, _ = transformer.transform(global_ids, cache_ids)
        self.assertEqual(num_transformed, global_ids.numel())

        global_ids = torch.tensor([1, 3, 5, 7], dtype=torch.long)
        num_transformed, _ = transformer.transform(global_ids, cache_ids)
        self.assertEqual(num_transformed, global_ids.numel())

        num_to_evict = 2
        evicted_tensor = transformer.evict(num_to_evict)
        self.assertEqual(num_to_evict, evicted_tensor.shape[0])
        evicted_ids = sorted(evicted_tensor.tolist())
        self.assertEqual(evicted_ids, [[2, 0], [4, 1]])
