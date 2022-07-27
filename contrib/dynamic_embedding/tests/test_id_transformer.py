import unittest

import torch
import torchrec_dynamic_embedding


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
        self.assertIsNotNone(torch.classes.tde.IDTransformer(1024, 4))

    def testTransform(self):
        num_embedding = 1024
        num_threads = 4
        shape = (1024,)
        transformer = torch.classes.tde.IDTransformer(num_embedding, num_threads)
        global_ids = torch.empty(shape, dtype=torch.int64)
        global_ids.random_(0, 512)

        cache_ids = torch.empty_like(global_ids)
        num_transformed = transformer.transform(global_ids, cache_ids)
        self.assertEqual(num_transformed, global_ids.numel())

        python_transformer = PythonIdTransformer(num_embedding, num_threads)
        python_cache_ids = python_transformer.transform(global_ids)
        self.assertTrue(torch.all(cache_ids == python_cache_ids))
