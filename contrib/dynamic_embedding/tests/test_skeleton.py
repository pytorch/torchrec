import torchrec_dynamic_embedding
import unittest
import torch


class TestSkeleton(unittest.TestCase):
    def testSkeleton(self):
        self.assertEqual(torch.ops.tde.foo(), 42)
        self.assertIsNotNone(torch.classes.tde.IDTransformer(10))
