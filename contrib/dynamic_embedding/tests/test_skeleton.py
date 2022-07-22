import unittest

import torch
import torchrec_dynamic_embedding


class TestSkeleton(unittest.TestCase):
    def testSkeleton(self):
        self.assertEqual(torch.ops.tde.foo(), 42)
        self.assertIsNotNone(torch.classes.tde.IDTransformer(10))
