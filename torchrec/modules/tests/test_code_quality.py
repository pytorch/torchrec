#!/usr/bin/env python3

import inspect
import sys
import unittest

import torch
import torchrec  # noqa
from torchrec.linter.module_linter import MAX_NUM_ARGS_IN_MODULE_CTOR


class CodeQualityTest(unittest.TestCase):
    def test_num_ctor_args(self) -> None:
        classes = inspect.getmembers(sys.modules["torchrec"], inspect.isclass)
        for class_name, clazz in classes:
            if issubclass(clazz, torch.nn.Module):
                num_args_excluding_self = (
                    len(inspect.getfullargspec(clazz.__init__).args) - 1
                )
                self.assertLessEqual(
                    num_args_excluding_self,
                    MAX_NUM_ARGS_IN_MODULE_CTOR,
                    "Modules in TorchRec can have no more than {} constructor args, but {} has {}.".format(
                        MAX_NUM_ARGS_IN_MODULE_CTOR, class_name, num_args_excluding_self
                    ),
                )


if __name__ == "__main__":
    unittest.main()
