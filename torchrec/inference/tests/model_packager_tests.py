#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch.nn as nn
from torch.package import PackageExporter
from torchrec.inference.model_packager import PredictFactoryPackager
from torchrec.inference.modules import PredictFactory


class TestPredictFactory(PredictFactory):
    def create_predict_module(self) -> nn.Module:
        return nn.Module()


class TestPredictFactoryPackager(PredictFactoryPackager):
    @classmethod
    def set_extern_modules(cls, pe: PackageExporter) -> None:
        pe.extern(["numpy"])

    @classmethod
    def set_mocked_modules(cls, pe: PackageExporter) -> None:
        pe.mock(["fbgemm_gpu"])


class ModelPackagerTest(unittest.TestCase):
    def test_model_packager(self) -> None:
        with tempfile.NamedTemporaryFile() as file:
            output = file.name
            TestPredictFactoryPackager.save_predict_factory(
                TestPredictFactory, {"model_name": "sparsenn"}, output
            )
