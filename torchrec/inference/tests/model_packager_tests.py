#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import tempfile
import unittest

import torch.nn as nn
from torch.package import PackageExporter, PackageImporter
from torchrec.inference.model_packager import PredictFactoryPackager
from torchrec.inference.modules import PredictFactory


class TestPredictFactory(PredictFactory):
    def create_predict_module(self) -> nn.Module:
        return nn.Module()


class TestPredictFactoryPackager(PredictFactoryPackager):
    @classmethod
    def set_extern_modules(cls, pe: PackageExporter) -> None:
        pe.extern(["io", "numpy"])

    @classmethod
    def set_mocked_modules(cls, pe: PackageExporter) -> None:
        pe.mock(["fbgemm_gpu"])


class ModelPackagerTest(unittest.TestCase):
    def test_model_packager(self) -> None:
        with tempfile.NamedTemporaryFile() as file:
            output = file.name
            TestPredictFactoryPackager.save_predict_factory(
                predict_factory=TestPredictFactory,
                configs={"model_name": "sparsenn"},
                output=output,
                extra_files={"metadata": "test"},
            )

    def test_model_packager_unpack(self) -> None:
        buf = io.BytesIO()
        TestPredictFactoryPackager.save_predict_factory(
            predict_factory=TestPredictFactory,
            configs={"model_name": "sparsenn"},
            output=buf,
            extra_files={
                "dummy_text_field": "test",
                "dummy_binary_field": b"abc",
            },
        )

        buf.seek(0)

        pi = PackageImporter(buf)
        dummy_text_value = pi.load_text("extra_files", "dummy_text_field")
        self.assertEqual("test", dummy_text_value)
        dummy_binary_value = pi.load_binary("extra_files", "dummy_binary_field")
        self.assertEqual(b"abc", dummy_binary_value)
