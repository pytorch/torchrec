#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.test_model import ModelInput, TestNegSamplingModule

from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.utils import (
    _get_node_args,
    _rewrite_model,
    PipelinedForward,
    TrainPipelineContext,
)
from torchrec.distributed.types import ShardingType

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TrainPipelineUtilsTest(TrainPipelineSparseDistTestBase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_rewrite_model(self) -> None:
        sharding_type = ShardingType.TABLE_WISE.value
        kernel_type = EmbeddingComputeKernel.FUSED.value
        fused_params = {}

        extra_input = ModelInput.generate(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            batch_size=10,
            world_size=1,
            num_float_features=10,
            randomize_indices=False,
        )[0].to(self.device)

        preproc_module = TestNegSamplingModule(
            extra_input=extra_input,
        )
        model = self._setup_model(preproc_module=preproc_module)

        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type, fused_params
        )

        # Try to rewrite model without ignored_preproc_modules defined, EBC forwards not overwritten to PipelinedForward due to KJT modification
        _rewrite_model(
            model=sharded_model,
            batch=None,
            context=TrainPipelineContext(),
            dist_stream=None,
        )
        self.assertNotIsInstance(
            sharded_model.module.sparse.ebc.forward, PipelinedForward
        )
        self.assertNotIsInstance(
            sharded_model.module.sparse.weighted_ebc.forward, PipelinedForward
        )

        # Now provide preproc module explicitly
        _rewrite_model(
            model=sharded_model,
            batch=None,
            context=TrainPipelineContext(),
            dist_stream=None,
            pipeline_preproc=True,
        )
        self.assertIsInstance(sharded_model.module.sparse.ebc.forward, PipelinedForward)
        self.assertIsInstance(
            sharded_model.module.sparse.weighted_ebc.forward, PipelinedForward
        )
        self.assertEqual(
            sharded_model.module.sparse.ebc.forward._args[0].preproc_modules[0],
            sharded_model.module.preproc_module,
        )
        self.assertEqual(
            sharded_model.module.sparse.weighted_ebc.forward._args[0].preproc_modules[
                0
            ],
            sharded_model.module.preproc_module,
        )


class TestUtils(unittest.TestCase):
    def test_get_node_args_helper_call_module_kjt(self) -> None:
        graph = torch.fx.Graph()
        kjt_args = []

        kjt_args.append(
            torch.fx.Node(graph, "values", "placeholder", "torch.Tensor", (), {})
        )
        kjt_args.append(
            torch.fx.Node(graph, "lengths", "placeholder", "torch.Tensor", (), {})
        )
        kjt_args.append(
            torch.fx.Node(
                graph, "weights", "call_module", "PositionWeightedModule", (), {}
            )
        )

        kjt_node = torch.fx.Node(
            graph,
            "keyed_jagged_tensor",
            "call_function",
            KeyedJaggedTensor,
            tuple(kjt_args),
            {},
        )

        num_found = 0
        _, num_found = _get_node_args(
            MagicMock(), kjt_node, set(), TrainPipelineContext(), False
        )

        # Weights is call_module node, so we should only find 2 args unmodified
        self.assertEqual(num_found, len(kjt_args) - 1)
