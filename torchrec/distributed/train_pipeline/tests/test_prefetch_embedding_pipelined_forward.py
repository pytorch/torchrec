#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

from torchrec.distributed.train_pipeline.pipeline_context import (
    PrefetchTrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import (
    PrefetchEmbeddingPipelinedForward,
    PrefetchPipelinedForward,
)
from torchrec.distributed.train_pipeline.types import CallArgs


class TestPrefetchEmbeddingPipelinedForward(unittest.TestCase):
    """Test PrefetchEmbeddingPipelinedForward key functionality"""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_module = MagicMock()
        self.prefetch_context = PrefetchTrainPipelineContext()
        self.mock_args = CallArgs(args=[], kwargs={})

    def test_prefetch_returns_true(self) -> None:
        """Test that prefetch() returns True."""
        forward = PrefetchEmbeddingPipelinedForward(
            name="test_prefetch",
            args=self.mock_args,
            module=self.mock_module,
            context=self.prefetch_context,
        )

        # Test that prefetch returns True
        self.assertIsInstance(forward, PrefetchPipelinedForward)

    def test_call_fails_without_compute_and_output_dist(self) -> None:
        """Test that __call__ fails if compute_and_output_dist is not called first."""
        forward = PrefetchEmbeddingPipelinedForward(
            name="test_call_error",
            args=self.mock_args,
            module=self.mock_module,
            context=self.prefetch_context,
        )

        # Should raise exception when called without compute_and_output_dist
        with self.assertRaises(Exception) as context:
            forward()

        self.assertIn(
            "compute_and_output_dist must be called before __call__",
            str(context.exception),
        )

    def test_call_succeeds_after_compute_and_output_dist(self) -> None:
        """Test that __call__ succeeds when compute_and_output_dist is called first."""
        forward = PrefetchEmbeddingPipelinedForward(
            name="test_call_success",
            args=self.mock_args,
            module=self.mock_module,
            context=self.prefetch_context,
        )

        # Set up mock data in context
        test_data = MagicMock()
        test_ctx = MagicMock()
        self.prefetch_context.module_input_post_prefetch = {
            "test_call_success": test_data
        }
        self.prefetch_context.module_contexts_post_prefetch = {
            "test_call_success": test_ctx
        }

        # Mock the module's compute_and_output_dist method
        mock_awaitable = MagicMock()
        self.mock_module.compute_and_output_dist.return_value = mock_awaitable

        # Call compute_and_output_dist first
        forward.compute_and_output_dist()

        # Now __call__ should succeed and return the awaitable
        result = forward()
        self.assertEqual(result, mock_awaitable)


if __name__ == "__main__":
    unittest.main()
