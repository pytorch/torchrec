#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.metrics.metrics_config import (
    DefaultTaskInfo,
)
from torchrec.metrics.model_utils import (
    parse_task_model_outputs,
)
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.rec_metric import RecComputeMode
from torchrec.metrics.tests.test_utils import gen_test_batch


class RecMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        # Create testing labels, predictions and weights
        model_output = gen_test_batch(128)
        model_output["weight"] = torch.ones_like(model_output["label"])
        labels, predictions, weights = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        self.labels, self.predictions, self.weights = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )

    def test_optional_weight(self) -> None:
        ne1 = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=128,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        ne2 = NEMetric(
            world_size=1,
            my_rank=1,
            batch_size=128,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )

        model_output = gen_test_batch(128)
        model_output["weight"] = torch.ones_like(model_output["label"])
        labels, predictions, weights = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        ne1.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        ne2.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=None,
        )
        ne1 = ne1._metrics_computations[0]
        ne2 = ne2._metrics_computations[0]
        self.assertTrue(ne1.cross_entropy_sum == ne2.cross_entropy_sum)
        self.assertTrue(ne1.weighted_num_samples == ne2.weighted_num_samples)
        self.assertTrue(ne1.pos_labels == ne2.pos_labels)
        self.assertTrue(ne1.neg_labels == ne2.neg_labels)

    def test_compute(self) -> None:
        # Rank 0 does computation.
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=128,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        res = ne.compute()
        self.assertIn("ne-DefaultTask|lifetime_ne", res)
        self.assertIn("ne-DefaultTask|window_ne", res)

        # Rank non-zero skip computation.
        ne = NEMetric(
            world_size=1,
            my_rank=1,
            batch_size=128,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        res = ne.compute()
        self.assertEqual({}, res)

        # Rank non-zero does computation if `compute_on_all_ranks` enabled.
        ne = NEMetric(
            world_size=1,
            my_rank=1,
            batch_size=128,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
            compute_on_all_ranks=True,
        )
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        res = ne.compute()
        self.assertIn("ne-DefaultTask|lifetime_ne", res)
        self.assertIn("ne-DefaultTask|window_ne", res)
