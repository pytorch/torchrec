#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, Type

import torch
from torchrec.metrics.mae import compute_mae, MAEMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)


class TestMAEMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        error_sum = torch.sum(weights * torch.abs(labels - predictions))
        return {
            "error_sum": error_sum,
            "weighted_num_samples": torch.sum(weights),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_mae(
            states["error_sum"],
            states["weighted_num_samples"],
        )


WORLD_SIZE = 4


class MAEMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = MAEMetric
    task_name: str = "mae"

    def test_unfused_mae(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MAEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMAEMetric,
            metric_name="mae",
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_fused_mae(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MAEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestMAEMetric,
            metric_name="mae",
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )
