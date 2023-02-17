#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, Type

import torch
from torchrec.metrics.ctr import CTRMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)


class TestCTRMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        ctr_num = torch.sum(labels * weights)
        ctr_denom = torch.sum(weights)
        num_samples = torch.tensor(labels.size()[0]).double()
        return {"ctr_num": ctr_num, "ctr_denom": ctr_denom, "num_samples": num_samples}

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.where(
            states["ctr_denom"] == 0.0, 0.0, states["ctr_num"] / states["ctr_denom"]
        ).double()


WORLD_SIZE = 4


class CTRMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = CTRMetric
    task_name: str = "ctr"

    def test_unfused_ctr(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CTRMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestCTRMetric,
            metric_name=CTRMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_fused_ctr(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CTRMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestCTRMetric,
            metric_name=CTRMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )
