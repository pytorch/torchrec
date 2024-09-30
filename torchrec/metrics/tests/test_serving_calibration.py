#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest
from typing import Dict, Type

import torch
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.serving_calibration import ServingCalibrationMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)


class TestServingCalibrationMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        calibration_num = torch.sum(predictions * weights)
        calibration_denom = torch.sum(labels * weights)
        num_samples = torch.tensor(labels.size()[0]).double()
        return {
            "calibration_num": calibration_num,
            "calibration_denom": calibration_denom,
            "num_samples": num_samples,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.where(
            states["calibration_denom"] <= 0.0,
            0.0,
            states["calibration_num"] / states["calibration_denom"],
        ).double()


WORLD_SIZE = 4


class ServingCalibrationMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = ServingCalibrationMetric
    task_name: str = "calibration"

    def test_unfused_calibration(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=ServingCalibrationMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestServingCalibrationMetric,
            metric_name=ServingCalibrationMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_fused_calibration(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=ServingCalibrationMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestServingCalibrationMetric,
            metric_name=ServingCalibrationMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )
