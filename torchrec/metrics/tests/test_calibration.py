#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import Dict, List, Type

import torch
import torch.distributed as dist
from torchrec.metrics.calibration import (
    CalibrationMetric,
)
from torchrec.metrics.rec_metric import (
    RecComputeMode,
    RecMetric,
)
from torchrec.metrics.tests.test_utils import (
    TestMetric,
    rec_metric_value_test_helper,
    rec_metric_value_test_launcher,
)


class TestCalibrationMetric(TestMetric):
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


class CalibrationMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = CalibrationMetric
    task_name: str = "calibration"

    @staticmethod
    def _test_calibration(
        target_clazz: Type[RecMetric],
        target_compute_mode: RecComputeMode,
        task_names: List[str],
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
    ) -> None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="gloo",
            world_size=world_size,
            rank=rank,
        )

        calibration_metrics, test_metrics = rec_metric_value_test_helper(
            target_clazz=target_clazz,
            target_compute_mode=target_compute_mode,
            test_clazz=TestCalibrationMetric,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=False,
            world_size=world_size,
            my_rank=rank,
            task_names=task_names,
        )

        if rank == 0:
            for name in task_names:
                assert torch.allclose(
                    calibration_metrics[f"calibration-{name}|lifetime_calibration"],
                    test_metrics[0][name],
                )
                assert torch.allclose(
                    calibration_metrics[f"calibration-{name}|window_calibration"],
                    test_metrics[1][name],
                )
                assert torch.allclose(
                    calibration_metrics[
                        f"calibration-{name}|local_lifetime_calibration"
                    ],
                    test_metrics[2][name],
                )
                assert torch.allclose(
                    calibration_metrics[f"calibration-{name}|local_window_calibration"],
                    test_metrics[3][name],
                )
        dist.destroy_process_group()

    def test_unfused_calibration(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CalibrationMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestCalibrationMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_calibration,
        )

    def test_fused_calibration(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CalibrationMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestCalibrationMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_calibration,
        )
