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
from torchrec.metrics.mae import compute_mae, MAEMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    rec_metric_value_test_helper,
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

    @staticmethod
    def _test_mae(
        target_clazz: Type[RecMetric],
        target_compute_mode: RecComputeMode,
        task_names: List[str],
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
    ) -> None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="gloo",
            world_size=world_size,
            rank=rank,
        )

        mae_metrics, test_metrics = rec_metric_value_test_helper(
            target_clazz=target_clazz,
            target_compute_mode=target_compute_mode,
            test_clazz=TestMAEMetric,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=False,
            should_validate_update=should_validate_update,
            world_size=world_size,
            my_rank=rank,
            task_names=task_names,
        )

        if rank == 0:
            for name in task_names:
                assert torch.allclose(
                    mae_metrics[f"mae-{name}|lifetime_mae"], test_metrics[0][name]
                )
                assert torch.allclose(
                    mae_metrics[f"mae-{name}|window_mae"], test_metrics[1][name]
                )
                assert torch.allclose(
                    mae_metrics[f"mae-{name}|local_lifetime_mae"], test_metrics[2][name]
                )
                assert torch.allclose(
                    mae_metrics[f"mae-{name}|local_window_mae"], test_metrics[3][name]
                )
        dist.destroy_process_group()

    def test_unfused_mae(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MAEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMAEMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_mae,
        )

    def test_fused_mae(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MAEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestMAEMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_mae,
        )
